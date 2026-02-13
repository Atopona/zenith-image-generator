import { Errors } from '@z-image/shared'
import type { Context } from 'hono'
import type { OpenAIChatContentPart, OpenAIChatMessage } from '@z-image/shared'
import {
  ensureCustomChannelsInitialized,
  getChannel,
  getImageChannel,
  getLLMChannel,
} from '../channels'
import { parseTokens, runWithTokenRotation } from '../core/token-manager'
import { sendError } from '../middleware'
import { parseSize } from './adapter'
import { isKnownImageModel, resolveModel } from './model-resolver'
import type { OpenAIChatRequest, OpenAIChatResponse } from './types'

/** Extract the text portion from a message content (string or multimodal parts) */
function getTextContent(content: string | OpenAIChatContentPart[]): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
    .map((p) => p.text)
    .join('\n')
}

/** Extract the first image URL from multimodal content parts */
function getImageUrlFromContent(
  content: string | OpenAIChatContentPart[],
): string | undefined {
  if (typeof content === 'string') return undefined
  if (!Array.isArray(content)) return undefined
  const imgPart = content.find(
    (p): p is { type: 'image_url'; image_url: { url: string } } =>
      p.type === 'image_url',
  )
  return imgPart?.image_url?.url
}

function parseChatBearerToken(authHeader?: string): {
  providerHint?: string
  token?: string
} {
  if (!authHeader) return {}
  if (!authHeader.startsWith('Bearer ')) return {}

  const raw = authHeader.slice('Bearer '.length).trim()
  if (!raw) return {}

  if (raw.startsWith('gitee:')) {
    const token = raw.slice('gitee:'.length).trim()
    return token ? { providerHint: 'gitee', token } : {}
  }
  if (raw.startsWith('ms:')) {
    const token = raw.slice('ms:'.length).trim()
    return token ? { providerHint: 'modelscope', token } : {}
  }
  if (raw.startsWith('hf:')) {
    const token = raw.slice('hf:'.length).trim()
    return token ? { providerHint: 'huggingface', token } : {}
  }
  if (raw.startsWith('deepseek:')) {
    const token = raw.slice('deepseek:'.length).trim()
    return token ? { providerHint: 'deepseek', token } : {}
  }
  if (raw.startsWith('a4f:')) {
    const token = raw.slice('a4f:'.length).trim()
    return token ? { providerHint: 'a4f', token } : {}
  }

  return { token: raw }
}

// -------- Image-via-Chat helpers --------

const IMAGE_MODEL_PREFIX = 'image/'

/**
 * Try to resolve a model as an image model for chat-based image generation.
 * Supports both explicit `image/` prefix and bare image model names.
 *
 * Returns null if the model is not an image model (i.e. the resolved channel
 * has no image capability), so the caller can fall through to normal LLM chat.
 */
function tryResolveImageModel(model: string): { channelId: string; model: string } | null {
  const trimmed = model.trim()

  // Strip explicit image/ prefix if present
  const inner = trimmed.startsWith(IMAGE_MODEL_PREFIX)
    ? trimmed.slice(IMAGE_MODEL_PREFIX.length)
    : trimmed

  // Support custom/ prefix
  if (inner.startsWith('custom/')) {
    const rest = inner.slice('custom/'.length)
    const firstSlash = rest.indexOf('/')
    if (firstSlash > 0) {
      const channelId = rest.slice(0, firstSlash).trim()
      const m = rest.slice(firstSlash + 1).trim()
      if (channelId) {
        // Only treat as image if the channel actually has image capability
        const ch = getImageChannel(channelId)
        if (ch) return { channelId, model: m }
        // If explicit image/ prefix was used, still treat as image (will error later)
        if (trimmed.startsWith(IMAGE_MODEL_PREFIX)) return { channelId, model: m }
      }
    }
    return null
  }

  const resolved = resolveModel(inner || undefined)
  const imageCapability = getImageChannel(resolved.provider)

  if (trimmed.startsWith(IMAGE_MODEL_PREFIX)) {
    // Explicit image/ prefix — always treat as image request
    return { channelId: resolved.provider, model: resolved.model }
  }

  if (!imageCapability) return null

  // For channels that have BOTH image and LLM (e.g. gitee), only auto-detect
  // if the model name matches a known image model (not a LLM model).
  const channel = getChannel(resolved.provider)
  if (channel?.llm) {
    // Channel has LLM too — check via model-resolver's known image model list
    return isKnownImageModel(inner) ? { channelId: resolved.provider, model: resolved.model } : null
  }

  // Channel only has image capability — auto-detect
  return { channelId: resolved.provider, model: resolved.model }
}

/** Extract --size, --negative, and --image from user prompt, return cleaned prompt + params */
function parseImageParams(raw: string): {
  prompt: string
  size?: string
  negativePrompt?: string
  sourceImageUrl?: string
} {
  let size: string | undefined
  let negativePrompt: string | undefined
  let sourceImageUrl: string | undefined

  let text = raw.replace(/--size\s+(\d+x\d+)/i, (_, s) => {
    size = s
    return ''
  })
  text = text.replace(/--negative\s+"([^"]+)"/i, (_, n) => {
    negativePrompt = n
    return ''
  })
  text = text.replace(/--negative\s+(\S+)/i, (_, n) => {
    negativePrompt = n
    return ''
  })
  text = text.replace(/--image\s+(\S+)/i, (_, u) => {
    sourceImageUrl = u
    return ''
  })

  return { prompt: text.trim(), size, negativePrompt, sourceImageUrl }
}

// -------- LLM Chat helpers --------

function resolveChatModel(model: string): {
  channelId: string
  model: string
  forceAnonymous?: boolean
} {
  const trimmed = model.trim()
  if (!trimmed) return { channelId: 'huggingface', model: 'openai-fast', forceAnonymous: true }

  if (trimmed.startsWith('custom/')) {
    const rest = trimmed.slice('custom/'.length)
    const firstSlash = rest.indexOf('/')
    if (firstSlash > 0) {
      const channelId = rest.slice(0, firstSlash).trim()
      const m = rest.slice(firstSlash + 1).trim()
      if (channelId) return { channelId, model: m }
    }
  }

  if (trimmed.startsWith('gitee/'))
    return { channelId: 'gitee', model: trimmed.slice('gitee/'.length) }
  if (trimmed.startsWith('ms/'))
    return { channelId: 'modelscope', model: trimmed.slice('ms/'.length) }
  if (trimmed.startsWith('hf/'))
    return { channelId: 'huggingface', model: trimmed.slice('hf/'.length) }
  if (trimmed.startsWith('deepseek/'))
    return { channelId: 'deepseek', model: trimmed.slice('deepseek/'.length) }
  if (trimmed.startsWith('pollinations/'))
    return {
      channelId: 'huggingface',
      model: trimmed.slice('pollinations/'.length),
      forceAnonymous: true,
    }
  if (trimmed.startsWith('a4f/')) return { channelId: 'a4f', model: trimmed.slice('a4f/'.length) }

  // Default: treat as Pollinations model id.
  return { channelId: 'huggingface', model: trimmed, forceAnonymous: true }
}

function getSystemPrompt(messages: OpenAIChatRequest['messages']): string {
  return messages
    .filter((m) => m.role === 'system')
    .map((m) => getTextContent(m.content))
    .join('\n')
    .trim()
}

function getUserPrompt(messages: OpenAIChatRequest['messages']): string | null {
  const parts = messages
    .filter((m) => m.role === 'user')
    .map((m) => getTextContent(m.content))
    .filter((s) => s && s.trim().length > 0)
  if (parts.length === 0) return null
  return parts.join('\n').trim()
}

/** Get only the last user message — used for image generation to avoid history pollution. */
function getLastUserPrompt(messages: OpenAIChatRequest['messages']): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') {
      const text = getTextContent(messages[i].content).trim()
      if (text) return text
    }
  }
  return null
}

/** Get the first image URL from the last user message (for multimodal image editing) */
function getLastUserImageUrl(messages: OpenAIChatRequest['messages']): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') {
      const url = getImageUrlFromContent(messages[i].content)
      if (url) return url
    }
  }
  return undefined
}

function makeChatResponse(model: string, content: string): OpenAIChatResponse {
  const id = `chatcmpl-${Math.random().toString(36).slice(2)}`
  return {
    id,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content },
        finish_reason: 'stop',
      },
    ],
  }
}

/** Return a streaming SSE response compatible with OpenAI chat completions. */
function streamChatResponse(c: Context, model: string, content: string) {
  const id = `chatcmpl-${Math.random().toString(36).slice(2)}`
  const created = Math.floor(Date.now() / 1000)

  const chunk = {
    id,
    object: 'chat.completion.chunk',
    created,
    model,
    choices: [{ index: 0, delta: { role: 'assistant', content }, finish_reason: null }],
  }
  const doneChunk = {
    id,
    object: 'chat.completion.chunk',
    created,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
  }

  const body = `data: ${JSON.stringify(chunk)}\n\ndata: ${JSON.stringify(doneChunk)}\n\ndata: [DONE]\n\n`

  return c.body(body, 200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  })
}

/** Return either streaming or non-streaming chat response based on request. */
function sendChatResult(c: Context, model: string, content: string, stream?: boolean) {
  if (stream) return streamChatResponse(c, model, content)
  return c.json(makeChatResponse(model, content))
}

export async function handleChatCompletion(c: Context) {
  ensureCustomChannelsInitialized(
    c.env as unknown as Record<string, string | undefined> | undefined
  )

  let body: OpenAIChatRequest
  try {
    body = (await c.req.json()) as OpenAIChatRequest
  } catch {
    return sendError(c, Errors.invalidParams('body', 'Invalid JSON body'))
  }

  if (!body?.model || typeof body.model !== 'string') {
    return sendError(c, Errors.invalidParams('model', 'model is required'))
  }

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return sendError(c, Errors.invalidParams('messages', 'messages is required'))
  }

  const userPrompt = getUserPrompt(body.messages)
  if (!userPrompt) {
    return sendError(c, Errors.invalidParams('messages', 'At least one user message is required'))
  }

  const auth = parseChatBearerToken(c.req.header('Authorization'))

  // -------- Image generation via Chat --------
  const imageResolved = tryResolveImageModel(body.model)
  if (imageResolved) {
    const imagePrompt = getLastUserPrompt(body.messages)
    if (!imagePrompt) {
      return sendError(c, Errors.invalidPrompt('Image prompt is required'))
    }

    const { channelId, model: imageModel } = imageResolved

    if (auth.providerHint && auth.providerHint !== channelId) {
      return sendError(
        c,
        Errors.invalidParams(
          'Authorization',
          'Token prefix does not match requested model provider'
        )
      )
    }

    const channel = getChannel(channelId)
    const imageCapability = getImageChannel(channelId)
    if (!channel || !imageCapability) {
      return sendError(c, Errors.invalidParams('model', `Unsupported image provider: ${channelId}`))
    }

    const allowAnonymous =
      channel.config.auth.type === 'none' || channel.config.auth.optional === true
    const headerTokens = parseTokens(auth.token)
    const tokens = headerTokens.length ? headerTokens : channel.config.tokens || []
    if (!allowAnonymous && tokens.length === 0) {
      return sendError(c, Errors.authRequired(channel.name))
    }

    const { prompt, size, negativePrompt, sourceImageUrl: promptImageUrl } =
      parseImageParams(imagePrompt)
    if (!prompt) {
      return sendError(c, Errors.invalidPrompt('Image prompt is required'))
    }

    // Source image: prefer multimodal image_url from message, fall back to --image in prompt
    const multimodalImageUrl = getLastUserImageUrl(body.messages)
    const sourceImageUrl = multimodalImageUrl || promptImageUrl

    const { width, height } = parseSize(size)
    const resolvedModel = imageModel || channel.config.imageModels?.[0]?.id

    try {
      const result = await runWithTokenRotation(
        channel.id,
        tokens,
        (token) =>
          imageCapability.generate(
            {
              prompt,
              negativePrompt,
              width,
              height,
              model: resolvedModel,
              sourceImageUrl,
            },
            token
          ),
        { allowAnonymous }
      )

      const isEdit =
        sourceImageUrl &&
        (resolvedModel === 'omni-edit' ||
          resolvedModel === 'omni-upscale' ||
          resolvedModel === 'omni-dewatermark')
      const content = isEdit
        ? `🎨 图片编辑成功\n\n` + `提示词: ${prompt}\n\n` + `![${prompt}](${result.url})`
        : `🎨 图片生成成功\n\n` +
          `提示词: ${prompt}\n` +
          `尺寸: ${width}x${height}\n\n` +
          `![${prompt}](${result.url})`
      return sendChatResult(c, body.model, content, body.stream)
    } catch (err) {
      return sendError(c, err)
    }
  }

  // -------- Normal LLM Chat --------
  const systemPrompt = getSystemPrompt(body.messages)
  const resolved = resolveChatModel(body.model)

  if (auth.providerHint && auth.providerHint !== resolved.channelId) {
    return sendError(
      c,
      Errors.invalidParams('Authorization', 'Token prefix does not match requested model provider')
    )
  }

  const channel = getChannel(resolved.channelId)
  const llm = getLLMChannel(resolved.channelId)
  if (!channel || !llm) {
    return sendError(
      c,
      Errors.invalidParams('model', `Unsupported model provider: ${resolved.channelId}`)
    )
  }

  const allowAnonymous =
    resolved.forceAnonymous === true ||
    channel.config.auth.type === 'none' ||
    channel.config.auth.optional === true

  const headerTokens = resolved.forceAnonymous ? [] : parseTokens(auth.token)
  const tokens = headerTokens.length ? headerTokens : channel.config.tokens || []
  if (!allowAnonymous && tokens.length === 0) {
    return sendError(c, Errors.authRequired(channel.name))
  }

  try {
    const result = await runWithTokenRotation(
      channel.id,
      tokens,
      (token) =>
        llm.complete(
          {
            prompt: userPrompt,
            systemPrompt,
            model: resolved.model || channel.config.llmModels?.[0]?.id,
            maxTokens: body.max_tokens,
            temperature: body.temperature,
          },
          token
        ),
      { allowAnonymous }
    )

    return sendChatResult(c, result.model, result.content, body.stream)
  } catch (err) {
    return sendError(c, err)
  }
}
