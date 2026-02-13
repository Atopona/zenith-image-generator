import { Errors } from '@z-image/shared'
import type { Context } from 'hono'
import {
  ensureCustomChannelsInitialized,
  getChannel,
  getImageChannel,
  getLLMChannel,
} from '../channels'
import { parseTokens, runWithTokenRotation } from '../core/token-manager'
import { sendError } from '../middleware'
import { parseSize } from './adapter'
import { resolveModel } from './model-resolver'
import type { OpenAIChatRequest, OpenAIChatResponse } from './types'

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

function isImageModel(model: string): boolean {
  return model.trim().startsWith(IMAGE_MODEL_PREFIX)
}

function resolveImageModelForChat(model: string): { channelId: string; model: string } {
  const inner = model.trim().slice(IMAGE_MODEL_PREFIX.length)

  // Support custom/ prefix
  if (inner.startsWith('custom/')) {
    const rest = inner.slice('custom/'.length)
    const firstSlash = rest.indexOf('/')
    if (firstSlash > 0) {
      const channelId = rest.slice(0, firstSlash).trim()
      const m = rest.slice(firstSlash + 1).trim()
      if (channelId) return { channelId, model: m }
    }
  }

  const resolved = resolveModel(inner || undefined)
  return { channelId: resolved.provider, model: resolved.model }
}

/** Extract --size and --negative from user prompt, return cleaned prompt + params */
function parseImageParams(raw: string): {
  prompt: string
  size?: string
  negativePrompt?: string
} {
  let size: string | undefined
  let negativePrompt: string | undefined

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

  return { prompt: text.trim(), size, negativePrompt }
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
    .map((m) => m.content)
    .join('\n')
    .trim()
}

function getUserPrompt(messages: OpenAIChatRequest['messages']): string | null {
  const parts = messages
    .filter((m) => m.role === 'user')
    .map((m) => m.content)
    .filter((s) => s && s.trim().length > 0)
  if (parts.length === 0) return null
  return parts.join('\n').trim()
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
  if (isImageModel(body.model)) {
    const { channelId, model: imageModel } = resolveImageModelForChat(body.model)

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

    const { prompt, size, negativePrompt } = parseImageParams(userPrompt)
    if (!prompt) {
      return sendError(c, Errors.invalidPrompt('Image prompt is required'))
    }

    const { width, height } = parseSize(size)
    const resolvedModel = imageModel || channel.config.imageModels?.[0]?.id

    try {
      const result = await runWithTokenRotation(
        channel.id,
        tokens,
        (token) =>
          imageCapability.generate(
            { prompt, negativePrompt, width, height, model: resolvedModel },
            token
          ),
        { allowAnonymous }
      )

      const content = `![${prompt}](${result.url})\n\nGenerated with model \`${resolvedModel || channelId}\` (${width}x${height})`
      return c.json(makeChatResponse(body.model, content))
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

    return c.json(makeChatResponse(result.model, result.content))
  } catch (err) {
    return sendError(c, err)
  }
}
