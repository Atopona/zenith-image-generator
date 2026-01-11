import { Errors, getModelsByProvider, PROVIDER_CONFIGS } from '@z-image/shared'
import type { Hono } from 'hono'
import { sendError } from '../middleware'
import { getProvider } from '../providers'
import { convertRequest, convertResponse, parseBearerToken } from './adapter'
import { resolveModel } from './model-resolver'
import type { OpenAIImageRequest, OpenAIModelsListResponse } from './types'

function getOrigin(c: Parameters<Parameters<Hono['post']>[1]>[0]): string {
  try {
    return new URL(c.req.url).origin
  } catch {
    return ''
  }
}

function toProxyUrl(origin: string, url: string): string {
  if (!origin) return url

  try {
    const parsed = new URL(url)
    if (parsed.hostname.endsWith('.hf.space') && parsed.pathname.startsWith('/gradio_api/file=')) {
      return `${origin}/api/proxy-image?url=${encodeURIComponent(url)}`
    }
  } catch {
    // ignore invalid URL, fall back to raw string checks
  }

  if (url.includes('.hf.space/gradio_api/file=')) {
    return `${origin}/api/proxy-image?url=${encodeURIComponent(url)}`
  }

  return url
}

function listModels(): OpenAIModelsListResponse {
  const created = 1700000000
  const giteeModels = getModelsByProvider('gitee').map((m) => m.id)
  const modelscopeModels = getModelsByProvider('modelscope').map((m) => m.id)

  return {
    object: 'list',
    data: [
      ...['z-image-turbo', 'qwen-image-fast', 'ovis-image', 'flux-1-schnell'].map((id) => ({
        id,
        object: 'model' as const,
        created,
        owned_by: 'huggingface',
      })),
      ...giteeModels.flatMap((id) => {
        const entries = [{ id: `gitee/${id}`, owned_by: 'gitee' }]
        if (id === 'Qwen-Image') entries.push({ id: 'gitee/qwen-image', owned_by: 'gitee' })
        if (id === 'FLUX_1-Krea-dev')
          entries.push({ id: 'gitee/flux-1-krea-dev', owned_by: 'gitee' })
        if (id === 'FLUX.1-dev') entries.push({ id: 'gitee/flux-1-dev', owned_by: 'gitee' })
        return entries.map((e) => ({ ...e, object: 'model' as const, created }))
      }),
      ...modelscopeModels.flatMap((id) => {
        const entries = [{ id: `ms/${id}`, owned_by: 'modelscope' }]
        if (id === 'Tongyi-MAI/Z-Image-Turbo')
          entries.push({ id: 'ms/z-image-turbo', owned_by: 'modelscope' })
        if (id === 'black-forest-labs/FLUX.2-dev')
          entries.push({ id: 'ms/flux-2', owned_by: 'modelscope' })
        if (id === 'black-forest-labs/FLUX.1-Krea-dev')
          entries.push({ id: 'ms/flux-1-krea-dev', owned_by: 'modelscope' })
        if (id === 'MusePublic/489_ckpt_FLUX_1')
          entries.push({ id: 'ms/flux-1', owned_by: 'modelscope' })
        return entries.map((e) => ({ ...e, object: 'model' as const, created }))
      }),
    ],
  }
}

export function registerOpenAIRoutes(app: Hono) {
  app.post('/images/generations', async (c) => {
    let body: OpenAIImageRequest
    try {
      body = (await c.req.json()) as OpenAIImageRequest
    } catch {
      return sendError(c, Errors.invalidParams('body', 'Invalid JSON body'))
    }

    if (!body?.prompt) {
      return sendError(c, Errors.invalidPrompt('Prompt is required'))
    }

    if (body.n !== undefined && body.n !== 1) {
      return sendError(c, Errors.invalidParams('n', 'Only n=1 is supported'))
    }

    if (body.response_format !== undefined && body.response_format !== 'url') {
      return sendError(
        c,
        Errors.invalidParams('response_format', "Only response_format='url' is supported")
      )
    }

    const { provider, model } = resolveModel(body.model)
    const auth = parseBearerToken(c.req.header('Authorization'))

    if (auth.providerHint && auth.providerHint !== provider) {
      return sendError(
        c,
        Errors.invalidParams(
          'Authorization',
          'Token prefix does not match requested model provider'
        )
      )
    }

    const providerConfig = PROVIDER_CONFIGS[provider]
    if (providerConfig?.requiresAuth && !auth.token) {
      return sendError(c, Errors.authRequired(providerConfig.name))
    }

    try {
      const internalReq = convertRequest(body)
      const imageProvider = getProvider(provider)
      const result = await imageProvider.generate({
        ...internalReq,
        model,
        authToken: auth.token,
      })
      const proxyUrl = toProxyUrl(getOrigin(c), result.url)
      return c.json(convertResponse({ ...result, url: proxyUrl }))
    } catch (err) {
      return sendError(c, err)
    }
  })

  app.get('/models', (c) => c.json(listModels()))
}
