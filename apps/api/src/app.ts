/**
 * Z-Image API - Platform Agnostic Hono App
 *
 * This module exports a platform-agnostic Hono app that can be used
 * with any runtime (Node.js, Cloudflare Workers, Deno, Bun, etc.)
 */

import {
  type GenerateRequest,
  HF_SPACES,
  MODEL_CONFIGS,
  PROVIDER_CONFIGS,
  type ProviderType,
  getModelsByProvider,
  isAllowedImageUrl,
  validateDimensions,
  validatePrompt,
  validateScale,
  validateSteps,
} from '@z-image/shared'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { getProvider, hasProvider } from './providers'

/** Extract complete event data from SSE stream */
function extractCompleteEventData(sseStream: string): unknown {
  const lines = sseStream.split('\n')
  let isCompleteEvent = false

  for (const line of lines) {
    if (line.startsWith('event:')) {
      const eventType = line.substring(6).trim()
      if (eventType === 'complete') {
        isCompleteEvent = true
      } else if (eventType === 'error') {
        throw new Error('Quota exhausted, please set HF Token')
      } else {
        isCompleteEvent = false
      }
    } else if (line.startsWith('data:') && isCompleteEvent) {
      const jsonData = line.substring(5).trim()
      return JSON.parse(jsonData)
    }
  }
  throw new Error(`No complete event in response: ${sseStream.substring(0, 200)}`)
}

/** Call Gradio API for upscaling */
async function callGradioApi(baseUrl: string, endpoint: string, data: unknown[], hfToken?: string) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (hfToken) headers.Authorization = `Bearer ${hfToken}`

  const queue = await fetch(`${baseUrl}/gradio_api/call/${endpoint}`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ data }),
  })

  if (!queue.ok) throw new Error(`Queue request failed: ${queue.status}`)

  const queueData = (await queue.json()) as { event_id?: string }
  if (!queueData.event_id) throw new Error('No event_id returned')

  const result = await fetch(`${baseUrl}/gradio_api/call/${endpoint}/${queueData.event_id}`, {
    headers,
  })
  const text = await result.text()

  return extractCompleteEventData(text) as unknown[]
}

export interface AppConfig {
  corsOrigins?: string[]
}

export function createApp(config: AppConfig = {}) {
  const app = new Hono().basePath('/api')

  // Default CORS origins for development
  const defaultOrigins = ['http://localhost:5173', 'http://localhost:3000']
  const origins = config.corsOrigins || defaultOrigins

  // CORS middleware
  app.use('/*', async (c, next) => {
    return cors({
      origin: origins,
      allowMethods: ['GET', 'POST', 'OPTIONS'],
      allowHeaders: ['Content-Type', 'X-API-Key', 'X-HF-Token', 'X-MS-Token'],
    })(c, next)
  })

  // Health check
  app.get('/', (c) => {
    return c.json({ message: 'Z-Image API is running' })
  })

  // Get all providers
  app.get('/providers', (c) => {
    const providers = Object.values(PROVIDER_CONFIGS).map((p) => ({
      id: p.id,
      name: p.name,
      requiresAuth: p.requiresAuth,
      authHeader: p.authHeader,
    }))
    return c.json({ providers })
  })

  // Get models by provider
  app.get('/providers/:provider/models', (c) => {
    const provider = c.req.param('provider') as ProviderType
    if (!PROVIDER_CONFIGS[provider]) {
      return c.json({ error: `Invalid provider: ${provider}` }, 400)
    }
    const models = getModelsByProvider(provider).map((m) => ({
      id: m.id,
      name: m.name,
      features: m.features,
    }))
    return c.json({ provider, models })
  })

  // Get all models
  app.get('/models', (c) => {
    const models = MODEL_CONFIGS.map((m) => ({
      id: m.id,
      name: m.name,
      provider: m.provider,
      features: m.features,
    }))
    return c.json({ models })
  })

  // Unified generate endpoint
  app.post('/generate', async (c) => {
    let body: GenerateRequest & { negative_prompt?: string; num_inference_steps?: number }
    try {
      body = await c.req.json()
    } catch {
      return c.json({ error: 'Invalid JSON body' }, 400)
    }

    // Determine provider (default to gitee for backward compatibility)
    const providerId = body.provider || 'gitee'
    if (!hasProvider(providerId)) {
      return c.json({ error: `Invalid provider: ${providerId}` }, 400)
    }

    // Get auth token based on provider
    const providerConfig = PROVIDER_CONFIGS[providerId]
    const authToken = c.req.header(providerConfig?.authHeader || 'X-API-Key')

    // Debug: log token info (uncomment for debugging)
    // console.log(`[${providerId}] Auth header: ${providerConfig?.authHeader}, Token present: ${!!authToken}, Token length: ${authToken?.length || 0}`)

    // Check auth requirement
    if (providerConfig?.requiresAuth && !authToken) {
      return c.json(
        { error: `${providerConfig.authHeader} is required for ${providerConfig.name}` },
        401
      )
    }

    // Validate prompt
    const promptValidation = validatePrompt(body.prompt)
    if (!promptValidation.valid) {
      return c.json({ error: promptValidation.error }, 400)
    }

    // Validate dimensions
    const width = body.width ?? 1024
    const height = body.height ?? 1024
    const dimensionsValidation = validateDimensions(width, height)
    if (!dimensionsValidation.valid) {
      return c.json({ error: dimensionsValidation.error }, 400)
    }

    // Validate steps
    const steps = body.steps ?? body.num_inference_steps ?? 9
    const stepsValidation = validateSteps(steps)
    if (!stepsValidation.valid) {
      return c.json({ error: stepsValidation.error }, 400)
    }

    try {
      const provider = getProvider(providerId as ProviderType)
      const result = await provider.generate({
        model: body.model,
        prompt: body.prompt,
        negativePrompt: body.negativePrompt || body.negative_prompt,
        width,
        height,
        steps,
        seed: body.seed,
        guidanceScale: body.guidanceScale,
        authToken,
      })
      return c.json(result)
    } catch (err) {
      console.error(`${providerId} Error:`, err)
      const message = err instanceof Error ? err.message : 'Image generation failed'
      return c.json({ error: message }, 500)
    }
  })

  // Legacy HuggingFace endpoint (for backward compatibility)
  app.post('/generate-hf', async (c) => {
    let body: { prompt: string; width?: number; height?: number; model?: string; seed?: number }
    try {
      body = await c.req.json()
    } catch {
      return c.json({ error: 'Invalid JSON body' }, 400)
    }

    // Validate prompt
    const promptValidation = validatePrompt(body.prompt)
    if (!promptValidation.valid) {
      return c.json({ error: promptValidation.error }, 400)
    }

    const hfToken = c.req.header('X-HF-Token')
    const width = body.width ?? 1024
    const height = body.height ?? 1024

    const dimensionsValidation = validateDimensions(width, height)
    if (!dimensionsValidation.valid) {
      return c.json({ error: dimensionsValidation.error }, 400)
    }

    try {
      const provider = getProvider('huggingface')
      const result = await provider.generate({
        model: body.model || 'z-image',
        prompt: body.prompt,
        width,
        height,
        seed: body.seed,
        authToken: hfToken,
      })
      return c.json(result)
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : 'Generation failed' }, 500)
    }
  })

  // Upscale endpoint
  app.post('/upscale', async (c) => {
    let body: { url: string; scale?: number }
    try {
      body = await c.req.json()
    } catch {
      return c.json({ error: 'Invalid JSON body' }, 400)
    }

    if (!body.url || typeof body.url !== 'string') {
      return c.json({ error: 'url is required' }, 400)
    }

    if (!isAllowedImageUrl(body.url)) {
      return c.json({ error: 'URL not allowed' }, 400)
    }

    const hfToken = c.req.header('X-HF-Token')
    const scale = body.scale ?? 4

    const scaleValidation = validateScale(scale)
    if (!scaleValidation.valid) {
      return c.json({ error: scaleValidation.error }, 400)
    }

    try {
      const data = await callGradioApi(
        HF_SPACES.upscaler,
        'realesrgan',
        [
          { path: body.url, meta: { _type: 'gradio.FileData' } },
          'RealESRGAN_x4plus',
          0.5,
          false,
          scale,
        ],
        hfToken
      )
      const result = data as Array<{ url?: string }>
      const imageUrl = result[0]?.url
      if (!imageUrl) return c.json({ error: 'No image returned' }, 500)
      return c.json({ url: imageUrl })
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : 'Upscale failed' }, 500)
    }
  })

  return app
}

// Default app instance for simple usage
const app = createApp()

export default app
