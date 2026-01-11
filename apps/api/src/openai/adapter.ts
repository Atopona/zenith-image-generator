import {
  Errors,
  type ProviderType,
  validateDimensions,
  validatePrompt,
  validateSteps,
} from '@z-image/shared'
import type { ProviderGenerateRequest, ProviderGenerateResult } from '../providers/types'
import type { OpenAIImageRequest, OpenAIImageResponse } from './types'

export type OpenAIConvertedRequest = Pick<
  ProviderGenerateRequest,
  'prompt' | 'negativePrompt' | 'width' | 'height' | 'steps' | 'seed' | 'guidanceScale'
>

export function parseSize(size?: string): { width: number; height: number } {
  if (!size) return { width: 1024, height: 1024 }

  const [w, h] = size.split('x').map((n) => Number.parseInt(n, 10))
  return {
    width: Number.isFinite(w) && w > 0 ? w : 1024,
    height: Number.isFinite(h) && h > 0 ? h : 1024,
  }
}

export function parseBearerToken(authHeader?: string): {
  providerHint?: ProviderType
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

  return { token: raw }
}

export function convertRequest(req: OpenAIImageRequest): OpenAIConvertedRequest {
  const promptValidation = validatePrompt(req.prompt)
  if (!promptValidation.valid) {
    throw Errors.invalidPrompt(promptValidation.error || 'Invalid prompt')
  }

  const { width, height } = parseSize(req.size)
  const dimensionsValidation = validateDimensions(width, height)
  if (!dimensionsValidation.valid) {
    throw Errors.invalidDimensions(dimensionsValidation.error || 'Invalid dimensions')
  }

  const steps = req.steps ?? (req.quality === 'hd' ? 30 : undefined)
  if (steps !== undefined) {
    const stepsValidation = validateSteps(steps)
    if (!stepsValidation.valid) {
      throw Errors.invalidParams('steps', stepsValidation.error || 'Invalid steps')
    }
  }

  return {
    prompt: req.prompt,
    negativePrompt: req.negative_prompt,
    width,
    height,
    steps,
    seed: req.seed,
    guidanceScale: req.guidance_scale,
  }
}

export function convertResponse(result: ProviderGenerateResult): OpenAIImageResponse {
  return {
    created: Math.floor(Date.now() / 1000),
    data: [{ url: result.url }],
  }
}
