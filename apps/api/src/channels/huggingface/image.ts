import { ApiError, ApiErrorCode, Errors, HF_SPACES } from '@z-image/shared'
import { MAX_INT32 } from '../../constants'
import type { ImageCapability, ImageRequest, ImageResult } from '../../core/types'
import { callGradioApi } from '../../utils'

function normalizeImageUrl(baseUrl: string, url: string): string {
  try {
    return new URL(url, baseUrl).toString()
  } catch {
    return url
  }
}

function parseSeedFromResponse(modelId: string, result: unknown[], fallbackSeed: number): number {
  if (modelId === 'qwen-image-fast' && typeof result[1] === 'string') {
    const match = result[1].match(/Seed used for generation:\s*(\d+)/)
    if (match) return Number.parseInt(match[1], 10)
  }
  if (typeof result[1] === 'number') return result[1]
  return fallbackSeed
}

function getCandidateBaseUrls(modelId: string): string[] {
  // All omni-* models share the same Space
  const spaceKey = modelId.startsWith('omni-') ? 'omni-image' : modelId
  const primary = HF_SPACES[spaceKey as keyof typeof HF_SPACES] || HF_SPACES['z-image-turbo']
  const fallbackMap: Record<string, string[]> = {
    'z-image-turbo': ['https://mrfakename-z-image-turbo.hf.space'],
    'z-image': ['https://mrfakename-z-image.hf.space'],
  }
  const fallbacks = fallbackMap[modelId] || ([] as string[])
  return [primary, ...fallbacks].filter(Boolean)
}

function isNotFoundProviderError(err: unknown): boolean {
  if (err instanceof ApiError) {
    return err.code === ApiErrorCode.PROVIDER_ERROR && (err.details?.upstream || '').includes('404')
  }
  if (err && typeof err === 'object' && 'code' in err) {
    const code = (err as { code?: unknown }).code
    const details = (err as { details?: unknown }).details as { upstream?: string } | undefined
    return code === ApiErrorCode.PROVIDER_ERROR && (details?.upstream || '').includes('404')
  }
  return false
}

/** Size string to aspect ratio for omni-image text-to-image */
function sizeToAspectRatio(width: number, height: number): string {
  const ratio = width / height
  if (Math.abs(ratio - 16 / 9) < 0.1) return '16:9'
  if (Math.abs(ratio - 9 / 16) < 0.1) return '9:16'
  if (Math.abs(ratio - 4 / 3) < 0.1) return '4:3'
  if (Math.abs(ratio - 3 / 4) < 0.1) return '3:4'
  return '1:1'
}

/** Build a Gradio FileData object from a public image URL */
function makeGradioFileData(url: string): Record<string, unknown> {
  return {
    path: url,
    url,
    orig_name: 'input.jpg',
    mime_type: 'image/jpeg',
    size: null,
    is_stream: false,
    meta: { _type: 'gradio.FileData' },
  }
}

/** Extract the first image URL from an HTML string returned by omni-image */
function extractImageUrlFromHtml(html: string): string | undefined {
  // Match <img ... src="..."> pattern
  const imgMatch = html.match(/<img[^>]+src=["']([^"']+)["']/i)
  if (imgMatch?.[1]) return imgMatch[1]
  // Fallback: match any https URL ending with common image extensions
  const urlMatch = html.match(/https?:\/\/[^\s"'<>]+\.(?:jpg|jpeg|png|webp|gif)(?:\?[^\s"'<>]*)?/i)
  return urlMatch?.[0]
}

/** Set of omni-image model IDs that return HTML instead of direct image data */
const OMNI_IMAGE_MODELS = new Set(['omni-image', 'omni-edit', 'omni-upscale', 'omni-dewatermark'])

const MODEL_CONFIGS: Record<
  string,
  { endpoint: string; buildData: (r: ImageRequest, seed: number) => unknown[] }
> = {
  'z-image-turbo': {
    endpoint: 'generate_image',
    buildData: (r, seed) => [r.prompt, r.height, r.width, r.steps ?? 9, seed, false],
  },
  'qwen-image-fast': {
    endpoint: 'generate_image',
    buildData: (r, seed) => [r.prompt, seed, true, '1:1', 3, r.steps ?? 8],
  },
  'ovis-image': {
    endpoint: 'generate',
    buildData: (r, seed) => [r.prompt, r.height, r.width, seed, r.steps ?? 24, 4],
  },
  'flux-1-schnell': {
    endpoint: 'infer',
    buildData: (r, seed) => [r.prompt, seed, false, r.width, r.height, r.steps ?? 8],
  },
  'z-image': {
    endpoint: 'generate_image',
    buildData: (r, seed) => [
      r.prompt,
      r.negativePrompt || '',
      r.height,
      r.width,
      r.steps ?? 28,
      r.guidanceScale ?? 4.0,
      seed,
      false,
    ],
  },
  'omni-image': {
    endpoint: 'text_to_image_interface',
    buildData: (r) => [r.prompt, r.aspectRatio || sizeToAspectRatio(r.width, r.height)],
  },
  'omni-edit': {
    endpoint: 'edit_image_interface',
    buildData: (r) => [makeGradioFileData(r.sourceImageUrl || ''), r.prompt],
  },
  'omni-upscale': {
    endpoint: 'image_upscale_interface',
    buildData: (r) => [makeGradioFileData(r.sourceImageUrl || '')],
  },
  'omni-dewatermark': {
    endpoint: 'watermark_removal_interface',
    buildData: (r) => [makeGradioFileData(r.sourceImageUrl || ''), false],
  },
}

export const huggingfaceImage: ImageCapability = {
  async generate(request: ImageRequest, token?: string | null): Promise<ImageResult> {
    const seed = request.seed ?? Math.floor(Math.random() * MAX_INT32)
    const modelId = request.model || 'z-image-turbo'
    const config = MODEL_CONFIGS[modelId] || MODEL_CONFIGS['z-image-turbo']

    // Validate source image for editing models
    const needsSourceImage =
      modelId === 'omni-edit' || modelId === 'omni-upscale' || modelId === 'omni-dewatermark'
    if (needsSourceImage && !request.sourceImageUrl) {
      throw Errors.invalidParams(
        'image',
        'Source image URL is required for image editing. Use --image <url> in chat or "image" field in API.'
      )
    }

    let lastErr: unknown
    let imageUrl: string | undefined
    let data: unknown[] | undefined

    for (const baseUrl of getCandidateBaseUrls(modelId)) {
      try {
        data = await callGradioApi(
          baseUrl,
          config.endpoint,
          config.buildData(request, seed),
          token || undefined
        )
        const result = data as Array<{ url?: string } | number | string>

        // Omni-image models return HTML containing the image URL
        if (OMNI_IMAGE_MODELS.has(modelId)) {
          const html = typeof result[0] === 'string' ? result[0] : ''
          // Check status text (result[1]) for errors
          const status = typeof result[1] === 'string' ? result[1] : ''
          if (status && (status.includes('Rate limit') || status.includes('error'))) {
            throw Errors.providerError('HuggingFace', status)
          }
          imageUrl = extractImageUrlFromHtml(html)
        } else {
          const first = result[0]
          const rawUrl =
            typeof first === 'string' ? first : (first as { url?: string } | null | undefined)?.url
          imageUrl = rawUrl ? normalizeImageUrl(baseUrl, rawUrl) : undefined
        }
        if (!imageUrl) throw Errors.generationFailed('HuggingFace', 'No image returned')
        break
      } catch (err) {
        lastErr = err
        if (isNotFoundProviderError(err)) continue
        throw err
      }
    }

    if (!imageUrl) {
      if (lastErr) throw lastErr
      throw Errors.generationFailed('HuggingFace', 'No image returned')
    }

    return {
      url: imageUrl,
      seed: parseSeedFromResponse(modelId, data as unknown[], seed),
      model: modelId,
    }
  },
}
