/**
 * OpenAI-Compatible API Types (Images)
 */

export interface OpenAIImageRequest {
  model?: string
  prompt: string
  n?: number
  size?: string
  quality?: 'standard' | 'hd'
  response_format?: 'url' | 'b64_json'
  negative_prompt?: string
  steps?: number
  seed?: number
  guidance_scale?: number
}

export interface OpenAIImageResponse {
  created: number
  data: Array<{
    url: string
    revised_prompt?: string
  }>
}

export interface OpenAIModelInfo {
  id: string
  object: 'model'
  created: number
  owned_by: string
}

export interface OpenAIModelsListResponse {
  object: 'list'
  data: OpenAIModelInfo[]
}
