/**
 * OpenAI-Compatible API Types
 *
 * These types are used by the OpenAI-format `/v1/*` endpoints.
 */

// -------- Errors --------

export interface OpenAIErrorResponse {
  error: {
    message: string
    type: string
    param?: string | null
    code?: string | null
  }
}

// -------- Models --------

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

// -------- Images --------

export interface OpenAIImageRequest {
  model?: string
  prompt: string
  n?: number
  size?: string
  quality?: 'standard' | 'hd'
  response_format?: 'url' | 'b64_json'
  negative_prompt?: string
  /** Source image URL for image editing (omni-edit, omni-upscale, omni-dewatermark) */
  image?: string
  // Extensions / provider-compat aliases
  // - Gitee AI uses `num_inference_steps` instead of `steps`
  // - Some providers use `cfg_scale` instead of `guidance_scale`
  num_inference_steps?: number
  steps?: number
  seed?: number
  cfg_scale?: number
  guidance_scale?: number
}

export interface OpenAIImageResponse {
  created: number
  data: Array<{
    url: string
    revised_prompt?: string
  }>
}

// -------- Chat Completions --------

export type OpenAIChatRole = 'system' | 'user' | 'assistant'

export interface OpenAIChatTextPart {
  type: 'text'
  text: string
}

export interface OpenAIChatImagePart {
  type: 'image_url'
  image_url: { url: string; detail?: string }
}

export type OpenAIChatContentPart = OpenAIChatTextPart | OpenAIChatImagePart

export interface OpenAIChatMessage {
  role: OpenAIChatRole
  /** String for simple text, or array of content parts for multimodal messages */
  content: string | OpenAIChatContentPart[]
}

export interface OpenAIChatRequest {
  model: string
  messages: OpenAIChatMessage[]
  temperature?: number
  max_tokens?: number
  stream?: boolean
}

export interface OpenAIChatResponse {
  id: string
  object: 'chat.completion'
  created: number
  model: string
  choices: Array<{
    index: number
    message: {
      role: 'assistant'
      content: string
    }
    finish_reason: 'stop' | 'length' | 'content_filter' | null
  }>
}
