import type { ChannelConfig } from '../../core/types'

export const a4fConfig: ChannelConfig = {
  baseUrl: 'https://api.a4f.co/v1',
  auth: { type: 'bearer' },
  endpoints: {
    image: '/images/generations',
    llm: '/chat/completions',
  },
  imageModels: [
    { id: 'gpt-image-1', name: 'GPT Image 1' },
    { id: 'dall-e-3', name: 'DALL-E 3' },
  ],
  llmModels: [{ id: 'gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash Lite' }],
}
