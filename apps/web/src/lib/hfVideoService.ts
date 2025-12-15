import { VIDEO_NEGATIVE_PROMPT } from '@z-image/shared'

const WAN2_VIDEO_API = 'https://zerogpu-aoti-wan2-2-fp8da-aoti-faster.hf.space'

export async function generateVideoHF(
  imageUrl: string,
  prompt: string,
  seed: number = Math.floor(Math.random() * 2147483647)
): Promise<string> {
  const queueResponse = await fetch(`${WAN2_VIDEO_API}/gradio_api/call/generate_video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data: [
        { path: imageUrl, meta: { _type: 'gradio.FileData' } },
        prompt,
        6,
        VIDEO_NEGATIVE_PROMPT,
        3,
        1,
        1,
        seed,
        false,
      ],
    }),
  })

  if (!queueResponse.ok) {
    throw new Error(`Failed to queue video generation: ${queueResponse.status}`)
  }

  const { event_id } = await queueResponse.json()

  const resultResponse = await fetch(`${WAN2_VIDEO_API}/gradio_api/call/generate_video/${event_id}`)

  if (!resultResponse.ok) {
    throw new Error(`Failed to get video result: ${resultResponse.status}`)
  }

  const text = await resultResponse.text()
  return parseGradioSSE(text)
}

function parseGradioSSE(text: string): string {
  const lines = text.split('\n')
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      try {
        const data = JSON.parse(line.slice(6))
        if (data[0]?.video?.url) return data[0].video.url
        if (data[0]?.url) return data[0].url
      } catch {
        // Continue parsing
      }
    }
  }
  throw new Error('No video URL in response')
}
