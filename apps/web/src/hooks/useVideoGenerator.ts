import { useEffect, useRef, useState } from 'react'
import { createVideoTask, getVideoTaskStatus } from '@/lib/api'
import { generateVideoHF } from '@/lib/hfVideoService'

export interface VideoState {
  status: 'idle' | 'generating' | 'polling' | 'success' | 'failed'
  taskId?: string
  videoUrl?: string
  error?: string
  provider?: 'gitee' | 'huggingface'
}

export function useVideoGenerator() {
  const [videoState, setVideoState] = useState<VideoState>({ status: 'idle' })
  const pollingRef = useRef<NodeJS.Timeout>()

  useEffect(() => {
    if (videoState.status !== 'polling' || !videoState.taskId) return

    const pollInterval = setInterval(async () => {
      const giteeToken = localStorage.getItem('gitee_token')
      if (!giteeToken || !videoState.taskId) {
        clearInterval(pollInterval)
        setVideoState({ status: 'failed', error: 'Missing token or task ID' })
        return
      }

      const result = await getVideoTaskStatus(videoState.taskId, giteeToken)

      if (result.success) {
        if (result.data.status === 'success') {
          setVideoState({
            status: 'success',
            videoUrl: result.data.videoUrl,
            provider: videoState.provider,
          })
          clearInterval(pollInterval)
        } else if (result.data.status === 'failed') {
          setVideoState({
            status: 'failed',
            error: result.data.error || 'Video generation failed',
            provider: videoState.provider,
          })
          clearInterval(pollInterval)
        }
      } else {
        setVideoState({
          status: 'failed',
          error: result.error,
          provider: videoState.provider,
        })
        clearInterval(pollInterval)
      }
    }, 3000)

    pollingRef.current = pollInterval

    return () => clearInterval(pollInterval)
  }, [videoState.status, videoState.taskId, videoState.provider])

  const generateVideo = async (
    imageUrl: string,
    prompt: string,
    width: number,
    height: number,
    provider: 'gitee' | 'huggingface',
    giteeToken?: string
  ) => {
    setVideoState({ status: 'generating', provider })

    try {
      if (provider === 'huggingface') {
        const videoUrl = await generateVideoHF(imageUrl, prompt)
        setVideoState({ status: 'success', videoUrl, provider })
      } else {
        if (!giteeToken) {
          setVideoState({ status: 'failed', error: 'Gitee token required', provider })
          return
        }

        const result = await createVideoTask(
          { provider, imageUrl, prompt, width, height },
          giteeToken
        )

        if (result.success) {
          setVideoState({ status: 'polling', taskId: result.data.taskId, provider })
        } else {
          setVideoState({ status: 'failed', error: result.error, provider })
        }
      }
    } catch (err) {
      setVideoState({
        status: 'failed',
        error: err instanceof Error ? err.message : 'Unknown error',
        provider,
      })
    }
  }

  const resetVideo = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current)
    }
    setVideoState({ status: 'idle' })
  }

  return { videoState, generateVideo, resetVideo }
}
