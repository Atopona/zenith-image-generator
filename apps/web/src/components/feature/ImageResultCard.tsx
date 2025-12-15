import type { ImageDetails } from '@z-image/shared'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Check,
  Download,
  Eye,
  EyeOff,
  ImageIcon,
  Info,
  Loader2,
  Trash2,
  Video,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ImageComparison } from '@/components/ui/ImageComparison'
import { useVideoGenerator } from '@/hooks/useVideoGenerator'
import { upscaleImage } from '@/lib/api'

interface ImageResultCardProps {
  imageDetails: ImageDetails | null
  loading: boolean
  elapsed: number
  showInfo: boolean
  isBlurred: boolean
  isUpscaled: boolean
  isUpscaling: boolean
  setShowInfo: (v: boolean) => void
  setIsBlurred: (v: boolean) => void
  handleUpscale: () => void
  handleDownload: () => void
  handleDelete: () => void
}

export function ImageResultCard({
  imageDetails,
  loading,
  elapsed,
  showInfo,
  isBlurred,
  isUpscaled,
  isUpscaling: externalIsUpscaling,
  setShowInfo,
  setIsBlurred,
  handleUpscale: _externalHandleUpscale,
  handleDownload,
  handleDelete,
}: ImageResultCardProps) {
  // Comparison mode state
  const [isComparing, setIsComparing] = useState(false)
  const [tempUpscaledUrl, setTempUpscaledUrl] = useState<string | null>(null)
  const [isUpscalingLocal, setIsUpscalingLocal] = useState(false)
  const [isUpscaledLocal, setIsUpscaledLocal] = useState(false)
  const [displayUrl, setDisplayUrl] = useState<string | null>(null)

  // Fullscreen preview state
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [scale, setScale] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const dragStartRef = useRef({ x: 0, y: 0, posX: 0, posY: 0 })
  const imageContainerRef = useRef<HTMLDivElement>(null)

  // Video generation state
  const { videoState, generateVideo } = useVideoGenerator()
  const [showVideo, setShowVideo] = useState(false)

  // Use display URL if set (after applying upscale), otherwise original
  const currentImageUrl = displayUrl || imageDetails?.url

  // Combined upscaling state
  const isUpscaling = externalIsUpscaling || isUpscalingLocal

  // Combined upscaled state (either from parent or local)
  const isImageUpscaled = isUpscaled || isUpscaledLocal

  // Reset position when scale changes to 1
  useEffect(() => {
    if (scale === 1) {
      setPosition({ x: 0, y: 0 })
    }
  }, [scale])

  // Handle upscale with comparison
  const handleUpscaleWithCompare = async () => {
    if (!currentImageUrl || isUpscaling || isImageUpscaled) return

    setIsUpscalingLocal(true)
    try {
      const result = await upscaleImage(currentImageUrl, 4)

      if (result.success && result.data.url) {
        setTempUpscaledUrl(result.data.url)
        setIsComparing(true)
        toast.success('4x upscale complete! Drag slider to compare.')
      } else if (!result.success) {
        toast.error(result.error || 'Upscale failed')
      }
    } catch (_err) {
      toast.error('Upscale failed')
    } finally {
      setIsUpscalingLocal(false)
    }
  }

  // Confirm upscaled image
  const handleConfirmUpscale = useCallback(() => {
    if (tempUpscaledUrl) {
      setDisplayUrl(tempUpscaledUrl)
      setTempUpscaledUrl(null)
      setIsComparing(false)
      // Mark as upscaled locally (don't call parent's handleUpscale to avoid double API call)
      setIsUpscaledLocal(true)
      toast.success('Upscaled image applied!')
    }
  }, [tempUpscaledUrl])

  // Cancel comparison
  const handleCancelComparison = useCallback(() => {
    setTempUpscaledUrl(null)
    setIsComparing(false)
  }, [])

  // Handle video generation
  const handleGenerateVideo = useCallback(async () => {
    if (!currentImageUrl || !imageDetails) return

    const giteeToken = localStorage.getItem('gitee_token')
    if (!giteeToken) {
      toast.error('Please configure Gitee AI token first')
      return
    }

    const width = Number.parseInt(imageDetails.dimensions.split('x')[0], 10) || 1024
    const height = Number.parseInt(imageDetails.dimensions.split('x')[1], 10) || 1024

    toast.info('Starting video generation...')
    await generateVideo(currentImageUrl, imageDetails.prompt, width, height, 'gitee', giteeToken)
  }, [currentImageUrl, imageDetails, generateVideo])

  // Show video when generation succeeds
  useEffect(() => {
    if (videoState.status === 'success') {
      setShowVideo(true)
      toast.success('Video generated successfully!')
    } else if (videoState.status === 'failed') {
      toast.error(videoState.error || 'Video generation failed')
    }
  }, [videoState.status, videoState.error])

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    setScale((s) => Math.min(s * 1.5, 8))
  }, [])

  const handleZoomOut = useCallback(() => {
    setScale((s) => Math.max(s / 1.5, 1))
  }, [])

  const handleResetZoom = useCallback(() => {
    setScale(1)
    setPosition({ x: 0, y: 0 })
  }, [])

  // Close fullscreen
  const closeFullscreen = useCallback(() => {
    setIsFullscreen(false)
    setScale(1)
    setPosition({ x: 0, y: 0 })
    // Reset comparison state when closing fullscreen
    if (isComparing) {
      setIsComparing(false)
      setTempUpscaledUrl(null)
    }
  }, [isComparing])

  // Mouse wheel zoom handler
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      if (isComparing) return // Disable wheel zoom during comparison
      e.preventDefault()
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      setScale((s) => Math.min(Math.max(s * delta, 1), 8))
    },
    [isComparing]
  )

  // Drag handlers for panning
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (scale <= 1 || isComparing) return
      e.preventDefault()
      setIsDragging(true)
      dragStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        posX: position.x,
        posY: position.y,
      }
    },
    [scale, position, isComparing]
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging || scale <= 1) return
      const dx = e.clientX - dragStartRef.current.x
      const dy = e.clientY - dragStartRef.current.y
      setPosition({
        x: dragStartRef.current.posX + dx,
        y: dragStartRef.current.posY + dy,
      })
    },
    [isDragging, scale]
  )

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  // Double-click handler to open fullscreen
  const handleDoubleClick = useCallback(() => {
    if (!currentImageUrl || isComparing) return
    setIsFullscreen(true)
  }, [currentImageUrl, isComparing])

  // Keyboard shortcuts for fullscreen mode
  useEffect(() => {
    if (!isFullscreen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          closeFullscreen()
          break
        case '+':
        case '=':
          handleZoomIn()
          break
        case '-':
          handleZoomOut()
          break
        case '0':
          handleResetZoom()
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isFullscreen, closeFullscreen, handleZoomIn, handleZoomOut, handleResetZoom])

  return (
    <>
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader className="pb-3">
          <CardTitle className="text-zinc-500 text-sm font-normal">Result</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800 group">
            {imageDetails ? (
              <>
                {/* Video or Image display */}
                {showVideo && videoState.videoUrl ? (
                  <video
                    src={videoState.videoUrl}
                    controls
                    autoPlay
                    loop
                    muted
                    className="w-full"
                  />
                ) : (
                  <img
                    src={currentImageUrl || ''}
                    alt="Generated"
                    className={`w-full transition-all duration-300 cursor-pointer ${isBlurred ? 'blur-xl' : ''}`}
                    onDoubleClick={handleDoubleClick}
                    title="Double-click to view fullscreen"
                  />
                )}

                {/* Floating Toolbar */}
                <div className="absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-none">
                  <div className="pointer-events-auto flex items-center gap-1 p-1.5 rounded-2xl bg-black/60 backdrop-blur-md border border-white/10 shadow-2xl transition-opacity duration-300 opacity-100 md:opacity-0 md:group-hover:opacity-100">
                    {/* Info */}
                    <button
                      type="button"
                      onClick={() => setShowInfo(!showInfo)}
                      title="Details"
                      className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all ${
                        showInfo
                          ? 'bg-orange-600 text-white'
                          : 'text-white/70 hover:text-white hover:bg-white/10'
                      }`}
                    >
                      <Info className="w-5 h-5" />
                    </button>
                    <div className="w-px h-5 bg-white/10" />
                    {/* Blur Toggle */}
                    <button
                      type="button"
                      onClick={() => setIsBlurred(!isBlurred)}
                      title="Toggle Blur"
                      className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all ${
                        isBlurred
                          ? 'text-orange-400 bg-white/10'
                          : 'text-white/70 hover:text-white hover:bg-white/10'
                      }`}
                    >
                      {isBlurred ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                    <div className="w-px h-5 bg-white/10" />
                    {/* Generate Video */}
                    <button
                      type="button"
                      onClick={
                        videoState.status === 'success'
                          ? () => setShowVideo(!showVideo)
                          : handleGenerateVideo
                      }
                      disabled={
                        videoState.status === 'generating' || videoState.status === 'polling'
                      }
                      title={
                        videoState.status === 'generating' || videoState.status === 'polling'
                          ? 'Generating video...'
                          : videoState.status === 'success'
                            ? showVideo
                              ? 'Show image'
                              : 'Show video'
                            : 'Generate video'
                      }
                      className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all ${
                        videoState.status === 'success'
                          ? 'text-green-400 bg-green-500/10'
                          : 'text-white/70 hover:text-white hover:bg-white/10'
                      } disabled:cursor-not-allowed disabled:opacity-50`}
                    >
                      {videoState.status === 'generating' || videoState.status === 'polling' ? (
                        <Loader2 className="w-5 h-5 animate-spin text-orange-400" />
                      ) : (
                        <Video className="w-5 h-5" />
                      )}
                    </button>
                    <div className="w-px h-5 bg-white/10" />
                    {/* Download */}
                    <button
                      type="button"
                      onClick={handleDownload}
                      title="Download"
                      className="flex items-center justify-center w-10 h-10 rounded-xl transition-all text-white/70 hover:text-white hover:bg-white/10"
                    >
                      <Download className="w-5 h-5" />
                    </button>
                    {/* Delete */}
                    <button
                      type="button"
                      onClick={handleDelete}
                      title="Delete"
                      className="flex items-center justify-center w-10 h-10 rounded-xl transition-all text-white/70 hover:text-red-400 hover:bg-red-500/10"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                {/* Info Panel */}
                {showInfo && (
                  <div className="absolute top-3 left-3 right-3 p-3 rounded-xl bg-black/70 backdrop-blur-md border border-white/10 text-xs text-zinc-300 space-y-1">
                    <div>
                      <span className="text-zinc-500">Provider:</span> {imageDetails.provider}
                    </div>
                    <div>
                      <span className="text-zinc-500">Model:</span> {imageDetails.model}
                    </div>
                    <div>
                      <span className="text-zinc-500">Dimensions:</span> {imageDetails.dimensions}
                    </div>
                    <div>
                      <span className="text-zinc-500">Duration:</span> {imageDetails.duration}
                    </div>
                    <div>
                      <span className="text-zinc-500">Seed:</span> {imageDetails.seed}
                    </div>
                    <div>
                      <span className="text-zinc-500">Steps:</span> {imageDetails.steps}
                    </div>
                    <div>
                      <span className="text-zinc-500">Upscaled:</span>{' '}
                      {isImageUpscaled ? 'Yes (4x)' : 'No'}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="aspect-square flex flex-col items-center justify-center text-zinc-600">
                {loading ? (
                  <>
                    <div className="w-12 h-12 border-4 border-zinc-800 border-t-orange-500 rounded-full animate-spin mb-3" />
                    <span className="text-zinc-400 font-mono text-lg">{elapsed.toFixed(1)}s</span>
                    <span className="text-zinc-600 text-sm mt-1">Creating your image...</span>
                  </>
                ) : (
                  <>
                    <ImageIcon className="w-12 h-12 text-zinc-700 mb-2" />
                    <span className="text-zinc-600 text-sm">Your image will appear here</span>
                  </>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Fullscreen Preview Modal */}
      <AnimatePresence>
        {isFullscreen && currentImageUrl && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/90 backdrop-blur-sm flex items-center justify-center overflow-hidden"
            onClick={closeFullscreen}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {/* Close button */}
            <button
              type="button"
              onClick={closeFullscreen}
              className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors z-10"
            >
              <X size={24} />
            </button>

            {/* Keyboard shortcuts hint */}
            <div className="absolute top-4 left-4 text-xs text-white/40 space-y-1">
              <div>Esc - Close</div>
              <div>+/- - Zoom</div>
              <div>0 - Reset</div>
              <div>Scroll - Zoom</div>
            </div>

            {/* Image container with wheel zoom and drag */}
            {isComparing && tempUpscaledUrl ? (
              /* Fullscreen Comparison Mode */
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="w-[90vw] h-[80vh] flex items-center justify-center"
                onClick={(e) => e.stopPropagation()}
              >
                <ImageComparison
                  beforeImage={currentImageUrl}
                  afterImage={tempUpscaledUrl}
                  beforeLabel="Original"
                  afterLabel="4x Upscaled"
                  className="max-w-full max-h-full"
                />
              </motion.div>
            ) : (
              /* Normal fullscreen with zoom/pan */
              <motion.div
                ref={imageContainerRef}
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className={`${scale > 1 ? 'cursor-grab' : 'cursor-default'} ${isDragging ? 'cursor-grabbing' : ''}`}
                style={{
                  transform: `scale(${scale}) translate(${position.x / scale}px, ${position.y / scale}px)`,
                }}
                onClick={(e) => e.stopPropagation()}
                onWheel={handleWheel}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
              >
                <img
                  src={currentImageUrl}
                  alt="Preview"
                  className={`max-w-[90vw] max-h-[80vh] object-contain rounded-lg shadow-2xl transition-[filter] duration-300 select-none ${
                    isBlurred ? 'blur-xl' : ''
                  }`}
                  draggable={false}
                />
              </motion.div>
            )}

            {/* Bottom toolbar */}
            <div
              className="absolute bottom-6 left-1/2 -translate-x-1/2 pointer-events-none"
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => e.stopPropagation()}
              role="toolbar"
            >
              <div className="pointer-events-auto flex items-center gap-1 p-1.5 rounded-2xl bg-black/60 backdrop-blur-md border border-white/10 shadow-2xl">
                {isComparing ? (
                  /* Comparison mode toolbar */
                  <>
                    <button
                      type="button"
                      onClick={handleCancelComparison}
                      title="Cancel (Esc)"
                      className="flex items-center justify-center w-10 h-10 rounded-xl text-white/70 hover:text-white hover:bg-white/10 transition-all"
                    >
                      <X className="w-5 h-5" />
                    </button>
                    <div className="w-px h-5 bg-white/10" />
                    <span className="px-3 text-xs text-white/60">Drag slider to compare</span>
                    <div className="w-px h-5 bg-white/10" />
                    <button
                      type="button"
                      onClick={handleConfirmUpscale}
                      title="Apply upscaled image"
                      className="flex items-center justify-center w-10 h-10 rounded-xl text-green-400 hover:text-green-300 hover:bg-green-500/10 transition-all"
                    >
                      <Check className="w-5 h-5" />
                    </button>
                  </>
                ) : (
                  /* Normal fullscreen toolbar */
                  <>
                    {/* Info */}
                    <button
                      type="button"
                      onClick={() => setShowInfo(!showInfo)}
                      title="Details"
                      className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all ${
                        showInfo
                          ? 'bg-orange-600 text-white'
                          : 'text-white/70 hover:text-white hover:bg-white/10'
                      }`}
                    >
                      <Info className="w-5 h-5" />
                    </button>
                    <div className="w-px h-5 bg-white/10" />

                    {/* Zoom controls */}
                    <button
                      type="button"
                      onClick={handleZoomOut}
                      disabled={scale <= 1}
                      title="Zoom out (-)"
                      className="flex items-center justify-center w-10 h-10 rounded-xl text-white/70 hover:text-white hover:bg-white/10 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      <ZoomOut className="w-5 h-5" />
                    </button>
                    <button
                      type="button"
                      onClick={handleResetZoom}
                      title="Reset zoom (0)"
                      className="flex items-center justify-center px-2 h-10 rounded-xl text-white/70 hover:text-white hover:bg-white/10 transition-all text-xs font-medium min-w-[3rem]"
                    >
                      {Math.round(scale * 100)}%
                    </button>
                    <button
                      type="button"
                      onClick={handleZoomIn}
                      disabled={scale >= 8}
                      title="Zoom in (+)"
                      className="flex items-center justify-center w-10 h-10 rounded-xl text-white/70 hover:text-white hover:bg-white/10 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      <ZoomIn className="w-5 h-5" />
                    </button>
                    <div className="w-px h-5 bg-white/10" />

                    {/* 4x Upscale */}
                    <button
                      type="button"
                      onClick={handleUpscaleWithCompare}
                      disabled={isUpscaling || isImageUpscaled}
                      title={
                        isUpscaling
                          ? 'Upscaling...'
                          : isImageUpscaled
                            ? 'Already upscaled'
                            : '4x Upscale'
                      }
                      className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all ${
                        isImageUpscaled
                          ? 'text-orange-400 bg-orange-500/10'
                          : 'text-white/70 hover:text-white hover:bg-white/10'
                      } disabled:cursor-not-allowed`}
                    >
                      {isUpscaling ? (
                        <Loader2 className="w-5 h-5 animate-spin text-orange-400" />
                      ) : (
                        <span className="text-xs font-bold">4x</span>
                      )}
                    </button>
                    <div className="w-px h-5 bg-white/10" />

                    {/* Blur Toggle */}
                    <button
                      type="button"
                      onClick={() => setIsBlurred(!isBlurred)}
                      title="Toggle Blur"
                      className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all ${
                        isBlurred
                          ? 'text-orange-400 bg-white/10'
                          : 'text-white/70 hover:text-white hover:bg-white/10'
                      }`}
                    >
                      {isBlurred ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                    <div className="w-px h-5 bg-white/10" />

                    {/* Download */}
                    <button
                      type="button"
                      onClick={handleDownload}
                      title="Download"
                      className="flex items-center justify-center w-10 h-10 rounded-xl text-white/70 hover:text-white hover:bg-white/10 transition-all"
                    >
                      <Download className="w-5 h-5" />
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Info overlay in fullscreen */}
            {showInfo && imageDetails && !isComparing && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                className="absolute top-20 left-4 p-4 rounded-xl bg-zinc-900/90 border border-zinc-700 text-sm text-zinc-300 space-y-2 max-w-xs"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between">
                  <span className="text-zinc-500">Provider</span>
                  <span>{imageDetails.provider}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Model</span>
                  <span>{imageDetails.model}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Size</span>
                  <span>{imageDetails.dimensions}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Duration</span>
                  <span>{imageDetails.duration}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Seed</span>
                  <span className="font-mono">{imageDetails.seed}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Upscaled</span>
                  <span>{isImageUpscaled ? 'Yes (4x)' : 'No'}</span>
                </div>
              </motion.div>
            )}

            {/* Upscaling loading overlay */}
            {isUpscaling && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 bg-black/50 flex items-center justify-center z-20"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex flex-col items-center gap-4 p-6 rounded-2xl bg-zinc-900/90 border border-zinc-700">
                  <Loader2 className="w-10 h-10 animate-spin text-orange-400" />
                  <span className="text-white text-sm">Upscaling to 4x...</span>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
