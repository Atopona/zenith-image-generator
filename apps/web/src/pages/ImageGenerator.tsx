import { useState, useEffect, useRef } from 'react'
import { encryptAndStore, decryptFromStore } from '@/lib/crypto'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Slider } from '@/components/ui/slider'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { toast } from 'sonner'
import { ImageIcon, Download } from 'lucide-react'

const DEFAULT_PROMPT = "一张虚构的英语电影《回忆之味》（The Taste of Memory）的电影海报。场景设置在一个质朴的19世纪风格厨房里。画面中央，一位红棕色头发、留着小胡子的中年男子（演员阿瑟·彭哈利根饰）站在一张木桌后，他身穿白色衬衫、黑色马甲和米色围裙，正看着一位女士，手中拿着一大块生红肉，下方是一个木制切菜板。在他的右边，一位梳着高髻的黑发女子（演员埃莉诺·万斯饰）倚靠在桌子上，温柔地对他微笑。她穿着浅色衬衫和一条上白下蓝的长裙。桌上除了放有切碎的葱和卷心菜丝的切菜板外，还有一个白色陶瓷盘、新鲜香草，左侧一个木箱上放着一串深色葡萄。背景是一面粗糙的灰白色抹灰墙，墙上挂着一幅风景画。最右边的一个台面上放着一盏复古油灯。海报上有大量的文字信息。左上角是白色的无衬线字体\"ARTISAN FILMS PRESENTS\"，其下方是\"ELEANOR VANCE\"和\"ACADEMY AWARD® WINNER\"。右上角写着\"ARTHUR PENHALIGON\"和\"GOLDEN GLOBE® AWARD WINNER\"。顶部中央是圣丹斯电影节的桂冠标志，下方写着\"SUNDANCE FILM FESTIVAL GRAND JURY PRIZE 2024\"。主标题\"THE TASTE OF MEMORY\"以白色的大号衬线字体醒目地显示在下半部分。标题下方注明了\"A FILM BY Tongyi Interaction Lab\"。底部区域用白色小字列出了完整的演职员名单，包括\"SCREENPLAY BY ANNA REID\"、\"CULINARY DIRECTION BY JAMES CARTER\"以及Artisan Films、Riverstone Pictures和Heritage Media等众多出品公司标志。整体风格是写实主义，采用温暖柔和的灯光方案，营造出一种亲密的氛围。色调以棕色、米色和柔和的绿色等大地色系为主。两位演员的身体都在腰部被截断"
const DEFAULT_NEGATIVE_PROMPT = "低质量, 丑陋, 畸形, 模糊, 多余的肢体, 错误的文本"

const ASPECT_RATIOS: Record<string, [number, number]> = {
  "1:1 (256×256)": [256, 256],
  "1:1 (512×512)": [512, 512],
  "1:1 (1024×1024)": [1024, 1024],
  "1:1 (2048×2048)": [2048, 2048],
  "4:3 (1152×896)": [1152, 896],
  "4:3 (2048×1536)": [2048, 1536],
  "3:4 (768×1024)": [768, 1024],
  "3:4 (1536×2048)": [1536, 2048],
  "3:2 (2048×1360)": [2048, 1360],
  "2:3 (1360×2048)": [1360, 2048],
  "16:9 (1024×576)": [1024, 576],
  "16:9 (2048×1152)": [2048, 1152],
  "9:16 (576×1024)": [576, 1024],
  "9:16 (1152×2048)": [1152, 2048],
}

export default function ImageGenerator() {
  const [apiKey, setApiKey] = useState('')
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [negativePrompt, setNegativePrompt] = useState(DEFAULT_NEGATIVE_PROMPT)
  const [model, setModel] = useState('z-image-turbo')
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [steps, setSteps] = useState(9)
  const [loading, setLoading] = useState(false)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [status, setStatus] = useState('Ready.')
  const [elapsed, setElapsed] = useState(0)
  const initialized = useRef(false)

  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true
      decryptFromStore().then(setApiKey)
    }
  }, [])

  useEffect(() => {
    if (initialized.current) {
      encryptAndStore(apiKey)
    }
  }, [apiKey])

  useEffect(() => {
    if (!loading) return
    setElapsed(0)
    const timer = setInterval(() => setElapsed(e => e + 0.1), 100)
    return () => clearInterval(timer)
  }, [loading])

  const handleAspectRatioChange = (ratio: string) => {
    const dims = ASPECT_RATIOS[ratio]
    if (dims) {
      setWidth(dims[0])
      setHeight(dims[1])
    }
  }

  const addStatus = (msg: string) => {
    setStatus(prev => prev + '\n' + msg)
  }

  const handleDownload = () => {
    if (!imageUrl) return
    const a = document.createElement('a')
    a.href = imageUrl
    a.download = `generated-${Date.now()}.jpg`
    a.click()
  }

  const handleGenerate = async () => {
    if (!apiKey) {
      toast.error('Please enter your API Key')
      return
    }

    setLoading(true)
    setImageUrl(null)
    setStatus('Initializing client...')

    try {
      addStatus('Sending request to API...')
      const res = await fetch(`${import.meta.env.VITE_API_URL || ''}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey,
        },
        body: JSON.stringify({
          prompt,
          negative_prompt: negativePrompt,
          model,
          width,
          height,
          num_inference_steps: steps,
        }),
      })

      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.error || 'Failed to generate image')
      }

      addStatus('Image generated successfully!')

      if (data.url) {
        setImageUrl(data.url)
      } else if (data.b64_json) {
        setImageUrl(`data:image/png;base64,${data.b64_json}`)
      }

      toast.success('Image generated!')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'An error occurred'
      addStatus(`Error: ${msg}`)
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-white">Text-to-Image Generation UI</h1>
          <p className="text-zinc-400 mt-1">Enter your API key and a prompt to generate an image using the Gitee AI API.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Controls */}
          <div className="lg:col-span-2">
            <Card className="bg-zinc-900 border-zinc-800">
              <CardContent className="p-6 space-y-5">
                <div>
                  <Label htmlFor="apiKey" className="text-zinc-300">API Key</Label>
                  <Input
                    id="apiKey"
                    type="password"
                    placeholder="Enter your Gitee AI API Key here..."
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    className="mt-1.5 bg-zinc-800 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
                  />
                </div>

                <div>
                  <Label htmlFor="prompt" className="text-zinc-300">Prompt</Label>
                  <Textarea
                    id="prompt"
                    rows={5}
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="mt-1.5 bg-zinc-800 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
                  />
                </div>

                <div>
                  <Label htmlFor="negativePrompt" className="text-zinc-300">Negative Prompt</Label>
                  <Textarea
                    id="negativePrompt"
                    rows={2}
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    className="mt-1.5 bg-zinc-800 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
                  />
                </div>

                <Separator className="bg-zinc-800" />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="model" className="text-zinc-300">Model</Label>
                    <Input
                      id="model"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="mt-1.5 bg-zinc-800 border-zinc-700 text-zinc-100"
                    />
                  </div>
                  <div>
                    <Label className="text-zinc-300 flex items-center gap-2">
                      Inference Steps <Badge variant="secondary" className="bg-zinc-700 text-zinc-200">{steps}</Badge>
                    </Label>
                    <Slider
                      value={[steps]}
                      onValueChange={(v) => setSteps(v[0])}
                      min={1}
                      max={50}
                      step={1}
                      className="mt-3"
                    />
                  </div>
                </div>

                <Separator className="bg-zinc-800" />

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <Label className="text-zinc-300">Aspect Ratio</Label>
                    <Select onValueChange={handleAspectRatioChange} defaultValue="1:1 (1024×1024)">
                      <SelectTrigger className="mt-1.5 bg-zinc-800 border-zinc-700 text-zinc-100">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-zinc-800 border-zinc-700">
                        {Object.keys(ASPECT_RATIOS).map((ratio) => (
                          <SelectItem key={ratio} value={ratio} className="text-zinc-100 focus:bg-zinc-700">
                            {ratio}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-zinc-300 flex items-center gap-2">
                      Width <Badge variant="secondary" className="bg-zinc-700 text-zinc-200">{width}</Badge>
                    </Label>
                    <Slider
                      value={[width]}
                      onValueChange={(v) => setWidth(v[0])}
                      min={512}
                      max={2048}
                      step={64}
                      className="mt-3"
                    />
                  </div>
                  <div>
                    <Label className="text-zinc-300 flex items-center gap-2">
                      Height <Badge variant="secondary" className="bg-zinc-700 text-zinc-200">{height}</Badge>
                    </Label>
                    <Slider
                      value={[height]}
                      onValueChange={(v) => setHeight(v[0])}
                      min={512}
                      max={2048}
                      step={64}
                      className="mt-3"
                    />
                  </div>
                </div>

                <Button
                  onClick={handleGenerate}
                  disabled={loading}
                  className="w-full bg-orange-500 hover:bg-orange-600 text-white font-semibold h-11"
                >
                  {loading ? 'Generating...' : 'Generate Image'}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Output */}
          <div className="lg:col-span-1 space-y-4">
            <Card className="bg-zinc-900 border-zinc-800">
              <CardHeader className="pb-3">
                <CardTitle className="text-zinc-200 text-lg">Generated Image</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {imageUrl ? (
                  <>
                    <img src={imageUrl} alt="Generated" className="w-full rounded-md" />
                    <Button onClick={handleDownload} variant="outline" className="w-full border-zinc-700 text-zinc-300 hover:bg-zinc-800">
                      <Download className="w-4 h-4 mr-2" /> Download JPG
                    </Button>
                  </>
                ) : (
                  <div className="aspect-square bg-zinc-800 rounded-md flex flex-col items-center justify-center text-zinc-500 border border-zinc-700">
                    {loading ? (
                      <>
                        <div className="w-10 h-10 border-4 border-zinc-600 border-t-orange-500 rounded-full animate-spin mb-3" />
                        <span className="text-zinc-300 font-mono">{elapsed.toFixed(1)}s</span>
                      </>
                    ) : (
                      <>
                        <ImageIcon className="w-12 h-12 mb-2" />
                        <span>No image yet</span>
                      </>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="bg-zinc-900 border-zinc-800">
              <CardHeader className="pb-3">
                <CardTitle className="text-zinc-200 text-lg">Status</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-32 w-full rounded-md border border-zinc-700 bg-zinc-800 p-3">
                  <pre className="text-sm text-zinc-400 whitespace-pre-wrap font-mono">{status}</pre>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
