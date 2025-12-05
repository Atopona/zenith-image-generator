# Z-Image

基于 Gitee AI API (z-image-turbo 模型) 的现代文生图 Web 应用。

![Dark Mode UI](https://img.shields.io/badge/UI-Dark%20Mode-1a1a1a)
![Cloudflare Pages](https://img.shields.io/badge/Deploy-Cloudflare%20Pages-F38020)
![React](https://img.shields.io/badge/React-19-61DAFB)
![Hono](https://img.shields.io/badge/Hono-4-E36002)

## 功能特性

- 深色模式 Gradio 风格 UI
- 多种宽高比预设 (1:1, 16:9, 9:16, 4:3, 3:4 等)
- 可调节推理步数和尺寸
- 实时生成进度与计时器
- 一键下载图片 (JPG)
- API Key 浏览器持久化存储
- 响应式设计 (移动端 & 桌面端)

## 技术栈

- **前端**: React 19, Vite, Tailwind CSS, shadcn/ui
- **后端**: Hono (TypeScript)
- **部署**: Cloudflare Pages + Functions
- **API**: Gitee AI (z-image-turbo)

## 项目结构

```
z-image/
├── apps/
│   ├── web/                    # 前端应用
│   │   ├── src/
│   │   │   ├── pages/          # 页面组件
│   │   │   └── components/ui/  # shadcn/ui 组件
│   │   ├── functions/api/      # Cloudflare Pages Functions
│   │   └── dist/               # 构建输出
│   └── api/                    # Hono API (与 functions 共享)
│       └── src/index.ts
├── package.json
├── pnpm-workspace.yaml
└── turbo.json
```

## 前置要求

- Node.js 18+
- pnpm 9+
- Gitee AI API Key ([点此获取](https://ai.gitee.com))

## 本地开发

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/z-image.git
cd z-image
```

### 2. 安装依赖

```bash
pnpm install
```

### 3. 启动开发服务器

**方式 A: 使用 Cloudflare Pages 全栈开发 (推荐)**

```bash
cd apps/web
pnpm add wrangler -D
npx wrangler pages dev --port 5173 -- pnpm dev
```

**方式 B: 仅前端 (需要单独启动 API)**

```bash
# 终端 1: 启动 API
pnpm dev:api

# 终端 2: 启动 Web (先更新 .env)
# 在 apps/web/.env 中设置 VITE_API_URL=http://localhost:8787
pnpm dev:web
```

### 4. 打开浏览器

访问 `http://localhost:5173`

## 自托管部署

### 方式 1: Cloudflare Pages (推荐)

前端和 API 一起部署，零配置。

#### 使用 Cloudflare Dashboard

1. 将代码推送到 GitHub/GitLab

2. 前往 [Cloudflare Dashboard](https://dash.cloudflare.com) → **Pages** → **Create a project**

3. 连接你的 Git 仓库

4. 配置构建设置:
   | 设置 | 值 |
   |---------|-------|
   | Root directory | `apps/web` |
   | Build command | `pnpm build` |
   | Output directory | `dist` |

5. 点击 **Save and Deploy**

6. 应用将部署到 `https://your-project.pages.dev`

#### 使用 Wrangler CLI

```bash
# 全局安装 Wrangler
npm install -g wrangler

# 登录 Cloudflare
wrangler login

# 从 apps/web 目录部署
cd apps/web
pnpm build
wrangler pages deploy dist --project-name z-image
```

### 方式 2: Vercel (前端) + Cloudflare Workers (API)

#### 部署 API 到 Cloudflare Workers

```bash
cd apps/api

# 更新 wrangler.toml 中的 CORS 源
# CORS_ORIGINS = "https://your-app.vercel.app"

wrangler deploy
```

记下 Workers URL: `https://z-image-api.your-account.workers.dev`

#### 部署前端到 Vercel

1. 前往 [Vercel Dashboard](https://vercel.com/dashboard) → **New Project**

2. 导入你的 Git 仓库

3. 配置:
   | 设置 | 值 |
   |---------|-------|
   | Root Directory | `apps/web` |
   | Build Command | `pnpm build` |
   | Output Directory | `dist` |

4. 添加环境变量:
   | 名称 | 值 |
   |------|-------|
   | `VITE_API_URL` | `https://z-image-api.your-account.workers.dev` |

5. 部署

### 方式 3: Netlify (前端) + Cloudflare Workers (API)

#### 部署 API 到 Cloudflare Workers

同方式 2。

#### 部署前端到 Netlify

1. 前往 [Netlify Dashboard](https://app.netlify.com) → **Add new site**

2. 导入你的 Git 仓库

3. 配置:
   | 设置 | 值 |
   |---------|-------|
   | Base directory | `apps/web` |
   | Build command | `pnpm build` |
   | Publish directory | `apps/web/dist` |

4. 在 **Site settings** → **Environment variables** 添加环境变量:
   | 名称 | 值 |
   |------|-------|
   | `VITE_API_URL` | `https://z-image-api.your-account.workers.dev` |

5. 触发重新部署

## 安全性

### API Key 存储

你的 Gitee AI API Key 使用 **AES-256-GCM 加密** 安全存储在浏览器中:

- Key 在保存到 localStorage 前会被加密
- 加密密钥通过 PBKDF2 (100,000 次迭代) 从浏览器指纹派生
- 即使 localStorage 被访问，没有相同的浏览器环境也无法读取 API Key
- 更换浏览器或清除浏览器数据需要重新输入 API Key

**实现细节** (`src/lib/crypto.ts`):

- 使用 Web Crypto API (浏览器原生加密)
- AES-256-GCM 认证加密
- 每次加密使用随机 IV
- 浏览器指纹包括: User-Agent、语言、屏幕尺寸

**注意**: 虽然这提供了对普通访问和 XSS 攻击读取原始值的保护，但在共享环境中为了最大安全性，建议:

- 使用隐私/无痕窗口
- 使用后清除浏览器数据
- 自托管并使用服务端 API Key 存储

## 环境变量

### 前端 (`apps/web/.env`)

| 变量           | 描述                                      | 默认值 |
| -------------- | ----------------------------------------- | ------ |
| `VITE_API_URL` | API 基础 URL。Cloudflare Pages 部署时留空 | ``     |

### API (`apps/api/wrangler.toml`)

| 变量           | 描述             | 默认值                                        |
| -------------- | ---------------- | --------------------------------------------- |
| `CORS_ORIGINS` | 逗号分隔的允许源 | `http://localhost:5173,http://localhost:3000` |

## API 参考

### `POST /api/generate`

从文本提示生成图片。

**请求头:**

```
Content-Type: application/json
X-API-Key: your-gitee-ai-api-key
```

**请求体:**

```json
{
  "prompt": "美丽的山间日落",
  "negative_prompt": "低质量, 模糊",
  "model": "z-image-turbo",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 9
}
```

**响应:**

```json
{
  "url": "https://...",
  "b64_json": "base64-encoded-image-data"
}
```

**参数:**

| 字段                  | 类型   | 必填 | 默认值          | 描述                       |
| --------------------- | ------ | ---- | --------------- | -------------------------- |
| `prompt`              | string | 是   | -               | 图片描述 (最多 10000 字符) |
| `negative_prompt`     | string | 否   | `""`            | 图片中要避免的内容         |
| `model`               | string | 否   | `z-image-turbo` | 模型名称                   |
| `width`               | number | 否   | `1024`          | 图片宽度 (256-2048)        |
| `height`              | number | 否   | `1024`          | 图片高度 (256-2048)        |
| `num_inference_steps` | number | 否   | `9`             | 生成步数 (1-50)            |

## 支持的宽高比

| 比例 | 尺寸                                   |
| ---- | -------------------------------------- |
| 1:1  | 256×256, 512×512, 1024×1024, 2048×2048 |
| 4:3  | 1152×896, 2048×1536                    |
| 3:4  | 768×1024, 1536×2048                    |
| 3:2  | 2048×1360                              |
| 2:3  | 1360×2048                              |
| 16:9 | 1024×576, 2048×1152                    |
| 9:16 | 576×1024, 1152×2048                    |

## 故障排除

### API Key 无法保存

- 确保浏览器允许 localStorage
- 检查是否在隐私/无痕模式

### CORS 错误

- Cloudflare Pages: 应该自动工作
- 分离部署: 更新 `apps/api/wrangler.toml` 中的 `CORS_ORIGINS`

### 构建失败

- 确保安装了 Node.js 18+ 和 pnpm 9+
- 运行 `pnpm install` 更新依赖

## 许可证

MIT

## 致谢

- [Gitee AI](https://ai.gitee.com) 提供 z-image-turbo 模型
- [shadcn/ui](https://ui.shadcn.com) 提供 UI 组件
- [Hono](https://hono.dev) 提供轻量级 Web 框架
