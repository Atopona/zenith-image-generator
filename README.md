# Z-Image

A modern Text-to-Image generation web application powered by Gitee AI API (z-image-turbo model).

![Dark Mode UI](https://img.shields.io/badge/UI-Dark%20Mode-1a1a1a)
![Cloudflare Pages](https://img.shields.io/badge/Deploy-Cloudflare%20Pages-F38020)
![React](https://img.shields.io/badge/React-19-61DAFB)
![Hono](https://img.shields.io/badge/Hono-4-E36002)

## Features

- Dark mode Gradio-style UI
- Multiple aspect ratio presets (1:1, 16:9, 9:16, 4:3, 3:4, etc.)
- Adjustable inference steps and dimensions
- Real-time generation progress with timer
- One-click image download (JPG)
- API key persistence in browser
- Responsive design (mobile & desktop)

## Tech Stack

- **Frontend**: React 19, Vite, Tailwind CSS, shadcn/ui
- **Backend**: Hono (TypeScript)
- **Deployment**: Cloudflare Pages + Functions
- **API**: Gitee AI (z-image-turbo)

## Project Structure

```
z-image/
├── apps/
│   ├── web/                    # Frontend application
│   │   ├── src/
│   │   │   ├── pages/          # Page components
│   │   │   └── components/ui/  # shadcn/ui components
│   │   ├── functions/api/      # Cloudflare Pages Functions
│   │   └── dist/               # Build output
│   └── api/                    # Hono API (shared with functions)
│       └── src/index.ts
├── package.json
├── pnpm-workspace.yaml
└── turbo.json
```

## Prerequisites

- Node.js 18+
- pnpm 9+
- Gitee AI API Key ([Get one here](https://ai.gitee.com))

## Local Development

### 1. Clone the repository

```bash
git clone https://github.com/your-username/z-image.git
cd z-image
```

### 2. Install dependencies

```bash
pnpm install
```

### 3. Start development server

**Option A: Full stack with Cloudflare Pages (Recommended)**

```bash
cd apps/web
pnpm add wrangler -D
npx wrangler pages dev --port 5173 -- pnpm dev
```

**Option B: Frontend only (requires separate API)**

```bash
# Terminal 1: Start API
pnpm dev:api

# Terminal 2: Start Web (update .env first)
# Set VITE_API_URL=http://localhost:8787 in apps/web/.env
pnpm dev:web
```

### 4. Open browser

Navigate to `http://localhost:5173`

## Self-Hosting Deployment

### Option 1: Cloudflare Pages (Recommended)

Deploy both frontend and API together with zero configuration.

#### Using Cloudflare Dashboard

1. Push your code to GitHub/GitLab

2. Go to [Cloudflare Dashboard](https://dash.cloudflare.com) → **Pages** → **Create a project**

3. Connect your Git repository

4. Configure build settings:
   | Setting | Value |
   |---------|-------|
   | Root directory | `apps/web` |
   | Build command | `pnpm build` |
   | Output directory | `dist` |

5. Click **Save and Deploy**

6. Your app will be available at `https://your-project.pages.dev`

#### Using Wrangler CLI

```bash
# Install Wrangler globally
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Deploy from apps/web directory
cd apps/web
pnpm build
wrangler pages deploy dist --project-name z-image
```

### Option 2: Vercel (Frontend) + Cloudflare Workers (API)

#### Deploy API to Cloudflare Workers

```bash
cd apps/api

# Update wrangler.toml with your CORS origins
# CORS_ORIGINS = "https://your-app.vercel.app"

wrangler deploy
```

Note your Workers URL: `https://z-image-api.your-account.workers.dev`

#### Deploy Frontend to Vercel

1. Go to [Vercel Dashboard](https://vercel.com/dashboard) → **New Project**

2. Import your Git repository

3. Configure:
   | Setting | Value |
   |---------|-------|
   | Root Directory | `apps/web` |
   | Build Command | `pnpm build` |
   | Output Directory | `dist` |

4. Add Environment Variable:
   | Name | Value |
   |------|-------|
   | `VITE_API_URL` | `https://z-image-api.your-account.workers.dev` |

5. Deploy

### Option 3: Netlify (Frontend) + Cloudflare Workers (API)

#### Deploy API to Cloudflare Workers

Same as Option 2 above.

#### Deploy Frontend to Netlify

1. Go to [Netlify Dashboard](https://app.netlify.com) → **Add new site**

2. Import your Git repository

3. Configure:
   | Setting | Value |
   |---------|-------|
   | Base directory | `apps/web` |
   | Build command | `pnpm build` |
   | Publish directory | `apps/web/dist` |

4. Add Environment Variable in **Site settings** → **Environment variables**:
   | Name | Value |
   |------|-------|
   | `VITE_API_URL` | `https://z-image-api.your-account.workers.dev` |

5. Trigger redeploy

## Security

### API Key Storage

Your Gitee AI API key is stored securely in the browser using **AES-256-GCM encryption**:

- The key is encrypted before being saved to localStorage
- Encryption key is derived using PBKDF2 (100,000 iterations) from browser fingerprint
- Even if localStorage is accessed, the API key cannot be read without the same browser environment
- Changing browsers or clearing browser data will require re-entering the API key

**Implementation details** (`src/lib/crypto.ts`):
- Uses Web Crypto API (native browser cryptography)
- AES-256-GCM for authenticated encryption
- Random IV for each encryption operation
- Browser fingerprint includes: User-Agent, language, screen dimensions

**Note**: While this provides protection against casual access and XSS attacks reading raw values, for maximum security in shared environments, consider:
- Using a private/incognito window
- Clearing browser data after use
- Self-hosting with server-side API key storage

## Environment Variables

### Frontend (`apps/web/.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | API base URL. Leave empty for Cloudflare Pages deployment | `` |

### API (`apps/api/wrangler.toml`)

| Variable | Description | Default |
|----------|-------------|---------|
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:5173,http://localhost:3000` |

## API Reference

### `POST /api/generate`

Generate an image from text prompt.

**Headers:**
```
Content-Type: application/json
X-API-Key: your-gitee-ai-api-key
```

**Request Body:**
```json
{
  "prompt": "A beautiful sunset over mountains",
  "negative_prompt": "low quality, blurry",
  "model": "z-image-turbo",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 9
}
```

**Response:**
```json
{
  "url": "https://...",
  "b64_json": "base64-encoded-image-data"
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Image description (max 10000 chars) |
| `negative_prompt` | string | No | `""` | What to avoid in the image |
| `model` | string | No | `z-image-turbo` | Model name |
| `width` | number | No | `1024` | Image width (256-2048) |
| `height` | number | No | `1024` | Image height (256-2048) |
| `num_inference_steps` | number | No | `9` | Generation steps (1-50) |

## Supported Aspect Ratios

| Ratio | Dimensions |
|-------|------------|
| 1:1 | 256×256, 512×512, 1024×1024, 2048×2048 |
| 4:3 | 1152×896, 2048×1536 |
| 3:4 | 768×1024, 1536×2048 |
| 3:2 | 2048×1360 |
| 2:3 | 1360×2048 |
| 16:9 | 1024×576, 2048×1152 |
| 9:16 | 576×1024, 1152×2048 |

## Troubleshooting

### API Key not saving
- Make sure your browser allows localStorage
- Check if you're in private/incognito mode

### CORS errors
- For Cloudflare Pages: Should work automatically
- For separate deployments: Update `CORS_ORIGINS` in `apps/api/wrangler.toml`

### Build failures
- Ensure Node.js 18+ and pnpm 9+ are installed
- Run `pnpm install` to update dependencies

## License

MIT

## Acknowledgments

- [Gitee AI](https://ai.gitee.com) for the z-image-turbo model
- [shadcn/ui](https://ui.shadcn.com) for UI components
- [Hono](https://hono.dev) for the lightweight web framework
