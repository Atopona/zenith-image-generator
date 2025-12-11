/**
 * Z-Image API - Cloudflare Workers Entry Point
 *
 * This is the entry point for deploying the API to Cloudflare Workers.
 * It exports the Hono app as the default export which Workers expects.
 */

import { createApp } from './app'
import { getCorsOriginsFromBindings } from './config'

export interface Env {
  CORS_ORIGINS?: string
}

// Create app factory for Workers with bindings support
const handler = {
  fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const corsOrigins = getCorsOriginsFromBindings(env)
    const app = createApp({ corsOrigins })
    return app.fetch(request, env, ctx)
  },
}

export default handler
