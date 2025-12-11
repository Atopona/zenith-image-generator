/**
 * Netlify Functions - API Handler
 *
 * This file handles all /api/* routes when deployed to Netlify.
 * Uses Netlify Functions with streaming support.
 */

import { handle } from '@hono/node-server/vercel'
import type { Config } from '@netlify/functions'
import { createApp } from '../../api/src/app'

// Parse CORS origins from environment variable
const corsOrigins = process.env.CORS_ORIGINS
  ? process.env.CORS_ORIGINS.split(',').map((origin) => origin.trim())
  : ['http://localhost:5173', 'http://localhost:3000']

const app = createApp({ corsOrigins })

// Netlify Functions configuration
export const config: Config = {
  path: '/api/*',
}

// Use node-server vercel adapter which works with Netlify
export default handle(app)
