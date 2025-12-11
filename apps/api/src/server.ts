/**
 * Z-Image API - Node.js Server
 *
 * Standalone Node.js server using @hono/node-server.
 * Used for local development and Docker/Node.js deployments.
 */

import { serve } from '@hono/node-server'
import { createApp } from './app'
import { getConfig } from './config'

const config = getConfig()
const app = createApp({ corsOrigins: config.corsOrigins })

console.log('Starting Z-Image API server...')
console.log(`Environment: ${config.environment}`)
console.log(`CORS Origins: ${config.corsOrigins.join(', ') || '(none)'}`)

serve({
  fetch: app.fetch,
  port: config.port,
})

console.log(`Server is running on http://localhost:${config.port}`)
console.log('API endpoints:')
console.log('  GET  /api/         - Health check')
console.log('  POST /api/generate - Image generation')
console.log('  POST /api/upscale  - Image upscaling')
