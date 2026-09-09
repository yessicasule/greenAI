import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// The dashboard talks to the backend through a relative '/api' prefix
// (src/lib/api.js), so every request is same-origin and CORS never enters
// the picture. This proxy is what makes that work; it is applied to BOTH
// `vite dev` and `vite preview` — `server.proxy` alone does not cover
// preview, so a production build served locally would 404 on every call.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_BACKEND_URL || 'http://localhost:8000'

  const proxy = {
    '/api': {
      target,
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  }

  return {
    plugins: [react()],
    server: { port: 5173, proxy },
    preview: { port: 4173, proxy },
  }
})
