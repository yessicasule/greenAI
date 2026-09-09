/*
 * Backend client. Contract: major-project/contracts/api-spec.yaml
 *
 * BASE defaults to the relative '/api' prefix, which Vite's dev server and
 * `vite preview` both proxy to the FastAPI backend (see vite.config.js).
 * Going through the proxy keeps every request same-origin, so the dashboard
 * does not depend on api.py's CORS allow-list matching whatever port Vite
 * happened to pick — previously this was a hardcoded
 * 'http://localhost:8000', which silently broke the moment the backend ran
 * on another port/host or Vite fell back off 5173.
 *
 * Override with VITE_API_BASE (see .env.example) to point a built bundle at
 * a backend directly, e.g. VITE_API_BASE=http://192.168.1.20:8000
 */
const BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning': 'true',
    },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  health: () => request('/health'),
  route: (body) => request('/route', { method: 'POST', body: JSON.stringify(body) }),
  infer: (body) => request('/infer', { method: 'POST', body: JSON.stringify(body) }),
  results: () => request('/results'),
  energy: () => request('/energy'),
  accuracy: () => request('/accuracy'),
}
