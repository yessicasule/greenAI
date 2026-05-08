const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  health:   ()          => request('/health'),
  route:    (body)      => request('/route',    { method: 'POST', body: JSON.stringify(body) }),
  infer:    (body)      => request('/infer',    { method: 'POST', body: JSON.stringify(body) }),
  results:  ()          => request('/results'),
  energy:   ()          => request('/energy'),
  accuracy: ()          => request('/accuracy'),
}
