import { useState, useEffect, useCallback } from 'react'
import { api } from '../lib/api'

export function useHealth() {
  const [online, setOnline] = useState(false)
  const [gpuReady, setGpuReady] = useState(false)

  useEffect(() => {
    async function check() {
      try {
        const data = await api.health()
        setOnline(true)
        setGpuReady(data.gpu_ready || false)
      } catch {
        setOnline(false)
        setGpuReady(false)
      }
    }
    check()
    const id = setInterval(check, 4000)
    return () => clearInterval(id)
  }, [])

  return { online, gpuReady }
}

export function useAnalytics() {
  const [data, setData] = useState({ traces: [], energy: [], accuracy: null })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [r, e, a] = await Promise.all([api.results(), api.energy(), api.accuracy()])
      setData({ traces: r.entries || [], energy: e.tiers || [], accuracy: a })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  return { ...data, loading, error, refresh }
}


/**
 * Loads the project's measurement record for the Analytics page.
 *
 * Each endpoint is fetched independently and its failure isolated: a
 * missing artifact must degrade that one panel to "no data yet", never
 * blank the whole page or — far worse — leave a stale number on screen
 * under a fresh-looking header.
 */
export function useEvidence() {
  const [data, setData] = useState({
    validation: null, experiments: null, energy: null,
    routing: null, routerQuality: null,
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    const calls = [
      ['validation', api.evidence.validation],
      ['experiments', api.evidence.experiments],
      ['energy', api.evidence.energy],
      ['routing', api.evidence.routing],
      ['routerQuality', api.evidence.routerQuality],
    ]
    const settled = await Promise.allSettled(calls.map(([, fn]) => fn()))
    const next = {}
    const failures = []
    settled.forEach((res, i) => {
      const key = calls[i][0]
      if (res.status === 'fulfilled') {
        next[key] = res.value
      } else {
        next[key] = null
        failures.push(key)
      }
    })
    setData(next)
    if (failures.length === calls.length) setError('Backend offline — no evidence loaded.')
    else if (failures.length) setError(`Could not load: ${failures.join(', ')}`)
    setLoading(false)
  }, [])

  useEffect(() => { refresh() }, [refresh])

  return { ...data, loading, error, refresh }
}
