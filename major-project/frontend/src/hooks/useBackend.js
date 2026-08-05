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
