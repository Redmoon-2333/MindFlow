import { useState, useEffect, useCallback } from 'react'
import type { ApiResponse } from '../types'

const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1'

export function useApi<T>(url: string, enabled: boolean = true) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const fetcher = useCallback(async () => {
    if (!enabled) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}${url}`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      }
      const json: ApiResponse<T> = await res.json()
      if (json.code !== 0) {
        throw new Error(json.message || 'API error')
      }
      setData(json.data)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [url, enabled])

  useEffect(() => {
    fetcher()
  }, [fetcher])

  return { data, loading, error, refetch: fetcher }
}
