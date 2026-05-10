import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || '/ws/activities'

export function useWebSocket() {
  const [latestActivity, setLatestActivity] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectAttempts = useRef<number>(0)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = import.meta.env.DEV
      ? `${protocol}//${host}${WS_URL}`
      : `${protocol}//${host}${WS_URL}`

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data)
        if (data?.window_title) {
          setLatestActivity(JSON.stringify(data))
        }
      } catch {
        setLatestActivity(event.data)
      }
    }

    ws.onclose = () => {
      reconnectAttempts.current += 1
      const delay = Math.min(1000 * 2 ** reconnectAttempts.current, 30000)
      reconnectTimer.current = setTimeout(connect, delay)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [connect])

  return { latestActivity }
}
