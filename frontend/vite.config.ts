import { readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

function getMindflowDataDirs(): string[] {
  const homeDir = os.homedir()
  const defaultDataDirs = process.platform === 'win32'
    ? (() => {
        const localAppData = process.env.LOCALAPPDATA ?? path.join(homeDir, 'AppData', 'Local')
        const appDataRoot = path.join(localAppData, 'mindflow')
        return [path.join(appDataRoot, 'mindflow'), appDataRoot]
      })()
    : process.platform === 'darwin'
      ? [
          path.join(homeDir, 'Library', 'Application Support', 'mindflow', 'mindflow'),
          path.join(homeDir, 'Library', 'Application Support', 'mindflow'),
        ]
      : (() => {
          const dataHome = process.env.XDG_DATA_HOME ?? path.join(homeDir, '.local', 'share')
          const appDataRoot = path.join(dataHome, 'mindflow')
          return [path.join(appDataRoot, 'mindflow'), appDataRoot]
        })()
  const configuredDataDir = process.env.MINDFLOW_DATA_DIR?.trim()

  if (!configuredDataDir) return defaultDataDirs
  return [
    path.isAbsolute(configuredDataDir)
      ? configuredDataDir
      : path.join(defaultDataDirs[0], configuredDataDir),
  ]
}

function readMindflowToken(): string | undefined {
  for (const dataDir of getMindflowDataDirs()) {
    try {
      const token = readFileSync(path.join(dataDir, 'token'), 'utf8').trim()
      if (token) return token
    } catch {
      // The backend will keep rejecting the request if no local token exists.
    }
  }
  return undefined
}

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: Number.parseInt(process.env.MINDFLOW_FRONTEND_PORT ?? '4173', 10),
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
        ws: true,
        configure(proxy) {
          proxy.on('proxyReq', (proxyReq, req) => {
            const requestPath = req.url?.split('?')[0]
            if (req.method !== 'POST' || requestPath !== '/api/v1/auth/bootstrap/ticket') return

            const token = readMindflowToken()
            if (token) proxyReq.setHeader('Authorization', `Bearer ${token}`)
          })
        },
      },
    },
  },
})
