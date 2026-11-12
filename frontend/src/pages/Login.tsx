import { useState } from "react";
import ParticleText from "../components/ParticleText";
import { AUTH_MARKER } from "../api";
import "./login.css";

const LOGIN_TIMEOUT_MS = 10_000;

async function fetchLogin(url: string, init: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), LOGIN_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("后端响应超时，请确认 8765 端口的 MindFlow 后端已启动");
    }
    throw new Error("无法连接 MindFlow 后端，请确认 8765 端口可访问");
  } finally {
    window.clearTimeout(timeout);
  }
}

interface LoginProps {
  bootstrapError?: string | null;
  onClearBootstrapError?: () => void;
}

export default function Login({ bootstrapError = null, onClearBootstrapError }: LoginProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleLogin = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchLogin(
        import.meta.env.DEV ? "/api/v1/auth/bootstrap/ticket" : "/api/v1/auth/local-session", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-MindFlow-Login": "1" },
        credentials: "include",
        mode: "same-origin",
        body: "{}",
      });
      if (!response.ok) {
        if (response.status === 403) {
          throw new Error("请在本机 MindFlow 页面登录，或双击桌面的 MindFlow");
        }
        if (response.status === 502 || response.status === 503) {
          throw new Error(`后端尚未就绪 (${response.status})：请确认 MindFlow 后端已启动`);
        }
        throw new Error(`登录失败 (${response.status})，请重新打开 MindFlow 后重试`);
      }
      if (import.meta.env.DEV) {
        const { ticket } = await response.json();
        const exchanged = await fetchLogin("/api/v1/auth/bootstrap", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ ticket }),
        });
        if (!exchanged.ok) throw new Error(`认证失败 (${exchanged.status})，请重新点击进入`);
      }

      window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
      localStorage.setItem(AUTH_MARKER, "1");
      setSuccess(true);
      window.location.reload();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "登录失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mf-login-page">
      <ParticleText />
      <div className="login-overlay" />

      <div className="login-card">
        <div className="login-card-header">
          <h1 className="login-title">欢迎回来！</h1>
          <p className="login-subtitle">进入您的 MindFlow 本地工作台</p>
        </div>

        <form className="login-form" onSubmit={(e) => {
          e.preventDefault();
          void handleLogin();
        }}>
          <div className="info-box login-note">
            本机工作台，无需账号密码。
          </div>

          {bootstrapError && (
            <div className="login-error login-error-row" role="alert">
              认证失败：{bootstrapError}
              {onClearBootstrapError && (
                <button type="button" className="btn btn-sm btn-ghost" onClick={onClearBootstrapError}>
                  关闭
                </button>
              )}
            </div>
          )}

          <button className="login-btn" type="submit" disabled={loading || success}>
            {loading ? "认证中…" : success ? "认证成功，正在进入…" : "进入 MindFlow"}
          </button>
        </form>

        {error && (
          <div className="login-error login-error-row" role="alert">
            {error}
            <button type="button" className="btn btn-sm btn-ghost" onClick={handleLogin} disabled={loading}>
              重试
            </button>
          </div>
        )}

        {/* §8.17.3: the dev shortcut shares its handler with "进入 MindFlow" and
            only confuses a real user, so it exists in a dev build only. */}
        {import.meta.env.DEV && (
          <div className="login-divider">
            <div className="login-divider-label">开发模式</div>
            <button
              type="button"
              className="login-btn login-btn--ghost"
              onClick={handleLogin}
              disabled={loading || success}
            >
              {loading ? "认证中…" : success ? "认证成功" : "Dev 登录（本地调试）"}
            </button>
          </div>
        )}

        <div className="login-footer">
          退出或认证失效后，可再次点击进入。
        </div>
      </div>

      <div className="introduction">MindFlow&nbsp;&nbsp;&nbsp;一款注意力分析工具软件</div>
    </div>
  );
}
