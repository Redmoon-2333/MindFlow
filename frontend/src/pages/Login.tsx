import { useState } from "react";
import ParticleText from "../components/ParticleText";
import { AUTH_MARKER, AUTH_REQUIRED_EVENT } from "../api";
import "./login.css";

const DEV_LOGIN_TIMEOUT_MS = 10_000;

async function fetchDevLogin(url: string, init: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), DEV_LOGIN_TIMEOUT_MS);
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
  const [devLoading, setDevLoading] = useState(false);
  const [devError, setDevError] = useState<string | null>(null);
  const [devSuccess, setDevSuccess] = useState(false);

  const handleDevLogin = async () => {
    setDevLoading(true);
    setDevError(null);
    try {
      const tokenRes = await fetchDevLogin("/api/v1/auth/bootstrap/ticket", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
      });
      if (!tokenRes.ok) {
        if (tokenRes.status === 401) {
          throw new Error("获取票据失败 (401)：请从 MindFlow 启动器打开开发界面，或重启 Vite 开发服务器");
        }
        if (tokenRes.status === 502 || tokenRes.status === 503) {
          throw new Error(`后端尚未就绪 (${tokenRes.status})：请确认 8765 端口的 MindFlow 后端已启动`);
        }
        throw new Error(`获取票据失败 (${tokenRes.status})`);
      }
      const { ticket } = await tokenRes.json();

      const bootstrapRes = await fetchDevLogin("/api/v1/auth/bootstrap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ ticket }),
      });
      if (!bootstrapRes.ok) {
        throw new Error(
          bootstrapRes.status === 401
            ? "认证失败 (401)：票据已失效，请重新点击进入"
            : `认证失败 (${bootstrapRes.status})`,
        );
      }

      localStorage.setItem(AUTH_MARKER, "1");
      setDevSuccess(true);
      window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT));
      setTimeout(() => window.location.reload(), 500);
    } catch (e: unknown) {
      setDevError(e instanceof Error ? e.message : "登录失败");
    } finally {
      setDevLoading(false);
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

        <form className="login-form" onSubmit={(e) => { e.preventDefault(); handleDevLogin(); }}>
          <div className="info-box login-note">
            本机认证使用一次性启动票据：请通过 MindFlow 启动器打开界面，主令牌不会暴露给网页脚本。
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

          <button className="login-btn" type="submit" disabled={devLoading || devSuccess}>
            {devLoading ? "认证中..." : devSuccess ? "认证成功，正在进入..." : "进入 MindFlow"}
          </button>
        </form>

        {devError && (
          <div className="login-error login-error-row" role="alert">
            {devError}
            <button type="button" className="btn btn-sm btn-ghost" onClick={handleDevLogin} disabled={devLoading}>
              重试
            </button>
          </div>
        )}

        <div className="login-divider">
          <div className="login-divider-label">开发模式</div>
          <button
            type="button"
            className="login-btn login-btn--ghost"
            onClick={handleDevLogin}
            disabled={devLoading || devSuccess}
          >
            {devLoading ? "认证中..." : devSuccess ? "认证成功" : "Dev 登录（本地调试）"}
          </button>
        </div>

        <div className="login-footer">
          没有账号体系：本地票据即凭证，退出后需重新认证。
        </div>
      </div>

      <div className="introduction">MindFlow&nbsp;&nbsp;&nbsp;一款注意力分析工具软件</div>
    </div>
  );
}
