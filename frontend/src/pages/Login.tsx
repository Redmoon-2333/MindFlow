import { useState } from "react";
import { AUTH_MARKER, AUTH_REQUIRED_EVENT } from "../api";

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
            ? "认证失败 (401)：票据已失效，请重新点击 Dev 登录"
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
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh", background: "var(--color-bg)" }}>
      <div className="card" style={{ width: 440, padding: 40, textAlign: "center" }}>
        <h1 style={{ fontSize: 28, color: "var(--color-primary)", marginBottom: 8 }}>MindFlow</h1>
        <p style={{ color: "var(--color-text-secondary)", marginBottom: 24 }}>本地优先的智能专注助手</p>
        <div className="info-box" style={{ lineHeight: 1.7, marginBottom: 16 }}>
          请通过 MindFlow 启动器打开界面。启动器会生成一次性认证链接，主令牌不会暴露给网页脚本。
        </div>
        {bootstrapError && (
          <div style={{ marginTop: 12, fontSize: 12, color: "var(--color-danger)" }}>
            认证失败：{bootstrapError}
            {onClearBootstrapError && (
              <button className="btn btn-sm mt8" onClick={onClearBootstrapError} style={{ marginLeft: 8 }}>
                关闭
              </button>
            )}
          </div>
        )}
        <button className="btn btn-primary mt16" onClick={() => window.location.reload()}>
          已通过启动器打开，重新检查
        </button>

        <div style={{ marginTop: 24, paddingTop: 16, borderTop: "1px solid var(--color-border)" }}>
          <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 8 }}>开发模式</div>
          <button
            className="btn btn-ghost"
            onClick={handleDevLogin}
            disabled={devLoading || devSuccess}
            style={{ width: "100%" }}
          >
            {devLoading ? "认证中..." : devSuccess ? "认证成功" : "Dev 登录（本地调试）"}
          </button>
          {devError && (
            <div style={{ marginTop: 8, fontSize: 12, color: "var(--color-danger)" }}>
              {devError}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
