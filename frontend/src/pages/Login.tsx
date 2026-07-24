import { useState } from "react";
import { setToken } from "../api";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async () => {
    if (!username || !password) {
      setError("请输入账号和密码");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/v1/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.detail || "登录失败");
      }
      const { token } = await res.json();
      setToken(token);
      window.location.reload();
    } catch (e: any) {
      setError(e.message || "无法连接后端，请检查服务是否启动");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleLogin();
  };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        background: "var(--color-bg)",
      }}
    >
      <div className="card" style={{ width: 400, padding: 40 }}>
        <h1
          style={{
            fontSize: 28,
            color: "var(--color-primary)",
            marginBottom: 4,
            textAlign: "center",
          }}
        >
          MindFlow
        </h1>
        <p
          style={{
            color: "var(--color-text-secondary)",
            fontSize: 14,
            textAlign: "center",
            marginBottom: 28,
          }}
        >
          本地优先的智能专注助手
        </p>

        {error && (
          <div className="error-box mb16" style={{ textAlign: "center" }}>
            {error}
          </div>
        )}

        <div className="mb16">
          <label
            style={{
              display: "block",
              fontSize: 13,
              fontWeight: 500,
              marginBottom: 6,
              color: "var(--color-text-secondary)",
            }}
          >
            账号
          </label>
          <input
            type="text"
            placeholder="请输入账号"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={handleKeyDown}
          />
        </div>

        <div className="mb16">
          <label
            style={{
              display: "block",
              fontSize: 13,
              fontWeight: 500,
              marginBottom: 6,
              color: "var(--color-text-secondary)",
            }}
          >
            密码
          </label>
          <input
            type="password"
            placeholder="请输入密码"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={handleKeyDown}
          />
        </div>

        <button
          className="btn"
          onClick={handleLogin}
          disabled={loading}
          style={{ width: "100%", justifyContent: "center", marginTop: 8 }}
        >
          {loading ? "登录中..." : "登录"}
        </button>
      </div>
    </div>
  );
}
