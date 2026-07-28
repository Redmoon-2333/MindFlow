import { useState } from "react";

export default function Login() {
  const [message] = useState(
    "请通过 MindFlow 启动器打开界面。启动器会生成一次性认证链接，主令牌不会暴露给网页脚本。",
  );

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh", background: "var(--color-bg)" }}>
      <div className="card" style={{ width: 440, padding: 40, textAlign: "center" }}>
        <h1 style={{ fontSize: 28, color: "var(--color-primary)", marginBottom: 8 }}>MindFlow</h1>
        <p style={{ color: "var(--color-text-secondary)", marginBottom: 24 }}>本地优先的智能专注助手</p>
        <div className="info-box" style={{ lineHeight: 1.7 }}>{message}</div>
        <button className="btn btn-primary mt16" onClick={() => window.location.reload()}>
          已通过启动器打开，重新检查
        </button>
      </div>
    </div>
  );
}
