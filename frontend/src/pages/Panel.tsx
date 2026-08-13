import { useState } from "react";
import { triggerPanel, getPanelResult, getErrorMessage } from "../api";
import type { PanelResult } from "../api";

const ROLE_COLORS: Record<string, string> = {
  analyst: "#4F6BF6",
  分析师: "#4F6BF6",
  psychologist: "#8B5CF6",
  心理学家: "#8B5CF6",
  coach: "#22C55E",
  教练: "#22C55E",
  strategist: "#F59E0B",
  策略师: "#F59E0B",
  facilitator: "#06B6D4",
  主持人: "#06B6D4",
  evaluator: "#EC4899",
  评估师: "#EC4899",
};

function roleColor(role: string): string {
  const key = (role || "").toLowerCase();
  return ROLE_COLORS[key] || ROLE_COLORS[role] || "#64748B";
}

function confidenceColor(v: number): string {
  if (v >= 0.7) return "#22C55E";
  if (v >= 0.4) return "#F59E0B";
  return "#EF4444";
}

function confidencePct(v: number): string {
  return `${Math.round((v ?? 0) * 100)}%`;
}

export default function Panel() {
  const [loading, setLoading] = useState<"trigger" | "read" | null>(null);
  const [result, setResult] = useState<PanelResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTrigger = async (retryIfDegraded = false) => {
    setLoading("trigger");
    setError(null);
    try {
      const data = await triggerPanel(retryIfDegraded ? { retryIfDegraded: true } : undefined);
      setResult(data);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "专家面板请求失败"));
    } finally {
      setLoading(null);
    }
  };

  const handleRead = async () => {
    setLoading("read");
    setError(null);
    try {
      const data = await getPanelResult();
      setResult(data);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "读取专家面板结果失败"));
    } finally {
      setLoading(null);
    }
  };

  const types: { name: string; confidence: number }[] = result?.types.map((name) => ({
    name,
    confidence: result.confidence[name] ?? 0,
  })) ?? [];
  const discussionRounds = result?.transcript ?? [];
  const hasContent = Boolean(result && (types.length > 0 || result.technique || result.rationale || discussionRounds.length > 0));

  return (
    <div>
      <div className="header">
        <h1>专家面板</h1>
        <p>多智能体协作分析</p>
      </div>

      {error && (
        <div className="error-box mb16">
          {error}
          <button className="btn btn-sm" style={{ marginLeft: 12 }} onClick={() => setError(null)}>
            关闭
          </button>
        </div>
      )}

      <div className="flex gap8 mb24">
        <button className="btn" onClick={() => handleTrigger()} disabled={loading !== null}>
          {loading === "trigger" && (
            <span className="spinner" style={{ width: 16, height: 16, margin: 0, borderWidth: 2 }} />
          )}
          运行专家面板
        </button>
        <button className="btn btn-ghost" onClick={handleRead} disabled={loading !== null}>
          {loading === "read" && (
            <span className="spinner" style={{ width: 16, height: 16, margin: 0, borderWidth: 2 }} />
          )}
          查看上次结果
        </button>
      </div>

      {loading && !result && <div className="spinner" />}

      {result?.degraded && (
        <div className="card mb16" style={{ borderLeft: "3px solid var(--color-warning)" }}>
          <div className="flex gap8" style={{ alignItems: "center", flexWrap: "wrap" }}>
            <span className="badge badge-warning">降级模式</span>
            <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
              来源：{result.meta?.source || "未知"}；再次点击将重新尝试 DeepSeek
            </span>
            <button className="btn btn-sm" onClick={() => handleTrigger(true)} disabled={loading !== null} style={{ marginLeft: "auto" }}>
              {loading === "trigger" ? "重试中…" : "重试 DeepSeek"}
            </button>
          </div>
        </div>
      )}

      {hasContent && (
        <div className="flex gap16" style={{ flexDirection: "column" }}>
          {types.length > 0 && (
            <div className="card">
              <h3>拖延类型分析</h3>
              <div className="flex gap16" style={{ flexDirection: "column" }}>
                {types.map((t, i) => {
                  const v = typeof t.confidence === "number" ? t.confidence : 0;
                  return (
                    <div key={i}>
                      <div className="flex flex-between mb16" style={{ marginBottom: 6 }}>
                        <span style={{ fontSize: 14, fontWeight: 500 }}>{t.name || `类型 ${i + 1}`}</span>
                        <span
                          className="badge"
                          style={{
                            background: confidenceColor(v) + "20",
                            color: confidenceColor(v),
                          }}
                        >
                          {confidencePct(v)}
                        </span>
                      </div>
                      <div
                        style={{
                          height: 8,
                          borderRadius: 4,
                          background: "var(--color-bg-inset)",
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            height: "100%",
                            width: confidencePct(v),
                            background: confidenceColor(v),
                            borderRadius: 4,
                            transition: "width 0.5s ease",
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {result?.technique && (
            <div className="card">
              <h3>推荐 CBT 技术</h3>
              <p style={{ fontSize: 14, lineHeight: 1.6 }}>{result.technique}</p>
            </div>
          )}

          {result?.rationale && (
            <div className="card">
              <h3>分析依据</h3>
              <p style={{ fontSize: 14, lineHeight: 1.6, color: "var(--color-text-secondary)" }}>
                {result.rationale}
              </p>
            </div>
          )}

          {discussionRounds.length > 0 && (
            <div className="card">
              <h3>专家讨论记录</h3>
              <div className="flex gap16" style={{ flexDirection: "column" }}>
                {discussionRounds.map((round, i) => (
                  <div
                    key={i}
                    style={{
                      borderLeft: `3px solid ${roleColor(round.role)}`,
                      paddingLeft: 12,
                    }}
                  >
                    <div className="flex gap8 mb16" style={{ alignItems: "center", marginBottom: 6, flexWrap: "wrap" }}>
                      {round.round != null && (
                        <span
                          className="badge"
                          style={{
                            background: roleColor(round.role) + "20",
                            color: roleColor(round.role),
                            fontSize: 11,
                          }}
                        >
                          第 {round.round} 轮
                        </span>
                      )}
                      <span
                        className="badge"
                        style={{
                          background: roleColor(round.role) + "20",
                          color: roleColor(round.role),
                        }}
                      >
                        {round.role || "专家"}
                      </span>
                    </div>
                    <p style={{ fontSize: 13, lineHeight: 1.6, color: "var(--color-text-secondary)" }}>
                      {round.content}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {!loading && !hasContent && !error && (
        <p style={{ color: "var(--color-text-tertiary)", fontSize: 13 }}>
          点击上方按钮触发专家面板分析或查看历史记录
        </p>
      )}
    </div>
  );
}
