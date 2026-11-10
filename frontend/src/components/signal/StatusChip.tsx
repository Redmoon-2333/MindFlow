import { useCallback, useEffect, useRef, useState } from "react";
import {
  getAiProviderStatus,
  getCollectorStatus,
  getHealth,
  getModelStatus,
  getErrorMessage,
  startCollector,
  stopCollector,
} from "../../api";
import type { AiProviderStatus, CollectorStatus, HealthData, ModelStatus } from "../../api";
import { toast } from "../toast-store";
import HelpTip from "./HelpTip";
import "./signal.css";

type Severity = "unknown" | "alarm" | "drift" | "ok";

interface ChipState {
  severity: Severity;
  text: string;
}

/**
 * System health, collapsed into one capsule (spec §3.5).
 *
 * "Normal is invisible": four healthy subsystems used to occupy the first row
 * of the dashboard, which is the most expensive space on the page and said
 * nothing the user needed. Now a single dot carries them; the details open on
 * demand, and only an abnormal state changes the capsule itself.
 */
export default function StatusChip() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [healthFailed, setHealthFailed] = useState(false);
  const [collector, setCollector] = useState<CollectorStatus | null>(null);
  const [provider, setProvider] = useState<AiProviderStatus | null>(null);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [settled, setSettled] = useState(false);
  const [open, setOpen] = useState(false);
  const [toggling, setToggling] = useState(false);

  const rootRef = useRef<HTMLDivElement | null>(null);
  const chipRef = useRef<HTMLButtonElement | null>(null);
  /** Fixed id: the capsule is mounted once, and the spec names this region. */
  const popoverId = "status-popover";

  const load = useCallback(async () => {
    const [h, cs, ps, ms] = await Promise.allSettled([
      getHealth(), getCollectorStatus(), getAiProviderStatus(), getModelStatus(),
    ]);
    setHealthFailed(h.status === "rejected");
    setHealth(h.status === "fulfilled" ? h.value : null);
    // A rejected collector request must not be shown as "stopped": that would
    // claim a fact we do not have.
    if (cs.status === "fulfilled") setCollector(cs.value);
    if (ps.status === "fulfilled") setProvider(ps.value);
    if (ms.status === "fulfilled") setModel(ms.value);
    setSettled(true);
  }, []);

  useEffect(() => { load(); }, [load]);

  // 60 s poll: the capsule is the only place the user learns the collector
  // died, so it must not go stale while the page sits open.
  useEffect(() => {
    const id = window.setInterval(load, 60_000);
    return () => window.clearInterval(id);
  }, [load]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setOpen(false);
        chipRef.current?.focus();
      }
    };
    const onPointerDown = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("mousedown", onPointerDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("mousedown", onPointerDown);
    };
  }, [open]);

  const collectorRunning = collector?.running === true;

  const chip: ChipState = !settled
    ? { severity: "unknown", text: "检查中…" }
    : healthFailed
      ? { severity: "alarm", text: "服务无响应" }
      : health && health.database?.status !== "ok"
        ? { severity: "alarm", text: "数据库异常" }
        : health && health.status !== "ok" && health.status !== "healthy"
          ? { severity: "alarm", text: "服务异常" }
          : collector && !collectorRunning
            ? { severity: "drift", text: "采集已停止" }
            : { severity: "ok", text: "运行正常" };

  const dotColor = {
    unknown: "var(--text-dim)",
    alarm: "var(--alarm)",
    drift: "var(--drift)",
    ok: "var(--ok)",
  }[chip.severity];

  const handleCollectorToggle = async () => {
    setToggling(true);
    try {
      setCollector(collectorRunning ? await stopCollector() : await startCollector());
    } catch (error) {
      toast.error(getErrorMessage(error, "采集器开关失败"));
    } finally {
      setToggling(false);
    }
  };

  const serviceText = healthFailed ? "无响应" : health ? (health.status === "ok" || health.status === "healthy" ? "正常" : health.status) : "—";
  const databaseText = health ? (health.database?.status === "ok" ? "已连接" : "异常") : "—";
  const providerText = provider
    ? provider.configured ? `已配置 · ${provider.model}` : "未配置（使用本地规则）"
    : "—";
  const modelText = model?.demo_only
    ? "合成演示模型 · 影子模式"
    : model?.loaded ? "个人模型已加载" : "规则引擎模式";

  return (
    <div className="sig-status" ref={rootRef}>
      <button
        type="button"
        className="sig-chip"
        ref={chipRef}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? popoverId : undefined}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="sig-chip-dot" style={{ background: dotColor }} aria-hidden="true" />
        {chip.text}
        <span className="sig-chip-caret" aria-hidden="true">▾</span>
      </button>
      {open && (
        <div className="sig-popover" id={popoverId} role="dialog" aria-label="系统状态">
          <dl className="sig-popover-list">
            <dt>服务</dt>
            <dd><span className="sig-popover-value">{serviceText}</span></dd>
            <dt>数据库</dt>
            <dd><span className="sig-popover-value">{databaseText}</span></dd>
            <dt>采集器</dt>
            <dd>
              <span className="sig-popover-value">{collector ? (collectorRunning ? "运行中" : "已停止") : "—"}</span>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handleCollectorToggle}
                disabled={toggling || !collector}
              >
                {toggling ? "切换中…" : collectorRunning ? "停止采集" : "启动采集"}
              </button>
              <HelpTip
                term="采集"
                what="MindFlow 每隔几秒记下你正在用哪个软件、窗口叫什么。数据只存在这台电脑上。"
                effect="停止后不再记录，报告、预测和提醒都会缺这段时间的数据。重启 MindFlow 后会自动重新开始。"
              />
            </dd>
            <dt>
              云端 AI
              <HelpTip
                term="云端 AI"
                what="联网的大模型（例如 DeepSeek），用来写分析和回答问题。"
                effect="已配置时，你的行为汇总（不含窗口标题）会发给它，可能产生少量费用。未配置时只用本机规则，不联网。"
              />
            </dt>
            <dd><span className="sig-popover-value">{providerText}</span></dd>
            <dt>
              预测模型
              <HelpTip
                term="预测模型"
                what="根据你的使用习惯，猜你此刻是专注还是分心。"
                effect="演示模型和影子模式只用来展示，判断不算数；个人模型就绪后才会用来提醒你。"
              />
            </dt>
            <dd><span className="sig-popover-value">{modelText}</span></dd>
            <dt>版本</dt>
            <dd><span className="sig-popover-value">{health?.version || "—"}</span></dd>
          </dl>
        </div>
      )}
    </div>
  );
}
