import type { RealtimeStatus } from "../../realtime";
import "./signal.css";

const LABELS: Record<RealtimeStatus, string> = {
  connected: "实时",
  connecting: "连接中",
  reconnecting: "连接中",
  disconnected: "已断开",
  idle: "已断开",
};

/**
 * The realtime indicator (spec §3.7). Only `connected` glows — the glow is the
 * page's one "this is live" signal, so it must not be spent on "connecting".
 */
export default function LiveDot({ status }: { status: RealtimeStatus }) {
  return (
    <span className="sig-live">
      <span className="sig-live-dot" data-status={status === "connected" ? "connected" : status} aria-hidden="true" />
      {LABELS[status]}
    </span>
  );
}
