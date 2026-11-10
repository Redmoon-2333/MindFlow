import { useSyncExternalStore } from "react";
import { dismissToast, getToasts, subscribeToasts } from "./toast-store";
import "./primitives.css";

/** The bundled iconfont has no check glyph, so success draws its own mark. */
const TONE_ICONS: Record<string, string | null> = {
  info: "icon-tixing",
  success: null,
  warn: "icon-jinggao",
  error: "icon-chacha",
};

const TONE_ROLE: Record<string, "status" | "alert"> = {
  info: "status",
  success: "status",
  warn: "status",
  error: "alert",
};

/**
 * Global toast host. Mounted once next to the route outlet.
 *
 * `aria-live="polite"` is required by spec §3.11.4. Each toast also carries its
 * own `role` (`status` for confirmation, `alert` for errors), so a screen
 * reader announces the inner node's own politeness and the container's live
 * region adds no second copy of the same text — the container is empty
 * whenever no toast is showing.
 *
 * `aria-atomic="false"` keeps an appended toast from re-reading the whole
 * stack. Callers that locate "the aria-live region" by attribute alone must
 * scope their selector (see e2e/test-model-center.spec.ts), because the
 * ModelCenter polling notice is a second `aria-live="polite"` region.
 */
export default function Toaster() {
  const toasts = useSyncExternalStore(subscribeToasts, getToasts, getToasts);

  return (
    <div className="toast-region" aria-live="polite" aria-atomic="false">
      {toasts.map((entry) => (
        <div
          key={entry.id}
          className="toast"
          data-tone={entry.tone}
          role={TONE_ROLE[entry.tone]}
        >
          {TONE_ICONS[entry.tone]
            ? <span className={`iconfont ${TONE_ICONS[entry.tone]} toast-icon`} aria-hidden="true" />
            : <span className="toast-icon toast-icon--check" aria-hidden="true">✓</span>}
          <div className="toast-body">
            <div className="toast-title">
              {entry.title}
              {entry.count > 1 && <span className="toast-count">×{entry.count}</span>}
            </div>
            {entry.description && <div className="toast-desc">{entry.description}</div>}
          </div>
          <button
            type="button"
            className="toast-close"
            aria-label="关闭通知"
            onClick={() => dismissToast(entry.id)}
          >
            <span className="iconfont icon-chacha" aria-hidden="true" />
          </button>
        </div>
      ))}
    </div>
  );
}
