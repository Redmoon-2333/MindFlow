import { useId, useState, type ReactNode } from "react";
import HelpTip from "./HelpTip";
import "./signal.css";

interface DisclosureProps {
  /** Trigger text, e.g. "查看完整依据". */
  summary: string;
  defaultOpen?: boolean;
  /** Notified on every open/close so a layout can react to the extra height. */
  onOpenChange?: (open: boolean) => void;
  /** Jargon note rendered as a sibling of the trigger (§8.17.1: never inside
   *  the button, so opening the tip cannot toggle the disclosure). */
  help?: { what: string; effect?: string };
  children: ReactNode;
}

/**
 * Progressive disclosure (spec §3.6): evidence stays one click away instead of
 * occupying the first screen. Rendered as a button + region rather than
 * `<details>` so the trigger keeps the same measured height as the other
 * inline controls and the caret can be rotated on the exact transition §2.5
 * allows.
 */
export default function Disclosure({ summary, defaultOpen = false, onOpenChange, help, children }: DisclosureProps) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();
  const toggle = () => {
    const next = !open;
    setOpen(next);
    onOpenChange?.(next);
  };
  return (
    <div>
      <span className="sig-disclosure-row">
        <button
          type="button"
          className="sig-disclosure-trigger"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={toggle}
        >
          <span className="sig-disclosure-caret" data-open={open ? "true" : "false"} aria-hidden="true">▸</span>
          {summary}
        </button>
        {help && <HelpTip term={summary} what={help.what} effect={help.effect} />}
      </span>
      {open && (
        <div className="sig-disclosure-body" id={bodyId}>
          {children}
        </div>
      )}
    </div>
  );
}
