import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import "./signal.css";

interface HelpTipProps {
  /** The term being explained; also names the trigger for screen readers. */
  term: string;
  what: string;
  /** What happens when the reader enables, clicks or changes it (§8.17). */
  effect?: string;
}

const EDGE = 8;

/**
 * Small "?" next to a jargon term (spec §8.17.1).
 *
 * The spec's copy explains a term and what turning it on does, so it is
 * rendered on demand instead of as permanent prose: the page stays scannable
 * for a reader who already knows the word and informative for one who does not.
 *
 * Hover and focus open it provisionally; a click pins it so a touch reader can
 * read it without holding a finger down. Any scroll or resize dismisses it —
 * a fixed-position popover anchored to a moving trigger would otherwise drift
 * away from the thing it describes.
 */
export default function HelpTip({ term, what, effect }: HelpTipProps) {
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const popRef = useRef<HTMLDivElement | null>(null);
  const popId = useId();

  const close = useCallback(() => {
    setOpen(false);
    setPinned(false);
    setPosition(null);
  }, []);

  // Position is measured after paint, so the popover is never briefly painted
  // at 0,0 with the wrong width.
  useLayoutEffect(() => {
    if (!open) return;
    const trigger = triggerRef.current;
    const pop = popRef.current;
    if (!trigger || !pop) return;
    const rect = trigger.getBoundingClientRect();
    const width = pop.offsetWidth;
    const height = pop.offsetHeight;
    const left = Math.min(
      Math.max(rect.left + rect.width / 2 - width / 2, EDGE),
      window.innerWidth - width - EDGE,
    );
    const below = rect.bottom + 6;
    const top = below + height > window.innerHeight - EDGE ? rect.top - 6 - height : below;
    setPosition({ top, left });
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
      }
    };
    const onMouseDown = (event: MouseEvent) => {
      const target = event.target as Node;
      if (triggerRef.current?.contains(target) || popRef.current?.contains(target)) return;
      close();
    };
    const onViewportChange = () => close();
    document.addEventListener("keydown", onKeyDown);
    // A provisional (hover/focus) tip closes on any pointer press elsewhere;
    // a pinned one must keep its click-to-toggle behaviour, so the listener
    // only runs for the pinned case.
    if (pinned) document.addEventListener("mousedown", onMouseDown);
    window.addEventListener("scroll", onViewportChange, true);
    window.addEventListener("resize", onViewportChange);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("scroll", onViewportChange, true);
      window.removeEventListener("resize", onViewportChange);
    };
  }, [open, pinned, close]);

  const popover = open
    ? createPortal(
        <div
          role="tooltip"
          id={popId}
          className="help-tip-pop"
          ref={popRef}
          style={{ top: position?.top ?? 0, left: position?.left ?? 0, visibility: position ? "visible" : "hidden" }}
        >
          <p className="help-tip-what">{what}</p>
          {effect && <p className="help-tip-effect">{effect}</p>}
        </div>,
        document.body,
      )
    : null;

  return (
    <span className="help-tip">
      <button
        type="button"
        className="help-tip-trigger"
        ref={triggerRef}
        aria-label={`${term}是什么？`}
        aria-expanded={open}
        aria-describedby={open ? popId : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => {
          if (!pinned) setOpen(false);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          if (!pinned) setOpen(false);
        }}
        onClick={(event) => {
          // The trigger can sit inside a <label>; without this a click would
          // also toggle the associated checkbox.
          event.preventDefault();
          event.stopPropagation();
          if (pinned) {
            close();
          } else {
            setPinned(true);
            setOpen(true);
          }
        }}
      >
        ?
      </button>
      {popover}
    </span>
  );
}
