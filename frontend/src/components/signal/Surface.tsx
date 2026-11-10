import type { ElementType, ReactNode } from "react";
import "./signal.css";

interface SurfaceProps {
  /** Semantic element; a section keeps the document outline honest. */
  as?: ElementType;
  /** `lg` = 28px 32px, `md` (default) = 24px. */
  pad?: "md" | "lg";
  className?: string;
  "aria-labelledby"?: string;
  children: ReactNode;
}

/**
 * The single container layer (spec §3.1).
 *
 * Everything that "looks like a card" is one Surface: surface fill + 1px
 * hairline + 8px radius, never a shadow, never nested inside another Surface.
 * Inner grouping uses `border-top: 1px solid var(--line)` and spacing.
 */
export default function Surface({
  as: Tag = "div",
  pad = "md",
  className,
  children,
  ...rest
}: SurfaceProps) {
  return (
    <Tag className={`sig-surface${pad === "lg" ? " sig-surface--lg" : ""}${className ? ` ${className}` : ""}`} {...rest}>
      {children}
    </Tag>
  );
}
