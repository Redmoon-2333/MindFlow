import type { ReactNode } from "react";
import "./primitives.css";

interface EmptyStateProps {
  title: string;
  /** What the reader can do next — an empty screen is an invitation to act. */
  description?: string;
  action?: ReactNode;
  /** "error" reserves the alert colour for a failed load. */
  variant?: "empty" | "error";
}

/** The one empty/error surface (spec §8.0): a quiet centred line, no icon and
 *  no dashed placeholder box. Replaces .workspace-empty/.f-empty/.d-empty. */
export default function EmptyState({ title, description, action, variant = "empty" }: EmptyStateProps) {
  return (
    <div className="empty-state" data-variant={variant} role={variant === "error" ? "alert" : undefined}>
      <div className="empty-state-title">{title}</div>
      {description && <p className="empty-state-desc">{description}</p>}
      {action && <div className="empty-state-action">{action}</div>}
    </div>
  );
}
