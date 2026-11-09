import type { ReactNode } from "react";
import "./primitives.css";

interface PageHeaderProps {
  title: string;
  description?: string;
  /** Small uppercase eyebrow above the title; encodes section, not decoration. */
  eyebrow?: string;
  /** Right-aligned controls (filters, refresh). */
  actions?: ReactNode;
}

/**
 * The single page-title block. Renders the real <h1> for every workspace
 * page, so the document outline has exactly one top-level heading and the
 * dense reference layout keeps its rhythm underneath.
 */
export default function PageHeader({ title, description, eyebrow, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="page-header-text">
        {eyebrow && <div className="page-header-eyebrow">{eyebrow}</div>}
        <h1 className="page-header-title">{title}</h1>
        {description && <p className="page-header-desc">{description}</p>}
      </div>
      {actions && <div className="page-header-actions">{actions}</div>}
    </header>
  );
}
