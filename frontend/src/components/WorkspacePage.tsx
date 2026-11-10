import type { ReactNode } from "react";
import "./workspace.css";

/**
 * Workspace shell for the form-heavy pages.
 *
 * The page title already lives in the top bar (§7.3), so this renders only the
 * one-line lede plus the page body. The old `.workspace-heading` block stays
 * absent (`e2e/density-review.spec.ts` asserts that), and so does the
 * `.page-header` H1 block the redesign deleted.
 */
export default function WorkspacePage({ title, description, children, className = "" }: {
  title: string;
  description: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`workspace-page ${className}`} aria-label={title}>
      <p className="page-lede">{description}</p>
      {children}
    </div>
  );
}
