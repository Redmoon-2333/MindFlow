import type { ReactNode } from "react";
import PageHeader from "./PageHeader";
import "./workspace.css";

/**
 * Workspace shell for the form-heavy pages.
 *
 * Renders the page's real <h1> through PageHeader. The old `.workspace-heading`
 * block is deliberately NOT reintroduced: `e2e/density-review.spec.ts` asserts
 * it stays absent (the dense reference layout measures from the first card),
 * so the header here is `.page-header`, a different, single-line block.
 */
export default function WorkspacePage({ title, description, children, className = "" }: {
  title: string;
  description: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`workspace-page ${className}`} aria-label={title}>
      <PageHeader title={title} description={description} />
      {children}
    </div>
  );
}
