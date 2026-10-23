import type { ReactNode } from "react";
import "./workspace.css";

export default function WorkspacePage({ title, children, className = "" }: {
  title: string;
  description: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`workspace-page ${className}`} aria-label={title}>
      {children}
    </div>
  );
}
