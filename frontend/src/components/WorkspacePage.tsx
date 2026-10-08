import type { ReactNode } from "react";
import "./workspace.css";

export default function WorkspacePage({ title, description, children, className = "" }: {
  title: string;
  description: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`workspace-page ${className}`}>
      <header className="workspace-heading">
        <div className="workspace-kicker">MINDFLOW · 本地工作台</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </header>
      {children}
    </div>
  );
}
