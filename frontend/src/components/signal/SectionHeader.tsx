import type { ReactNode } from "react";
import HelpTip from "./HelpTip";
import "./signal.css";

interface SectionHeaderProps {
  /** id of the h2, so a Surface can put `aria-labelledby` on itself. */
  id?: string;
  title: string;
  /** One line under the title: what the block plots or counts. */
  summary?: string;
  /** Right-aligned controls on the title's baseline (legend, filters). */
  actions?: ReactNode;
  /** Heading level; the dashboard uses h2 under the top bar's h1. */
  as?: "h2" | "h3";
  /** Jargon definition shown as a "?" after the title (§8.17.1). */
  help?: { what: string; effect?: string };
}

/**
 * Section title + 32×2px signal rule + optional summary and right actions
 * (spec §3.2). The rule is the only ornament: it marks where a section starts
 * without drawing a box around it.
 */
export default function SectionHeader({ id, title, summary, actions, as: Tag = "h2", help }: SectionHeaderProps) {
  return (
    <div className="sh">
      <div className="sh-text">
        <Tag className="sh-title" id={id}>
          {title}
          {help && <HelpTip term={title} what={help.what} effect={help.effect} />}
        </Tag>
        {summary && <p className="sh-summary">{summary}</p>}
      </div>
      {actions && <div className="sh-actions">{actions}</div>}
    </div>
  );
}
