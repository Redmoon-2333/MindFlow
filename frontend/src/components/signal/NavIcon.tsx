import type { ReactElement } from "react";

/** Icon names the shell needs; one per navigation entry plus chrome. */
export type NavIconName =
  | "dashboard"
  | "focus"
  | "activities"
  | "analytics"
  | "reports"
  | "intervention"
  | "panel"
  | "chat"
  | "settings"
  | "model"
  | "execution"
  | "diagnostics"
  | "logout"
  | "sidebar"
  | "chevrons"
  | "close";

/**
 * Line icons for the sidebar and top bar (spec §7.5).
 *
 * Replaces the iconfont, whose glyphs mixed filled and outlined shapes at
 * uneven optical sizes. Every path here is drawn on the same 24-unit grid at
 * the same 1.75 stroke, so the rail reads as one set.
 */
export default function NavIcon({ name }: { name: NavIconName }) {
  return (
    <svg
      width={18}
      height={18}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {paths[name]}
    </svg>
  );
}

const paths: Record<NavIconName, ReactElement> = {
  dashboard: (
    <>
      <path d="M4 15a8 8 0 1 1 16 0" />
      <path d="M12 15l4-5" />
      <circle cx="12" cy="15" r="1.5" />
    </>
  ),
  focus: (
    <>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="4" />
      <circle cx="12" cy="12" r="1" />
    </>
  ),
  activities: (
    <>
      <path d="M9 6h11M9 12h11M9 18h11" />
      <circle cx="4.5" cy="6" r="1" />
      <circle cx="4.5" cy="12" r="1" />
      <circle cx="4.5" cy="18" r="1" />
    </>
  ),
  analytics: (
    <>
      <path d="M4 19h16" />
      <path d="M5 15l4-4 3 3 6-7" />
      <path d="M15 7h3v3" />
    </>
  ),
  reports: (
    <>
      <path d="M7 3h7l4 4v14H7z" />
      <path d="M14 3v4h4" />
      <path d="M10 12h5M10 16h5" />
    </>
  ),
  intervention: (
    <>
      <path d="M6 16v-5a6 6 0 0 1 12 0v5l1.5 2h-15z" />
      <path d="M10 20.5a2 2 0 0 0 4 0" />
    </>
  ),
  panel: (
    <>
      <circle cx="9" cy="9" r="3" />
      <path d="M3.5 19a5.5 5.5 0 0 1 11 0" />
      <circle cx="17" cy="8" r="2.5" />
      <path d="M16 13.2a4.5 4.5 0 0 1 5 4.8" />
    </>
  ),
  chat: <path d="M4 5h16v11H9l-5 4z" />,
  settings: (
    <>
      <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
      <circle cx="16" cy="7" r="2" />
      <circle cx="10" cy="17" r="2" />
    </>
  ),
  model: (
    <>
      <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" />
      <path d="M4 7.5l8 4.5 8-4.5M12 12v9" />
    </>
  ),
  execution: (
    <>
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <path d="M8.5 12l2.5 2.5 4.5-5" />
    </>
  ),
  diagnostics: <path d="M3 12h4l2-6 4 12 2-6h6" />,
  logout: (
    <>
      <path d="M14 4h5v16h-5" />
      <path d="M10 8l-4 4 4 4M6 12h10" />
    </>
  ),
  sidebar: (
    <>
      <rect x="4" y="5" width="16" height="14" rx="2" />
      <path d="M9 5v14" />
    </>
  ),
  chevrons: <path d="M12 6l-6 6 6 6M18 6l-6 6 6 6" />,
  close: <path d="M6 6l12 12M18 6L6 18" />,
};
