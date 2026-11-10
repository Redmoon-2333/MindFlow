import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import { applyTheme, readTheme } from "./themes";

applyTheme(readTheme());

/** Version badge / hashed chunk reload guard (§8.21.3).
 *
 *  After a deploy the already-open page still references the old chunk names;
 *  opening a route loads a file that no longer exists, Vite fires
 *  `vite:preloadError`, and without this the user would sit on a dead link.
 *  One reload fixes it — but only one: the guard is stored in sessionStorage
 *  with a 10 s window, so a genuinely broken chunk falls through to
 *  ErrorBoundary instead of looping. */
const PRELOAD_RELOAD_KEY = "mf-preload-reload";
const PRELOAD_RELOAD_WINDOW_MS = 10_000;

window.addEventListener("vite:preloadError", (event) => {
  const now = Date.now();
  const raw = sessionStorage.getItem(PRELOAD_RELOAD_KEY);
  const last = raw === null ? Number.NaN : Number(raw);
  if (Number.isFinite(last) && now - last <= PRELOAD_RELOAD_WINDOW_MS) return;
  sessionStorage.setItem(PRELOAD_RELOAD_KEY, String(now));
  event.preventDefault();
  window.location.reload();
});

// Bootstrap ticket exchange is handled inside <App /> so the login screen can
// show an in-progress state and keep the one-time ticket in the URL on failure
// (retryable). Rendering immediately also avoids a blank page while waiting.
createRoot(document.getElementById("root")!).render(<App />);
