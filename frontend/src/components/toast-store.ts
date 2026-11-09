/**
 * Minimal toast store — no dependency, no context provider.
 *
 * `useSyncExternalStore` keeps every mounted <Toaster /> in sync, and the
 * module-level queue survives route changes: a failure reported by page A
 * stays visible after navigating to page B.
 *
 * Rules:
 *   - errors persist until dismissed (8s for informational, manual for error);
 *   - identical consecutive messages collapse into one entry with a count;
 *   - at most 4 toasts on screen, oldest evicted first.
 */

export type ToastTone = "info" | "success" | "warn" | "error";

export interface Toast {
  id: number;
  tone: ToastTone;
  title: string;
  description?: string;
  /** ms; 0 = until dismissed. */
  duration: number;
  count: number;
}

type Listener = () => void;

const MAX_VISIBLE = 4;
const listeners = new Set<Listener>();
let toasts: Toast[] = [];
let nextId = 1;
const timers = new Map<number, number>();

function emit(): void {
  listeners.forEach((listener) => listener());
}

function scheduleDismiss(id: number, duration: number): void {
  if (duration <= 0) return;
  const handle = window.setTimeout(() => dismissToast(id), duration);
  timers.set(id, handle);
}

export function subscribeToasts(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getToasts(): Toast[] {
  return toasts;
}

export function pushToast(input: {
  tone?: ToastTone;
  title: string;
  description?: string;
  duration?: number;
}): number {
  const tone = input.tone ?? "info";
  // Errors are the one thing the user must acknowledge explicitly; anything
  // else disappears on its own so it cannot pile up during a live demo.
  const duration = input.duration ?? (tone === "error" ? 0 : 5000);
  const previous = toasts[toasts.length - 1];
  if (previous && previous.title === input.title && previous.description === input.description) {
    const collapsed: Toast = { ...previous, count: previous.count + 1, duration };
    toasts = [...toasts.slice(0, -1), collapsed];
    const existing = timers.get(previous.id);
    if (existing !== undefined) window.clearTimeout(existing);
    scheduleDismiss(previous.id, duration);
    emit();
    return previous.id;
  }
  const toast: Toast = { id: nextId++, tone, title: input.title, description: input.description, duration, count: 1 };
  toasts = [...toasts, toast].slice(-MAX_VISIBLE);
  scheduleDismiss(toast.id, duration);
  emit();
  return toast.id;
}

export function dismissToast(id: number): void {
  const handle = timers.get(id);
  if (handle !== undefined) {
    window.clearTimeout(handle);
    timers.delete(id);
  }
  const next = toasts.filter((toast) => toast.id !== id);
  if (next.length === toasts.length) return;
  toasts = next;
  emit();
}

export function clearToasts(): void {
  timers.forEach((handle) => window.clearTimeout(handle));
  timers.clear();
  toasts = [];
  emit();
}

export const toast = {
  info: (title: string, description?: string) => pushToast({ tone: "info", title, description }),
  success: (title: string, description?: string) => pushToast({ tone: "success", title, description }),
  warn: (title: string, description?: string) => pushToast({ tone: "warn", title, description }),
  error: (title: string, description?: string) => pushToast({ tone: "error", title, description }),
  dismiss: dismissToast,
  clear: clearToasts,
};
