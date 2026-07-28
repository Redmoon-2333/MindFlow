export type RealtimeStatus = "idle" | "connecting" | "connected" | "reconnecting" | "disconnected";

export interface ActivityUpdatePayload {
  app_name: string;
  window_title?: string;
  process_name?: string;
  is_idle: boolean;
}

export interface InterventionEventPayload {
  id: string;
  intervention_type: string;
  title: string;
  message: string;
  dismissible: boolean;
  cbt_technique?: string | null;
}

interface RealtimeEvents {
  activity_update: ActivityUpdatePayload;
  intervention: InterventionEventPayload;
}

type Listener<T> = (payload: T, timestamp: string) => void;
type StatusListener = (status: RealtimeStatus) => void;

class RealtimeClient {
  private socket: WebSocket | null = null;
  private retryTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private retryAttempt = 0;
  private stopped = true;
  private status: RealtimeStatus = "idle";
  private readonly listeners = new Map<keyof RealtimeEvents, Set<Listener<never>>>();
  private readonly statusListeners = new Set<StatusListener>();

  connect(): void {
    this.stopped = false;
    if (this.socket?.readyState === WebSocket.OPEN || this.socket?.readyState === WebSocket.CONNECTING) return;
    this.open(this.retryAttempt > 0 ? "reconnecting" : "connecting");
  }

  disconnect(): void {
    this.stopped = true;
    this.clearTimers();
    this.socket?.close();
    this.socket = null;
    this.setStatus("disconnected");
  }

  subscribe<K extends keyof RealtimeEvents>(type: K, listener: Listener<RealtimeEvents[K]>): () => void {
    const listeners = this.listeners.get(type) ?? new Set<Listener<never>>();
    listeners.add(listener as Listener<never>);
    this.listeners.set(type, listeners);
    return () => listeners.delete(listener as Listener<never>);
  }

  subscribeStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    listener(this.status);
    return () => this.statusListeners.delete(listener);
  }

  private open(status: RealtimeStatus): void {
    this.setStatus(status);
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = new URL(`${protocol}//${window.location.host}/api/v1/ws`);
    const socket = new WebSocket(url);
    this.socket = socket;

    socket.onopen = () => {
      this.retryAttempt = 0;
      this.setStatus("connected");
      this.heartbeatTimer = window.setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "ping" }));
      }, 30_000);
    };
    socket.onmessage = (event) => this.handleMessage(event.data);
    socket.onclose = (event) => {
      this.clearTimers();
      if (this.socket === socket) this.socket = null;
      if (this.stopped || event.code === 4001) {
        this.setStatus("disconnected");
        return;
      }
      const delay = Math.min(1_000 * 2 ** this.retryAttempt, 30_000);
      this.retryAttempt += 1;
      this.setStatus("reconnecting");
      this.retryTimer = window.setTimeout(() => this.open("reconnecting"), delay);
    };
  }

  private handleMessage(raw: unknown): void {
    if (typeof raw !== "string") return;
    let message: unknown;
    try { message = JSON.parse(raw); } catch { return; }
    if (typeof message !== "object" || message === null) return;
    const frame = message as { type?: unknown; payload?: unknown; timestamp?: unknown };
    if (frame.type !== "activity_update" && frame.type !== "intervention") return;
    if (typeof frame.payload !== "object" || frame.payload === null) return;
    const timestamp = typeof frame.timestamp === "string" ? frame.timestamp : new Date().toISOString();
    const listeners = this.listeners.get(frame.type);
    listeners?.forEach((listener) => listener(frame.payload as never, timestamp));
  }

  private clearTimers(): void {
    if (this.retryTimer !== null) window.clearTimeout(this.retryTimer);
    if (this.heartbeatTimer !== null) window.clearInterval(this.heartbeatTimer);
    this.retryTimer = null;
    this.heartbeatTimer = null;
  }

  private setStatus(status: RealtimeStatus): void {
    this.status = status;
    this.statusListeners.forEach((listener) => listener(status));
  }
}

export const realtimeClient = new RealtimeClient();
