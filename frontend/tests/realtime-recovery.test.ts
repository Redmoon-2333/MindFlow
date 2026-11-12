import assert from "node:assert/strict";
import { after, afterEach, test } from "node:test";
import { realtimeClient } from "../src/realtime.ts";

const originalWindow = globalThis.window;
const originalWebSocket = globalThis.WebSocket;
const originalNow = Date.now;
let now = 0;
let timerId = 0;
const timers = new Map<number, () => void>();
const intervals = new Map<number, () => void>();
const sockets: FakeSocket[] = [];

class FakeSocket {
  static OPEN = 1;
  static CONNECTING = 0;
  readyState = 0;
  onopen: (() => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  sent: string[] = [];
  constructor(_url: URL) { sockets.push(this); }
  open() { this.readyState = 1; this.onopen?.(); }
  send(data: string) { this.sent.push(data); }
  close(code = 1000) { this.readyState = 3; this.onclose?.({ code }); }
}

globalThis.WebSocket = FakeSocket as unknown as typeof WebSocket;
globalThis.window = {
  location: { protocol: "http:", host: "127.0.0.1" },
  setTimeout(callback: () => void) { timers.set(++timerId, callback); return timerId; },
  clearTimeout(id: number) { timers.delete(id); },
  setInterval(callback: () => void) { intervals.set(++timerId, callback); return timerId; },
  clearInterval(id: number) { intervals.delete(id); },
} as unknown as Window & typeof globalThis;
Date.now = () => now;

afterEach(() => {
  realtimeClient.disconnect();
  timers.clear();
  intervals.clear();
  sockets.length = 0;
  now = 0;
});
after(() => {
  globalThis.window = originalWindow;
  globalThis.WebSocket = originalWebSocket;
  Date.now = originalNow;
});

test("an old open event cannot reconnect a stopped client", () => {
  let status = "";
  const unsubscribe = realtimeClient.subscribeStatus(value => { status = value; });
  try {
    realtimeClient.connect();
    const old = sockets[0];
    realtimeClient.disconnect();
    old.open();
    assert.equal(status, "disconnected");
    assert.equal(intervals.size, 0);
  } finally { unsubscribe(); }
});

test("manual reconnect cancels the pending retry", () => {
  realtimeClient.connect();
  sockets[0].open();
  sockets[0].close(1006);
  assert.equal(timers.size, 1);
  realtimeClient.connect();
  assert.equal(sockets.length, 2);
  assert.equal(timers.size, 0);
});

test("a half-open connection recovers when the server stops answering", () => {
  realtimeClient.connect();
  const socket = sockets[0];
  socket.open();
  now = 30_000;
  for (const tick of intervals.values()) tick();
  assert.equal(JSON.parse(socket.sent[0]).type, "ping");
  now = 120_000;
  for (const tick of [...intervals.values()]) tick();
  assert.equal(socket.readyState, 3);
  assert.equal(timers.size, 1);
  assert.equal(intervals.size, 0);
});

test("a failing status observer does not stop connection heartbeat", () => {
  const unsubscribe = realtimeClient.subscribeStatus(status => {
    if (status === "connected") throw new Error("broken observer");
  });
  try {
    realtimeClient.connect();
    assert.doesNotThrow(() => sockets[0].open());
    assert.equal(intervals.size, 1);
  } finally { unsubscribe(); }
});
