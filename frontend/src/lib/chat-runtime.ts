import { useCallback, useSyncExternalStore } from "react";
import type { ChatSession } from "../api";

export interface ChatMessage {
  id?: string;
  role: "user" | "assistant";
  content: string;
  tools_used?: string[];
  evidence_cited?: string[] | boolean;
  degraded?: boolean;
}
interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  messages: ChatMessage[];
  input: string;
  loading: boolean;
  sessionsLoading: boolean;
  messagesLoading: boolean;
  error: string | null;
  typing: boolean;
}
const initialState = (): ChatState => ({
  sessions: [], activeSessionId: null, messages: [], input: "", loading: false,
  sessionsLoading: true, messagesLoading: false, error: null, typing: false,
});
let state = initialState();
const listeners = new Set<() => void>();
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };

// This authenticated runtime owns requests and display timers independently of
// route/component lifetimes. No conversation content is written to browser storage.
export const chatRuntime = {
  drafts: {current: new Map<string, ChatMessage[]>()},
  localSessions: {current: new Map<string, ChatSession>()},
  sessionsSeq: {current: 0},
  messagesSeq: {current: 0},
  activeSessionId: {current: null as string | null},
  conversationGen: {current: 0},
  typingFrame: {current: null as number | null},
  revealAll: {current: null as (() => void) | null},
};

export function useChatState<K extends keyof ChatState>(key: K) {
  const value = useSyncExternalStore(subscribe, () => state[key]);
  const setValue = useCallback((next: ChatState[K] | ((previous: ChatState[K]) => ChatState[K])) => {
    const value = typeof next === "function" ? next(state[key]) : next;
    state = {...state, [key]: value};
    if (key === "activeSessionId") chatRuntime.activeSessionId.current = state.activeSessionId;
    for (const listener of listeners) listener();
  }, [key]);
  return [value, setValue] as const;
}

export function resetChatRuntime() {
  // Invalidate old async completions before another authentication session starts.
  chatRuntime.conversationGen.current += 1;
  chatRuntime.sessionsSeq.current += 1;
  chatRuntime.messagesSeq.current += 1;
  if (chatRuntime.typingFrame.current !== null) cancelAnimationFrame(chatRuntime.typingFrame.current);
  chatRuntime.typingFrame.current = null;
  chatRuntime.revealAll.current = null;
  chatRuntime.activeSessionId.current = null;
  chatRuntime.drafts.current.clear();
  chatRuntime.localSessions.current.clear();
  state = initialState();
  for (const listener of listeners) listener();
}
