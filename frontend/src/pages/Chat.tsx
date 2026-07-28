import { useState, useEffect, useRef, useCallback } from "react";
import { sendChat, getChatSessions, getChatMessages, getErrorMessage } from "../api";
import type { ChatSession } from "../api";

interface ChatMessage {
  id?: string;
  role: "user" | "assistant";
  content: string;
  tools_used?: string[];
  evidence_cited?: string[] | boolean;
  degraded?: boolean;
}

export default function Chat() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const adjustTextareaHeight = () => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
    }
  };

  const loadSessions = useCallback(async () => {
    setSessionsLoading(true);
    try {
      const data = await getChatSessions();
      setSessions(Array.isArray(data) ? data : []);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "Request failed"));
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  const loadMessages = useCallback(async (sessionId: string) => {
    setMessagesLoading(true);
    setError(null);
    try {
      const data = await getChatMessages(sessionId);
      setMessages(Array.isArray(data) ? data : []);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "Request failed"));
      setMessages([]);
    } finally {
      setMessagesLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const handleSelectSession = (sessionId: string) => {
    setActiveSessionId(sessionId);
    loadMessages(sessionId);
  };

  const handleNewChat = () => {
    setActiveSessionId(null);
    setMessages([]);
    setInput("");
  };

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const userMsg: ChatMessage = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setError(null);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      const reply = await sendChat(trimmed, activeSessionId ?? undefined);
      if (!activeSessionId && reply?.session_id) {
        setActiveSessionId(reply.session_id);
        loadSessions();
      }
      const aiMsg: ChatMessage = {
        role: "assistant",
        content: reply.answer,
        tools_used: reply?.tools_used,
        evidence_cited: reply?.evidence_cited,
        degraded: reply?.degraded,
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "Request failed"));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div>
      <div className="header">
        <h1>AI 对话</h1>
        <p>MindFlow 智能助手</p>
      </div>

      {error && (
        <div className="error-box mb16">
          {error}
          <button className="btn btn-sm mt8" style={{ marginLeft: 12 }} onClick={() => setError(null)}>
            关闭
          </button>
        </div>
      )}

      <div style={{ display: "flex", gap: 16, height: "calc(100vh - 180px)", minHeight: 500 }}>
        {/* Session Sidebar */}
        <div
          className="card"
          style={{
            width: 200,
            flexShrink: 0,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <button className="btn btn-sm mb16" style={{ width: "100%" }} onClick={handleNewChat}>
            新对话
          </button>

          {sessionsLoading && <div className="spinner" />}

          {!sessionsLoading && sessions.length === 0 && (
            <div style={{ fontSize: 13, color: "var(--color-text-tertiary)", textAlign: "center", padding: 16 }}>
              暂无对话记录
            </div>
          )}

          {!sessionsLoading && sessions.length > 0 && (
            <div style={{ flex: 1, overflowY: "auto" }}>
              {sessions.map((s) => (
                <div
                  key={s.session_id}
                  onClick={() => handleSelectSession(s.session_id)}
                  style={{
                    padding: "10px 12px",
                    cursor: "pointer",
                    borderRadius: 8,
                    fontSize: 13,
                    marginBottom: 4,
                    background:
                      activeSessionId === s.session_id
                        ? "var(--color-primary-light)"
                        : "transparent",
                    color:
                      activeSessionId === s.session_id
                        ? "var(--color-primary)"
                        : "var(--color-text-secondary)",
                    fontWeight: activeSessionId === s.session_id ? 500 : 400,
                  }}
                  onMouseEnter={(e) => {
                    if (activeSessionId !== s.session_id) {
                      (e.target as HTMLElement).style.background = "var(--color-bg-inset)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (activeSessionId !== s.session_id) {
                      (e.target as HTMLElement).style.background = "transparent";
                    }
                  }}
                >
                  {`会话 ${s.session_id.slice(0, 8)}`}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Chat Main Area */}
        <div className="card" style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Message List */}
          <div style={{ flex: 1, overflowY: "auto", padding: "8px 0" }}>
            {messagesLoading && <div className="spinner" />}

            {!messagesLoading && messages.length === 0 && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  height: "100%",
                  color: "var(--color-text-tertiary)",
                  fontSize: 14,
                }}
              >
                {activeSessionId ? "暂无消息" : "开始新对话，向 MindFlow 智能助手提问"}
              </div>
            )}

            {!messagesLoading &&
              messages.map((msg, idx) => (
                <div key={msg.id ?? idx} style={{ marginBottom: 16 }}>
                  <div
                    className={`chat-bubble ${msg.role === "user" ? "chat-user" : "chat-ai"}`}
                    style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}
                  >
                    {msg.content}
                  </div>

                  {msg.role === "assistant" && (
                    <div className="flex gap8" style={{ marginTop: 6, flexWrap: "wrap" }}>
                      {msg.degraded && (
                        <span className="badge badge-warning">降级模式</span>
                      )}
                      {msg.tools_used && msg.tools_used.length > 0 &&
                        msg.tools_used.map((tool) => (
                          <span key={tool} className="badge badge-info">
                            工具: {tool}
                          </span>
                        ))}
                      {Array.isArray(msg.evidence_cited) && msg.evidence_cited.map((ev, i) => (
                        <span key={i} className="badge badge-primary">证据: {ev}</span>
                      ))}
                      {msg.evidence_cited === true && <span className="badge badge-primary">已引用行为证据</span>}
                    </div>
                  )}
                </div>
              ))}

            {loading && (
              <div style={{ marginBottom: 16 }}>
                <div className="chat-bubble chat-ai">
                  <div className="spinner" style={{ margin: 0 }} />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div
            className="flex gap8"
            style={{
              padding: "12px 0 0",
              borderTop: "1px solid var(--color-border)",
            }}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                adjustTextareaHeight();
              }}
              onKeyDown={handleKeyDown}
              placeholder="输入消息，Enter 发送，Shift+Enter 换行"
              rows={1}
              disabled={loading}
              style={{
                flex: 1,
                resize: "none",
                minHeight: 42,
                maxHeight: 200,
                lineHeight: 1.5,
              }}
            />
            <button
              className="btn"
              onClick={handleSend}
              disabled={loading || !input.trim()}
              style={{ alignSelf: "flex-end", height: 42 }}
            >
              发送
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
