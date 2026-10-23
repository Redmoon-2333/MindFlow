import WorkspacePage from "../components/WorkspacePage";
import { useState, useEffect, useRef, useCallback } from "react";
import { sendChat, getChatSessions, getChatMessages, getErrorMessage } from "../api";
import type { ChatSession } from "../api";
import { ApiError, renameChatSession, deleteChatSession } from "../api";
import { useLocation } from "react-router-dom";
import { createPortal } from "react-dom";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { chatRuntime, useChatState, type ChatMessage } from "../lib/chat-runtime";
import "./chat.css";

export default function Chat() {
  const location = useLocation();
  const activeSessionIdRef = chatRuntime.activeSessionId;
  const conversationGenRef = chatRuntime.conversationGen;
  // Empty drafts live only in this authenticated page instance, never storage.
  const draftsRef = chatRuntime.drafts;
  const localSessionsRef = chatRuntime.localSessions;
  const [sessions, setSessions] = useChatState("sessions");
  const [activeSessionId, setActiveSessionId] = useChatState("activeSessionId");
  const [messages, setMessages] = useChatState("messages");
  const [input, setInput] = useChatState("input");
  const [loading, setLoading] = useChatState("loading");
  const [sessionsLoading, setSessionsLoading] = useChatState("sessionsLoading");
  const [messagesLoading, setMessagesLoading] = useChatState("messagesLoading");
  const [error, setError] = useChatState("error");

  const [typing, setTyping] = useChatState("typing");
  const [sessionAction, setSessionAction] = useState<{kind: "delete"; session: ChatSession} | null>(null);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const renameBusyRef = useRef(false);
  const [renameBusy, setRenameBusy] = useState(false);
  const [renameError, setRenameError] = useState("");
  const [titleDraft, setTitleDraft] = useState("");
  const [actionBusy, setActionBusy] = useState(false);
  const [actionError, setActionError] = useState("");
  const dialogRef = useRef<HTMLDivElement>(null);
  const actionTriggerRef = useRef<HTMLElement | null>(null);
  const typingFrame = chatRuntime.typingFrame;
  const revealAllRef = chatRuntime.revealAll;
  const stopTyping = () => {
    if (typingFrame.current !== null) cancelAnimationFrame(typingFrame.current);
    typingFrame.current = null;
    revealAllRef.current = null;
    setTyping(false);
  };
  useEffect(() => {
    if (!sessionAction) return;
    const controls = () => Array.from(dialogRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input') ?? []);
    controls()[0]?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !actionBusy) setSessionAction(null);
      if (event.key === "Tab") {
        const items = controls();
        const first = items[0], last = items[items.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener("keydown", keydown);
    return () => { document.removeEventListener("keydown", keydown); actionTriggerRef.current?.focus(); };
  }, [sessionAction, actionBusy]);

  useEffect(() => {
    if (location.pathname !== "/chat") { setSessionAction(null); setEditingSessionId(null); }
  }, [location.pathname]);

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

  const sessionsSeqRef = chatRuntime.sessionsSeq;
  const loadSessions = useCallback(async () => {
    const seq = ++sessionsSeqRef.current;
    setSessionsLoading(true);
    try {
      const data = await getChatSessions();
      if (seq === sessionsSeqRef.current) {
        const saved = Array.isArray(data) ? data : [];
        for (const session of saved) localSessionsRef.current.delete(session.session_id);
        setSessions([...localSessionsRef.current.values(), ...saved]);
      }
    } catch (e: unknown) {
      if (seq === sessionsSeqRef.current) setError(getErrorMessage(e, "对话加载失败，请重试"));
    } finally {
      if (seq === sessionsSeqRef.current) setSessionsLoading(false);
    }
  }, [sessionsSeqRef, localSessionsRef, setSessionsLoading, setSessions, setError]);

  const messagesSeqRef = chatRuntime.messagesSeq;
  const loadMessages = useCallback(async (sessionId: string) => {
    const seq = ++messagesSeqRef.current;
    const targetGen = conversationGenRef.current;
    setMessagesLoading(true);
    setError(null);
    try {
      const data = await getChatMessages(sessionId);
      if (seq !== messagesSeqRef.current || targetGen !== conversationGenRef.current) return;
      setMessages(Array.isArray(data) ? data : []);
    } catch (e: unknown) {
      if (seq !== messagesSeqRef.current || targetGen !== conversationGenRef.current) return;
      setError(getErrorMessage(e, "对话加载失败，请重试"));
      setMessages([]);
    } finally {
      if (seq === messagesSeqRef.current && targetGen === conversationGenRef.current) {
        setMessagesLoading(false);
      }
    }
  }, [messagesSeqRef, conversationGenRef, setMessagesLoading, setMessages, setError]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const handleSelectSession = (sessionId: string) => {
    // Re-selecting the current session must not reload incomplete server history.
    if (sessionId === activeSessionIdRef.current) return;
    conversationGenRef.current += 1;
    stopTyping();
    setLoading(false);
    setActiveSessionId(sessionId);
    const draft = draftsRef.current.get(sessionId);
    if (draft) { setMessages(draft); setMessagesLoading(false); setError(null); }
    else void loadMessages(sessionId);
  };

  const handleNewChat = (createDraft = true) => {
    conversationGenRef.current += 1;
    stopTyping();
    setLoading(false);
    const id = createDraft ? crypto.randomUUID() : null;
    if (id) {
    const session: ChatSession = {session_id: id, title: "新对话", last_message_at: new Date().toISOString()};
    draftsRef.current.set(id, []);
    localSessionsRef.current.set(id, session);
    sessionsSeqRef.current += 1;
    setSessionsLoading(false);
    setSessions(prev => [session, ...prev]);
    }
    activeSessionIdRef.current = id;
    setActiveSessionId(id);
    setMessages([]);
    setMessagesLoading(false);
    setError(null);
    setInput("");
  };

  // Tracks the session a reply belongs to. If the user switches sessions (or
  // starts a new chat) while the AI reply is in flight, the reply must NOT be
  // appended to the wrong thread (audit report — chat session cross-talk).
  //
  // A generation counter is needed in addition to the session id: starting a
  // new chat sets the session to `null`, and a reply that was also sent with
  // `null` would compare equal and land in the fresh conversation. Bumping the
  // generation on every conversation change makes "same id" insufficient on
  // its own.

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading || typing) return;

    const targetSessionId = activeSessionIdRef.current;
    const targetGen = conversationGenRef.current;
    const userMsg: ChatMessage = { role: "user", content: trimmed };
    if (targetSessionId && draftsRef.current.has(targetSessionId)) {
      draftsRef.current.set(targetSessionId, [...messages, userMsg]);
      const session = {...localSessionsRef.current.get(targetSessionId)!, title: trimmed.slice(0, 80)};
      localSessionsRef.current.set(targetSessionId, session);
      setSessions(prev => prev.map(item => item.session_id === targetSessionId ? session : item));
    }
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setError(null);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      const reply = await sendChat(trimmed, targetSessionId ?? undefined);
      // Guard: ignore the reply if the conversation changed while awaiting.
      // Both checks are required — see the generation comment above.
      if (targetGen !== conversationGenRef.current) return;
      if (targetSessionId !== activeSessionIdRef.current) return;
      if (!targetSessionId && reply?.session_id) {
        setActiveSessionId(reply.session_id);
      }
      if (targetSessionId) draftsRef.current.delete(targetSessionId);
      const aiMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: reply.answer,
        tools_used: reply?.tools_used,
        evidence_cited: reply?.evidence_cited,
        degraded: reply?.degraded,
      };
      const characters = Array.from(aiMsg.content);
      setMessages((prev) => [...prev, {...aiMsg, content: ""}]);
      setLoading(false);
      setTyping(true);
      const duration = Math.min(10000, Math.max(400, characters.length * 18));
      const start = performance.now();
      let shown = 0;
      const reveal = (count: number) => {
        setMessages(prev => prev.map(item => item.id === aiMsg.id ? {...aiMsg, content: characters.slice(0, count).join("")} : item));
      };
      revealAllRef.current = () => { reveal(characters.length); stopTyping(); };
      const tick = (now: number) => {
        if (targetGen !== conversationGenRef.current) return;
        const count = Math.min(characters.length, Math.max(1, Math.floor((now - start) / duration * characters.length)));
        if (count !== shown) { reveal(count); shown = count; }
        if (count < characters.length) typingFrame.current = requestAnimationFrame(tick);
        else stopTyping();
      };
      typingFrame.current = requestAnimationFrame(tick);
      loadSessions();
    } catch (e: unknown) {
      if (targetGen !== conversationGenRef.current) return;
      if (targetSessionId !== activeSessionIdRef.current) return;
      setError(getErrorMessage(e, "对话加载失败，请重试"));
    } finally {
      if (targetGen === conversationGenRef.current) setLoading(false);
    }
  };

  const managementError = (e: unknown) => e instanceof ApiError && (e.status === 404 || e.status === 405)
    ? "当前服务未提供会话管理接口，请重启更新后的后端，并确认前端连接的是该后端。标题尚未保存。"
    : getErrorMessage(e, "操作失败，请重试");
  const saveTitle = async (session: ChatSession) => {
    if (renameBusyRef.current) return;
    const title = titleDraft.trim();
    if (!title) { setRenameError("会话标题不能为空"); return; }
    if (title === session.title) { setEditingSessionId(null); return; }
    renameBusyRef.current = true; setRenameBusy(true); setRenameError("");
    try {
      const updated = draftsRef.current.has(session.session_id)
        ? {...session, title}
        : await renameChatSession(session.session_id, title);
      if (localSessionsRef.current.has(session.session_id)) localSessionsRef.current.set(session.session_id, {...session, title});
      sessionsSeqRef.current += 1; setSessionsLoading(false);
      setSessions(prev => prev.map(s => s.session_id === session.session_id ? {...s, title: updated.title} : s));
      setEditingSessionId(current => current === session.session_id ? null : current);
    } catch (e) { setRenameError(managementError(e)); }
    finally { renameBusyRef.current = false; setRenameBusy(false); }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <WorkspacePage title="AI 对话" description="聊聊当前任务、专注状态或遇到的困难。选择左侧历史会话可以继续之前的话题。" className="chat-page">


      {error && (
        <div className="error-box mb16">
          {error}
          <button className="btn btn-sm mt8" style={{ marginLeft: 12 }} onClick={() => setError(null)}>
            关闭
          </button>
        </div>
      )}

      <div className="chat-layout">
        {/* Session Sidebar */}
        <div
          className="card chat-sessions"
          style={{
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <button className="btn btn-sm mb16" style={{ width: "100%" }} onClick={() => handleNewChat()}>
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
                <div key={s.session_id} className={`chat-session-item ${activeSessionId === s.session_id ? "is-active" : ""}`}>
                  <div className="chat-session-row">
                    {editingSessionId === s.session_id ? <input autoFocus className="chat-session-title-input" aria-label="会话标题"
                      value={titleDraft} maxLength={80} disabled={renameBusy} onChange={e => setTitleDraft(e.target.value)}
                      onBlur={() => { if (!renameBusyRef.current) void saveTitle(s); }}
                      onKeyDown={e => {
                        if (e.nativeEvent.isComposing) return;
                        if (e.key === "Enter") { e.preventDefault(); void saveTitle(s); }
                        if (e.key === "Escape") { setEditingSessionId(null); setRenameError(""); }
                      }} /> : <button type="button" className="chat-session-select" aria-pressed={activeSessionId === s.session_id}
                        title={s.title || `会话 ${s.session_id.slice(0, 8)}`} onClick={() => handleSelectSession(s.session_id)}>
                        {s.title || `会话 ${s.session_id.slice(0, 8)}`}
                      </button>}
                    <div className="chat-session-actions">
                      {editingSessionId === s.session_id ? <>
                        <button className="chat-session-icon" title="保存标题" aria-label="保存标题" disabled={renameBusy}
                          onMouseDown={e => e.preventDefault()} onClick={() => void saveTitle(s)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6" /></svg></button>
                        <button className="chat-session-icon" title="取消编辑" aria-label="取消编辑" disabled={renameBusy}
                          onMouseDown={e => e.preventDefault()} onClick={() => {setEditingSessionId(null); setRenameError("");}}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg></button>
                      </> : <>
                        <button className="chat-session-icon" title="重命名" aria-label={`重命名 ${s.title || s.session_id.slice(0, 8)}`}
                          disabled={loading || typing || renameBusy} onClick={() => {setEditingSessionId(s.session_id); setTitleDraft(s.title || "新对话"); setRenameError("");}}>
                          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m16 3 5 5-12 12-6 1 1-6Z M14 5l5 5" /></svg>
                        </button>
                        <button className="chat-session-icon" title="删除" aria-label={`删除 ${s.title || s.session_id.slice(0, 8)}`}
                          disabled={loading || typing || renameBusy} onClick={event => {actionTriggerRef.current=event.currentTarget; setSessionAction({kind:"delete",session:s}); setActionError("");}}>
                          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7" /></svg>
                        </button>
                      </>}
                    </div>
                  </div>
                  {editingSessionId === s.session_id && renameError && <div className="chat-session-error" role="alert">{renameError}</div>}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Chat Main Area */}
        <div className="card chat-main" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Message List */}
          <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: "8px 0" }} aria-live="polite">
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
                <div className="chat-starters"><p>{activeSessionId && !draftsRef.current.has(activeSessionId) ? "这个会话还没有消息" : "今天，想从哪里开始？"}</p>{(!activeSessionId || draftsRef.current.has(activeSessionId)) && ["帮我把当前任务拆成容易开始的小步骤", "如何减少工作时的分心？", "帮我回顾今天的专注情况"].map(question => <button key={question} type="button" className="btn btn-ghost" onClick={() => { setInput(question); textareaRef.current?.focus(); }}>{question}</button>)}</div>
              </div>
            )}

            {!messagesLoading &&
              messages.map((msg, idx) => (
                <div key={msg.id ?? idx} style={{ marginBottom: 16 }}>
                  <div
                    className={`chat-bubble ${msg.role === "user" ? "chat-user" : "chat-ai"}`}
                  >
                    {msg.role === "assistant" ? <Markdown skipHtml remarkPlugins={[remarkGfm]} components={{
                      a: ({children, ...props}) => <a {...props} target="_blank" rel="noopener noreferrer">{children}</a>,
                      table: ({children, ...props}) => <div className="chat-table-scroll"><table {...props}>{children}</table></div>,
                    }}>{msg.content}</Markdown> : msg.content}
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
                  <span role="status">正在整理回复，请稍候…</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {typing && <div className="chat-typing-status"><span role="status">正在显示回复…</span><button className="btn btn-ghost btn-sm" onClick={() => revealAllRef.current?.()}>立即显示全文</button></div>}
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
              aria-label="消息输入"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                adjustTextareaHeight();
              }}
              onKeyDown={handleKeyDown}
              placeholder="输入消息，Enter 发送，Shift+Enter 换行"
              rows={1}
              disabled={loading || typing}
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
              disabled={loading || typing || !input.trim()}
              style={{ alignSelf: "flex-end", height: 42 }}
            >
              发送
            </button>
          </div>
        </div>
      </div>
      {sessionAction && createPortal(<div className="mf-modal-overlay">
        <div className="mf-dialog chat-session-dialog" role="dialog" aria-modal="true" aria-labelledby="chat-action-title" aria-describedby="chat-delete-description" ref={dialogRef}>
          <h2 id="chat-action-title">删除会话</h2>
          <p id="chat-delete-description">确定删除“{sessionAction.session.title || "这个会话"}”及全部聊天记录？删除后无法恢复。</p>
          {actionError && <p role="alert" className="error-box">{actionError}</p>}
          <div className="chat-dialog-actions"><button className="btn btn-ghost" disabled={actionBusy} onClick={() => setSessionAction(null)}>取消</button>
            <button className="btn" disabled={actionBusy} onClick={async () => {
              setActionBusy(true); setActionError("");
              try {
                const id = sessionAction.session.session_id;
                if (!draftsRef.current.has(id)) await deleteChatSession(id);
                draftsRef.current.delete(id);
                localSessionsRef.current.delete(id);
                sessionsSeqRef.current += 1; setSessionsLoading(false);
                setSessions(prev => prev.filter(s => s.session_id !== id));
                if (activeSessionIdRef.current === id) handleNewChat(false);
                setSessionAction(null);
              } catch (e) { setActionError(e instanceof ApiError && (e.status === 404 || e.status === 405) ? "当前服务未提供会话管理接口，请重启更新后的后端，并确认前端连接地址。会话尚未删除。" : getErrorMessage(e, "操作失败，请重试")); }
              finally { setActionBusy(false); }
            }}>{actionBusy ? "处理中…" : "确认删除"}</button>
          </div>
        </div>
      </div>, document.body)}
    </WorkspacePage>
  );
}
