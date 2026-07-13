import { useState, useRef, useEffect, useCallback } from "react";

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

// ─── STYLES ──────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700;1,300&family=Spectral:ital,wght@0,300;0,400;1,300&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:          #0b0c0f;
    --bg-panel:    #0f1117;
    --bg-input:    #13151c;
    --bg-hover:    #1a1d26;
    --border:      #1e2130;
    --border-hi:   #2d3148;
    --text:        #c8ccd8;
    --text-dim:    #555a72;
    --text-bright: #e8ecf4;
    --accent:      #4f6ef7;
    --accent-dim:  #2a3680;
    --accent-glow: rgba(79,110,247,0.15);
    --green:       #3ecf8e;
    --green-dim:   #1a5a3f;
    --amber:       #f5a623;
    --red:         #f07070;
    --user-bg:     #141928;
    --radius:      6px;
    --font-mono:   'JetBrains Mono', monospace;
    --font-serif:  'Spectral', Georgia, serif;
  }

  html, body, #root { height: 100%; background: var(--bg); }

  body {
    font-family: var(--font-mono);
    color: var(--text);
    font-size: 13px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

  /* Layout */
  .app {
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* Sidebar */
  .sidebar {
    width: 220px;
    min-width: 220px;
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .sidebar-header {
    padding: 20px 16px 14px;
    border-bottom: 1px solid var(--border);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
  }

  .logo-icon {
    width: 24px; height: 24px;
    background: var(--accent);
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: #fff;
    flex-shrink: 0;
  }

  .logo-text {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-bright);
  }

  .logo-sub {
    font-size: 9px;
    color: var(--text-dim);
    letter-spacing: 0.08em;
  }

  .new-chat-btn {
    width: 100%;
    padding: 7px 10px;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
    cursor: pointer;
    display: flex; align-items: center; gap: 6px;
    transition: background 0.15s, color 0.15s;
  }
  .new-chat-btn:hover {
    background: var(--accent);
    color: #fff;
  }

  .sidebar-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }

  .conv-item {
    padding: 8px 10px;
    border-radius: var(--radius);
    cursor: pointer;
    display: flex; align-items: flex-start; gap: 6px;
    transition: background 0.1s;
    border: 1px solid transparent;
  }
  .conv-item:hover { background: var(--bg-hover); }
  .conv-item.active {
    background: var(--bg-hover);
    border-color: var(--border-hi);
  }

  .conv-id {
    font-size: 9px;
    color: var(--accent);
    font-weight: 600;
    margin-top: 1px;
    min-width: 20px;
  }

  .conv-preview {
    font-size: 11px;
    color: var(--text-dim);
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    line-height: 1.4;
  }
  .conv-item.active .conv-preview { color: var(--text); }

  .sidebar-footer {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
  }

  .status-row {
    display: flex; align-items: center; gap: 6px;
    font-size: 10px; color: var(--text-dim);
    margin-bottom: 4px;
  }
  .status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
    flex-shrink: 0;
  }
  .status-dot.off { background: var(--red); box-shadow: none; }

  /* Main chat area */
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
  }

  /* Top bar */
  .topbar {
    display: flex; align-items: center;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
    gap: 12px;
    flex-shrink: 0;
  }

  .topbar-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--text-bright);
  }
  .topbar-sub {
    font-size: 10px;
    color: var(--text-dim);
    margin-left: auto;
    font-style: italic;
  }

  .conv-badge {
    font-size: 9px;
    padding: 2px 7px;
    border-radius: 100px;
    background: var(--accent-dim);
    color: var(--accent);
    font-weight: 600;
    letter-spacing: 0.06em;
  }

  /* Messages */
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px 0;
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .msg-row {
    padding: 14px 24px;
    display: flex;
    gap: 14px;
    position: relative;
    animation: fadeSlide 0.2s ease;
  }

  @keyframes fadeSlide {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .msg-row.user {
    background: var(--user-bg);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }

  .msg-avatar {
    width: 28px; height: 28px;
    border-radius: 4px;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 700;
    margin-top: 1px;
  }

  .msg-avatar.user-av {
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent-dim);
  }
  .msg-avatar.ai-av {
    background: var(--green-dim);
    color: var(--green);
    border: 1px solid var(--green-dim);
  }

  .msg-body { flex: 1; min-width: 0; }

  .msg-meta {
    display: flex; align-items: baseline; gap: 8px;
    margin-bottom: 5px;
  }

  .msg-role {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .msg-role.user { color: var(--accent); }
  .msg-role.ai   { color: var(--green); }

  .msg-time {
    font-size: 9px;
    color: var(--text-dim);
  }

  .msg-text {
    font-family: var(--font-serif);
    font-size: 14px;
    line-height: 1.75;
    color: var(--text-bright);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg-row.user .msg-text {
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--text);
  }

  /* Cursor blink */
  .cursor {
    display: inline-block;
    width: 8px; height: 14px;
    background: var(--green);
    margin-left: 1px;
    vertical-align: middle;
    animation: blink 0.8s step-end infinite;
  }
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
  }

  /* Sources */
  .sources {
    margin-top: 10px;
    display: flex; flex-wrap: wrap; gap: 6px;
  }
  .source-tag {
    font-size: 9px;
    font-family: var(--font-mono);
    padding: 2px 8px;
    border-radius: 3px;
    background: var(--bg-input);
    border: 1px solid var(--border-hi);
    color: var(--text-dim);
    letter-spacing: 0.04em;
    max-width: 260px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Empty state */
  .empty {
    flex: 1;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 12px;
    padding: 40px;
    text-align: center;
  }
  .empty-icon {
    font-size: 32px;
    opacity: 0.15;
  }
  .empty-title {
    font-size: 13px;
    color: var(--text-dim);
    font-weight: 500;
    letter-spacing: 0.06em;
  }
  .empty-sub {
    font-size: 11px;
    color: var(--text-dim);
    opacity: 0.6;
    max-width: 280px;
    line-height: 1.6;
    font-family: var(--font-serif);
    font-style: italic;
  }

  .suggestions {
    display: flex; flex-wrap: wrap; gap: 6px;
    justify-content: center;
    margin-top: 8px;
  }
  .suggestion {
    font-size: 11px;
    padding: 5px 11px;
    border-radius: var(--radius);
    border: 1px solid var(--border-hi);
    background: var(--bg-panel);
    color: var(--text-dim);
    cursor: pointer;
    transition: all 0.15s;
    font-family: var(--font-mono);
  }
  .suggestion:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-glow);
  }

  /* Input area */
  .input-area {
    padding: 16px 20px 20px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
    flex-shrink: 0;
  }

  .input-wrap {
    display: flex; gap: 10px; align-items: flex-end;
    background: var(--bg-input);
    border: 1px solid var(--border-hi);
    border-radius: var(--radius);
    padding: 10px 12px;
    transition: border-color 0.2s;
  }
  .input-wrap:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-glow);
  }

  textarea {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    resize: none;
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--text-bright);
    line-height: 1.6;
    min-height: 22px;
    max-height: 140px;
    overflow-y: auto;
  }
  textarea::placeholder { color: var(--text-dim); }

  .send-btn {
    width: 32px; height: 32px;
    border-radius: 4px;
    border: none;
    background: var(--accent);
    color: #fff;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: background 0.15s, transform 0.1s;
  }
  .send-btn:hover:not(:disabled) { background: #6580ff; }
  .send-btn:active:not(:disabled) { transform: scale(0.93); }
  .send-btn:disabled { background: var(--accent-dim); cursor: not-allowed; opacity: 0.6; }

  .input-hint {
    margin-top: 7px;
    font-size: 10px;
    color: var(--text-dim);
    display: flex; gap: 12px;
  }
  .hint-key {
    background: var(--bg-input);
    border: 1px solid var(--border);
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 9px;
  }

  /* Error banner */
  .error-banner {
    margin: 12px 20px 0;
    padding: 8px 12px;
    background: rgba(240,112,112,0.1);
    border: 1px solid rgba(240,112,112,0.3);
    border-radius: var(--radius);
    font-size: 11px;
    color: var(--red);
    display: flex; align-items: center; gap: 8px;
  }

  /* Streaming indicator */
  .streaming-meta {
    font-size: 9px;
    color: var(--text-dim);
    margin-top: 8px;
    display: flex; align-items: center; gap: 6px;
  }
  .stream-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--green);
    animation: pulse 1s ease infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.7); }
  }
`;

// ─── HELPERS ─────────────────────────────────────────────────────────────────
function now() {
  return new Date().toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function basename(src) {
  if (!src) return "unknown";
  const parts = src.split(/[/\\]/);
  const name = parts[parts.length - 1] || src;
  return name.length > 40 ? name.slice(0, 37) + "…" : name;
}

const SUGGESTIONS = [
  "Какие товары подлежат маркировке?",
  "Порядок электронного документооборота?",
  "Требования к перевозкам пассажиров?",
  "Каков порядок прослеживаемости товаров?",
];

// ─── MAIN COMPONENT ──────────────────────────────────────────────────────────
export default function App() {
  const [conversations, setConversations] = useState([]);   // [{id, preview, messages}]
  const [activeId, setActiveId]           = useState(null);
  const [input, setInput]                 = useState("");
  const [streaming, setStreaming]         = useState(false);
  const [error, setError]                 = useState(null);
  const [apiOnline, setApiOnline]         = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef    = useRef(null);
  const abortRef       = useRef(null);

  // ── Active conversation object
  const activeConv = conversations.find(c => c.id === activeId) || null;

  // ── Health check on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) })
      .then(r => r.ok ? setApiOnline(true) : setApiOnline(false))
      .catch(() => setApiOnline(false));
  }, []);

  // ── Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConv?.messages]);

  // ── Auto-resize textarea
  const handleInputChange = (e) => {
    setInput(e.target.value);
    const ta = textareaRef.current;
    if (ta) { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + "px"; }
  };

  // ── New conversation
  const newConversation = useCallback(() => {
    const id = `local-${Date.now()}`;
    const conv = { id, serverId: null, preview: "Новый диалог", messages: [] };
    setConversations(prev => [conv, ...prev]);
    setActiveId(id);
    setError(null);
    setTimeout(() => textareaRef.current?.focus(), 50);
  }, []);

  // ── Select conversation
  const selectConv = (id) => {
    if (streaming) return;
    setActiveId(id);
    setError(null);
  };

  // ── Send message
  const sendMessage = useCallback(async (text) => {
    const q = (text || input).trim();
    if (!q || streaming) return;

    setInput("");
    setError(null);
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    // Ensure active conversation
    let convId = activeId;
    if (!convId) {
      const id = `local-${Date.now()}`;
      const conv = { id, serverId: null, preview: q.slice(0, 50), messages: [] };
      setConversations(prev => [conv, ...prev]);
      setActiveId(id);
      convId = id;
    }

    const userMsg = { role: "user", content: q, time: now() };
    const aiMsg   = { role: "assistant", content: "", time: now(), streaming: true, sources: [] };

    // Add user message + empty AI slot
    setConversations(prev => prev.map(c => c.id !== convId ? c : {
      ...c,
      preview: q.slice(0, 50),
      messages: [...c.messages, userMsg, aiMsg],
    }));

    setStreaming(true);

    try {
      // Get or create server-side conversation ID
      const conv = conversations.find(c => c.id === convId);
      let serverId = conv?.serverId || null;

      const body = { question: q, ...(serverId ? { conversation_id: serverId } : {}) };
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullText = "";
      let sources = [];
      let newServerId = serverId;

      // SSE parse loop
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const payload = JSON.parse(line.slice(6));
              if (payload.text !== undefined) {
                fullText += payload.text;
                // Update streaming message in real-time
                setConversations(prev => prev.map(c => c.id !== convId ? c : {
                  ...c,
                  messages: c.messages.map((m, i) =>
                    i === c.messages.length - 1
                      ? { ...m, content: fullText }
                      : m
                  ),
                }));
              }
            } catch {}
          } else if (line.startsWith("event: done")) {
            // next data line has metadata
          } else if (line.startsWith("data: ") && line.includes("conversation_id")) {
            try {
              const payload = JSON.parse(line.slice(6));
              if (payload.conversation_id) newServerId = payload.conversation_id;
              if (payload.sources)         sources = payload.sources;
            } catch {}
          }
        }

        // Also try to parse done event from buffer for sources
        if (buffer.includes("conversation_id")) {
          try {
            const m = buffer.match(/data: ({.+})/);
            if (m) {
              const payload = JSON.parse(m[1]);
              if (payload.conversation_id) newServerId = payload.conversation_id;
              if (payload.sources)         sources = payload.sources;
            }
          } catch {}
        }
      }

      // Finalize message
      setConversations(prev => prev.map(c => c.id !== convId ? c : {
        ...c,
        serverId: newServerId || c.serverId,
        messages: c.messages.map((m, i) =>
          i === c.messages.length - 1
            ? { ...m, content: fullText || "—", streaming: false, sources }
            : m
        ),
      }));

    } catch (e) {
      if (e.name === "AbortError") return;
      setError(`Ошибка соединения: ${e.message}. Убедись что API запущен на ${API_BASE}`);
      // Remove empty AI message on error
      setConversations(prev => prev.map(c => c.id !== convId ? c : {
        ...c,
        messages: c.messages.filter(m => !(m.role === "assistant" && m.streaming)),
      }));
    } finally {
      setStreaming(false);
      abortRef.current = null;
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  }, [input, streaming, activeId, conversations]);

  // ── Keyboard handler
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ── Abort streaming
  const abort = () => {
    abortRef.current?.abort();
    setStreaming(false);
  };

  // ─── RENDER ────────────────────────────────────────────────────────────────
  return (
    <>
      <style>{css}</style>
      <div className="app">

        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="logo">
              <div className="logo-icon">R</div>
              <div>
                <div className="logo-text">RAG</div>
                <div className="logo-sub">knowledge base</div>
              </div>
            </div>
            <button className="new-chat-btn" onClick={newConversation}>
              <span>+</span> Новый диалог
            </button>
          </div>

          <div className="sidebar-list">
            {conversations.length === 0 && (
              <div style={{ padding: "16px 8px", fontSize: 10, color: "var(--text-dim)", fontStyle: "italic" }}>
                Диалогов пока нет
              </div>
            )}
            {conversations.map(c => (
              <div
                key={c.id}
                className={`conv-item${c.id === activeId ? " active" : ""}`}
                onClick={() => selectConv(c.id)}
              >
                <span className="conv-id">#{c.serverId || "—"}</span>
                <span className="conv-preview">{c.preview}</span>
              </div>
            ))}
          </div>

          <div className="sidebar-footer">
            <div className="status-row">
              <div className={`status-dot${apiOnline === false ? " off" : ""}`} />
              <span>API {apiOnline === null ? "проверка..." : apiOnline ? "online" : "offline"}</span>
            </div>
            <div className="status-row">
              <div className="status-dot" style={{ background: "var(--amber)", boxShadow: "0 0 6px var(--amber)" }} />
              <span>qwen2.5:7b · bge-m3</span>
            </div>
          </div>
        </aside>

        {/* Main */}
        <main className="main">
          {/* Top bar */}
          <div className="topbar">
            <span className="topbar-title">
              {activeConv ? activeConv.preview.slice(0, 40) : "Выбери или начни диалог"}
            </span>
            {activeConv?.serverId && (
              <span className="conv-badge">conv #{activeConv.serverId}</span>
            )}
            {streaming && (
              <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--green)", display: "flex", alignItems: "center", gap: 5 }}>
                <span className="stream-dot" /> генерация...
                <button onClick={abort} style={{ marginLeft: 4, fontSize: 10, color: "var(--red)", background: "none", border: "1px solid var(--red)", borderRadius: 3, padding: "1px 6px", cursor: "pointer" }}>
                  стоп
                </button>
              </span>
            )}
            {!streaming && activeConv && (
              <span className="topbar-sub">{activeConv.messages.length / 2 | 0} обменов</span>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="error-banner">
              <span>⚠</span> {error}
            </div>
          )}

          {/* Messages */}
          {(!activeConv || activeConv.messages.length === 0) ? (
            <div className="empty">
              <div className="empty-icon">⬡</div>
              <div className="empty-title">Корпоративная база знаний</div>
              <div className="empty-sub">
                Задай вопрос — система найдёт ответ в проиндексированных документах
              </div>
              <div className="suggestions">
                {SUGGESTIONS.map(s => (
                  <button key={s} className="suggestion" onClick={() => sendMessage(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {activeConv.messages.map((msg, i) => (
                <div key={i} className={`msg-row ${msg.role}`}>
                  <div className={`msg-avatar ${msg.role === "user" ? "user-av" : "ai-av"}`}>
                    {msg.role === "user" ? "U" : "AI"}
                  </div>
                  <div className="msg-body">
                    <div className="msg-meta">
                      <span className={`msg-role ${msg.role === "user" ? "user" : "ai"}`}>
                        {msg.role === "user" ? "Пользователь" : "Ассистент"}
                      </span>
                      <span className="msg-time">{msg.time}</span>
                    </div>
                    <div className="msg-text">
                      {msg.content}
                      {msg.streaming && <span className="cursor" />}
                    </div>
                    {msg.streaming && msg.content.length > 0 && (
                      <div className="streaming-meta">
                        <div className="stream-dot" />
                        <span>{msg.content.length} символов...</span>
                      </div>
                    )}
                    {!msg.streaming && msg.sources && msg.sources.length > 0 && (
                      <div className="sources">
                        {msg.sources.map((s, si) => (
                          <span key={si} className="source-tag" title={s.source}>
                            📄 {basename(s.source)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}

          {/* Input */}
          <div className="input-area">
            <div className="input-wrap">
              <textarea
                ref={textareaRef}
                rows={1}
                placeholder="Введи вопрос по документам... (Enter — отправить, Shift+Enter — новая строка)"
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                disabled={streaming}
              />
              <button
                className="send-btn"
                onClick={() => sendMessage()}
                disabled={!input.trim() || streaming}
                title="Отправить (Enter)"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"/>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                </svg>
              </button>
            </div>
            <div className="input-hint">
              <span><span className="hint-key">Enter</span> отправить</span>
              <span><span className="hint-key">Shift+Enter</span> новая строка</span>
              <span style={{ marginLeft: "auto", opacity: 0.5 }}>SSE streaming · localhost:8000</span>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
