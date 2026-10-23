/**
 * frontend/js/chat.js
 * --------------------
 * Handles:
 *   - Rendering user and bot messages into the DOM
 *   - Calling POST /chat/stream  →  reading the SSE token stream
 *   - Calling POST /chat/retrieve →  passing results to sources.js
 *   - The streaming cursor and typing-dots animation
 *
 * HOW THE STREAM CONNECTION WORKS:
 *   1. We POST to /chat/stream with JSON { question, top_k }
 *   2. The server keeps the response open and sends lines:
 *        data: {"type": "token", "content": "Hello"}
 *        data: {"type": "token", "content": " world"}
 *        data: {"type": "done"}
 *   3. We read the response body with a ReadableStream reader.
 *   4. Each chunk of bytes is decoded, split by newlines, and parsed.
 *   5. Tokens are appended to the bot bubble in real-time.
 *
 * This file does NOT know about:
 *   - File uploads      → upload.js
 *   - The sources panel → sources.js
 *   - App wiring        → app.js
 */

'use strict';

const Chat = (() => {

  /* ── State ───────────────────────────────────────────────────────────────── */
  let isStreaming = false;

  /* ── Private helpers ─────────────────────────────────────────────────────── */

  /**
   * Minimal Markdown → HTML renderer for bot messages.
   * Handles: code blocks, inline code, bold, italic, line breaks.
   */
  function renderMarkdown(text) {
    // Escape HTML entities first to prevent XSS
    let safe = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Fenced code blocks  ```...```
    safe = safe.replace(
      /```([\s\S]*?)```/g,
      '<pre><code>$1</code></pre>'
    );

    // Inline code  `...`
    safe = safe.replace(
      /`([^`]+)`/g,
      '<code>$1</code>'
    );

    // Bold **...**
    safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic *...*
    safe = safe.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Newlines → <br>  (skip inside <pre> blocks)
    safe = safe.replace(/\n/g, '<br>');

    return safe;
  }

  /** Escape plain text for safe innerHTML insertion (user messages) */
  function escHtml(t) {
    return t
      .replace(/&/g,'&amp;')
      .replace(/</g,'&lt;')
      .replace(/>/g,'&gt;');
  }

  /** Scroll the messages container to the bottom */
  function scrollToBottom() {
    const el = document.getElementById('messages');
    el.scrollTop = el.scrollHeight;
  }

  /** Remove the welcome empty-state if it's still there */
  function removeEmptyState() {
    const es = document.getElementById('emptyState');
    if (es) es.remove();
  }

  /**
   * Add a message row to the DOM.
   * @param {string} role    'user' | 'bot'
   * @param {string} html    HTML content to put inside the bubble
   * @param {boolean} typing Show typing dots instead of html
   * @returns {HTMLElement}  The bubble element (so caller can update it)
   */
  function appendBubble(role, html, typing = false) {
    removeEmptyState();

    const row    = document.createElement('div');
    const avatar = document.createElement('div');
    const bubble = document.createElement('div');

    row.className    = `msg-row ${role}`;
    avatar.className = `avatar ${role}`;
    bubble.className = 'bubble';

    avatar.textContent = role === 'user' ? 'U' : '🔍';

    if (typing) {
      bubble.innerHTML = `
        <div class="typing-dots">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>`;
    } else {
      bubble.innerHTML = html;
    }

    if (role === 'user') {
      row.appendChild(bubble);
      row.appendChild(avatar);
    } else {
      row.appendChild(avatar);
      row.appendChild(bubble);
    }

    document.getElementById('messages').appendChild(row);
    scrollToBottom();

    return bubble;
  }

  /* ── Public API ──────────────────────────────────────────────────────────── */

  /**
   * Main entry point — called by app.js when the user submits a question.
   *
   * @param {string} question  The user's question text
   * @param {string} apiBase   Base URL of the FastAPI server e.g. 'http://localhost:8000'
   */
  async function send(question, apiBase) {
    console.log("Chat.send() called", question);
    if (!question || isStreaming) return;

    isStreaming = true;
    App.setStatus('busy', 'Thinking…');
    App.setSendDisabled(true);


    // 1. Render user message
    appendBubble('user', escHtml(question));

    // 2. Render bot bubble with typing dots (will be replaced by stream)
    const botBubble = appendBubble('bot', '', true);

    // 3. Fetch sources in parallel (for the Sources panel)
    //    We don't await this — it runs while the stream is open
    Sources.fetch(question, apiBase);

    // 4. Open the SSE stream to /chat/stream
    let accumulated = '';
    let firstToken  = true;

    try {
      const response = await fetch(`${apiBase}/chat/stream`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question, top_k: 5 }),
      });

      if (!response.ok) {
        botBubble.innerHTML =
          `<span style="color:var(--red)">⚠ Server error ${response.status}. `
          + `Make sure the API is running: <code>uvicorn api.main:app --reload</code></span>`;
        App.setStatus('error', 'Server error');
        return;
      }

      // ReadableStream reader — reads chunks of bytes as they arrive
      const reader  = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode bytes → string, split by newline, find SSE "data:" lines
        const raw   = decoder.decode(value, { stream: true });
        const lines = raw.split('\n');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          let payload;
          try {
            payload = JSON.parse(line.slice(6));   // strip "data: "
          } catch {
            continue;
          }

          if (payload.type === 'token') {
            if (firstToken) {
              // Replace typing dots with first real content
              botBubble.innerHTML = '';
              firstToken = false;
            }
            accumulated += payload.content;
            // Show rendered markdown + blinking cursor while streaming
            botBubble.innerHTML =
              renderMarkdown(accumulated)
              + '<span class="cursor"></span>';
            scrollToBottom();

          } else if (payload.type === 'done') {
            // Stream finished — remove cursor, render final markdown
            botBubble.innerHTML = renderMarkdown(accumulated);
            scrollToBottom();
            App.setStatus('ok', 'Ready');

          } else if (payload.type === 'error') {
            botBubble.innerHTML =
              `<span style="color:var(--red)">⚠ ${escHtml(payload.content)}</span>`;
            App.setStatus('error', 'Error');
          }
        }
      }

    } catch (err) {
      botBubble.innerHTML =
        `<span style="color:var(--red)">⚠ Cannot reach API. `
        + `Run: <code>uvicorn api.main:app --reload</code></span>`;
      App.setStatus('error', 'Offline');
    } finally {
      isStreaming = false;
      App.setSendDisabled(false);
      document.getElementById('queryInput').focus();
    }
  }

  return { send, isStreaming: () => isStreaming };

})();