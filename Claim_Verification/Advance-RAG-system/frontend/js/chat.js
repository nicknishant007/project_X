/**
 * frontend/js/chat.js
 * Renders messages and streams answers from POST /chat/stream (SSE).
 * Fixes: SSE events split across network chunks are now buffered (old code dropped them).
 */
'use strict';

const Chat = (() => {
  let isStreaming = false;

  function escHtml(t) {
    return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function renderMarkdown(text) {
    let safe = escHtml(text);
    safe = safe.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
    safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    safe = safe.replace(/\*(.+?)\*/g, '<em>$1</em>');
    return safe.replace(/\n/g, '<br>');
  }

  function scrollToBottom() {
    const el = document.getElementById('messages');
    el.scrollTop = el.scrollHeight;
  }

  function removeEmptyState() {
    const es = document.getElementById('emptyState');
    if (es) es.remove();
  }

  /** Add a message row; returns the bubble element. */
  function appendBubble(role, html, typing = false) {
    removeEmptyState();
    const row = document.createElement('div');
    const avatar = document.createElement('div');
    const bubble = document.createElement('div');

    row.className = `msg-row ${role}`;
    avatar.className = `avatar ${role}`;
    bubble.className = 'bubble';
    avatar.textContent = role === 'user' ? 'U' : '🔍';

    bubble.innerHTML = typing
      ? `<div class="typing-dots"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`
      : html;

    if (role === 'user') { row.appendChild(bubble); row.appendChild(avatar); }
    else { row.appendChild(avatar); row.appendChild(bubble); }

    document.getElementById('messages').appendChild(row);
    scrollToBottom();
    return bubble;
  }

  async function send(question, apiBase) {
    if (!question || isStreaming) return;

    isStreaming = true;
    App.setStatus('busy', 'Thinking…');
    App.setSendDisabled(true);

    appendBubble('user', escHtml(question));
    const botBubble = appendBubble('bot', '', true);
    Sources.fetch(question, apiBase);

    let accumulated = '';
    let firstToken = true;

    function handleEvent(payload) {
      if (payload.type === 'token') {
        if (firstToken) { botBubble.innerHTML = ''; firstToken = false; }
        accumulated += payload.content;
        botBubble.innerHTML = renderMarkdown(accumulated) + '<span class="cursor"></span>';
        scrollToBottom();
      } else if (payload.type === 'done') {
        if (accumulated) botBubble.innerHTML = renderMarkdown(accumulated);
        App.setStatus('ok', 'Ready');
      } else if (payload.type === 'error') {
        botBubble.innerHTML = `<span style="color:var(--red)">⚠ ${escHtml(payload.content)}</span>`;
        App.setStatus('error', 'Error');
      }
    }

    try {
      const response = await fetch(`${apiBase}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 5 }),
      });

      if (!response.ok) {
        botBubble.innerHTML = `<span style="color:var(--red)">⚠ Server error ${response.status}.</span>`;
        App.setStatus('error', 'Server error');
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split('\n\n');
        buffer = events.pop();                 // keep the incomplete tail for next read
        for (const ev of events) {
          for (const line of ev.split('\n')) {
            if (!line.startsWith('data: ')) continue;
            try { handleEvent(JSON.parse(line.slice(6))); } catch { /* ignore bad line */ }
          }
        }
      }
    } catch (err) {
      botBubble.innerHTML = `<span style="color:var(--red)">⚠ Cannot reach API. Run: <code>uvicorn api.main:app --reload</code></span>`;
      App.setStatus('error', 'Offline');
    } finally {
      isStreaming = false;
      App.setSendDisabled(false);
      document.getElementById('queryInput').focus();
    }
  }

  return { send, addMessage: appendBubble, isStreaming: () => isStreaming };
})();
