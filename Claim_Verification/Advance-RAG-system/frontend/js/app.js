/**
 * frontend/js/app.js
 * -------------------
 * Application entry point — loaded last (after chat.js, upload.js, sources.js).
 *
 * Responsibilities:
 *   - Define the API base URL (change this if you deploy the backend elsewhere)
 *   - Wire the send button and textarea keyboard shortcut
 *   - Wire the starter chips in the empty state
 *   - Run a health check on load (+ every 30 s) to show API status
 *   - Expose shared helpers (App.setStatus, App.setSendDisabled) used by chat.js
 */

'use strict';

/* ═══════════════════════════════════════════════════════════
   CONFIGURATION — change this if the API runs on a different host/port
═══════════════════════════════════════════════════════════ */
const API_BASE = 'http://localhost:8000';


/* ═══════════════════════════════════════════════════════════
   App namespace — shared helpers called by other modules
═══════════════════════════════════════════════════════════ */
const App = {

  /**
   * Update the status indicator in the topbar.
   * @param {'ok'|'busy'|'error'} state
   * @param {string} label  Text shown next to the dot
   */
  setStatus(state, label) {
    const dot   = document.getElementById('statusDot');
    const text  = document.getElementById('statusLabel');

    dot.className  = `status-dot ${state}`;
    text.textContent = label;
  },

  /**
   * Enable / disable the send button.
   * Also visually disables it so the user knows a response is in progress.
   * @param {boolean} disabled
   */
  setSendDisabled(disabled) {
    document.getElementById('sendBtn').disabled = disabled;
  },
};


/* ═══════════════════════════════════════════════════════════
   Health check
   Hits GET /health every 30 seconds.
   Connected to: api/main.py → /health route.
═══════════════════════════════════════════════════════════ */
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
    if (res.ok) {
      App.setStatus('ok', 'Ready');
    } else {
      App.setStatus('error', 'API error');
    }
  } catch (_) {
    App.setStatus('error', 'Offline — start API');
  }
}


/* ═══════════════════════════════════════════════════════════
   Send a query
   Reads the textarea, calls Chat.send(), clears the textarea.
═══════════════════════════════════════════════════════════ */
function handleSend() {
  const input    = document.getElementById('queryInput');
  const question = input.value.trim();

  if (!question || Chat.isStreaming()) return;

  input.value = '';
  input.style.height = 'auto';

  Chat.send(question, API_BASE);
}


/* ═══════════════════════════════════════════════════════════
   DOM ready — wire everything up
═══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {

  /* 1. Init sub-modules */
  Upload.init(API_BASE);
  Sources.init();

  /* 2. Send button */
  document.getElementById('sendBtn').addEventListener('click', handleSend);

  /* 3. Textarea: auto-grow + Enter to send */
  const textarea = document.getElementById('queryInput');

  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
  });

  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  /* 4. Starter chips — click any chip to prefill and send */
  document.getElementById('starterChips').addEventListener('click', e => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    textarea.value = chip.textContent.trim();
    handleSend();
  });

  /* 5. Health check */
  checkHealth();
  setInterval(checkHealth, 30_000);

  /* 6. Focus textarea */
  textarea.focus();

});