/**
 * frontend/js/app.js  - entry point (loaded last)
 */
'use strict';

// Same-origin: works whether you open localhost:8000 or 127.0.0.1:8000
const API_BASE = window.location.origin;

const App = {
  setStatus(state, label) {
    document.getElementById('statusDot').className = `status-dot ${state}`;
    document.getElementById('statusLabel').textContent = label;
  },
  setSendDisabled(disabled) {
    document.getElementById('sendBtn').disabled = disabled;
  },
};

async function checkHealth() {
  if (Chat.isStreaming() || Verify.isBusy()) return;
  try {
    const res = await fetch(`${API_BASE}/health`);
    App.setStatus(res.ok ? 'ok' : 'error', res.ok ? 'Ready' : 'API error');
  } catch (_) {
    App.setStatus('error', 'Offline — start API');
  }
}

/* ── attached claims file ─────────────────────────────────────────────────── */
let pendingClaims = null;

function setPending(file) {
  pendingClaims = file;
  const chip = document.getElementById('attachChip');
  chip.innerHTML = '';
  if (file) {
    const label = document.createElement('span');
    label.textContent = `📎 ${file.name}`;
    const remove = document.createElement('button');
    remove.textContent = '✕';
    remove.title = 'Remove';
    remove.addEventListener('click', () => setPending(null));
    chip.append(label, remove);
  }
  chip.classList.toggle('show', !!file);
}

function handleSend() {
  const input = document.getElementById('queryInput');
  if (Chat.isStreaming() || Verify.isBusy()) return;

  if (pendingClaims) {                       // claims JSON attached -> run verifier
    const file = pendingClaims;
    setPending(null);
    input.value = '';
    input.style.height = 'auto';
    Verify.run(file, API_BASE);
    return;
  }

  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  input.style.height = 'auto';
  Chat.send(question, API_BASE);
}

document.addEventListener('DOMContentLoaded', () => {
  // Each init is isolated so one failure cannot stop the others
  // (this is why the Sources panel sometimes needed a console command).
  for (const init of [() => Sources.init(), () => Upload.init(API_BASE)]) {
    try { init(); } catch (e) { console.error('init failed', e); }
  }

  document.getElementById('sendBtn').addEventListener('click', handleSend);

  const attachBtn = document.getElementById('attachBtn');
  const claimsInput = document.getElementById('claimsInput');
  attachBtn.addEventListener('click', () => claimsInput.click());
  claimsInput.addEventListener('change', () => {
    if (claimsInput.files[0]) setPending(claimsInput.files[0]);
    claimsInput.value = '';
  });

  const textarea = document.getElementById('queryInput');
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
  });
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  });

  document.getElementById('starterChips').addEventListener('click', e => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    textarea.value = chip.textContent.trim();
    handleSend();
  });

  checkHealth();
  setInterval(checkHealth, 30_000);
  textarea.focus();
});
