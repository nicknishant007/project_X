/**
 * frontend/js/verify.js
 * Attach / drop a claims JSON -> POST /verify/claims -> render verdict cards
 * + "Download predictions.json".
 */
'use strict';

const Verify = (() => {
  let busy = false;

  const VERDICTS = {
    supported:     { label: 'SUPPORTED',     cls: 'ok'   },
    overstated:    { label: 'OVERSTATED',    cls: 'warn' },
    cannot_verify: { label: 'CANNOT VERIFY', cls: 'unk'  },
  };
  const CHECKS = ['figure', 'population', 'drug_indication', 'actually_stated'];

  const esc = t => String(t ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  function card(p, claimText) {
    const v = VERDICTS[p.verdict] || VERDICTS.cannot_verify;
    const chips = CHECKS.map(n =>
      `<span class="v-chip ${esc(p.checks?.[n])}">${n.replace('_', ' ')}: ${esc(p.checks?.[n] ?? 'unclear')}</span>`
    ).join('');
    const quote = p.evidence_span
      ? `<div class="v-quote">${esc(p.evidence_span)}</div>
         <div class="v-meta">Evidence page: ${p.evidence_page ?? 'n/a'}</div>` : '';
    const comp = p.computed
      ? `<div class="v-meta">Computed in Python: claimed ${esc(p.computed.claimed)}, source ${esc(JSON.stringify(p.computed.from_source))} → ${esc(p.computed.result)}</div>` : '';
    return `
      <div class="v-card">
        <div class="v-head"><span class="v-id">${esc(p.claim_id)}</span>
          <span class="v-badge ${v.cls}">${v.label}</span></div>
        <div class="v-body">
          <div class="v-claim">${esc(claimText || '')}</div>
          <div class="v-checks">${chips}</div>
          ${quote}${comp}
          <div>${esc(p.reason)}</div>
        </div>
      </div>`;
  }

  function render(data) {
    const s = data.summary || {};
    const summary = Object.keys(VERDICTS)
      .map(k => `<span class="v-badge ${VERDICTS[k].cls}">${VERDICTS[k].label}: ${s[k] || 0}</span>`).join(' ');
    const cards = data.predictions.map(p => card(p, data.claims?.[p.claim_id])).join('');
    return `<div class="v-summary">Verified ${data.predictions.length} claims ${summary}</div>
            ${cards}
            <div class="v-actions"><button class="v-download">⬇ Download predictions.json</button></div>`;
  }

  function download(data) {
    const blob = new Blob([JSON.stringify({ predictions: data.predictions }, null, 2)],
                          { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'predictions.json';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async function run(file, apiBase) {
    if (busy) return;

    let count = '?';
    try {
      const parsed = JSON.parse(await file.text());
      const arr = Array.isArray(parsed) ? parsed : parsed.claims;
      if (!Array.isArray(arr)) throw new Error('Expected an array or {"claims": [...]}');
      count = arr.length;
    } catch (e) {
      Chat.addMessage('bot', `<span style="color:var(--red)">⚠ ${esc(file.name)} is not valid claims JSON: ${esc(e.message)}</span>`);
      return;
    }

    busy = true;
    App.setStatus('busy', 'Verifying claims…');
    App.setSendDisabled(true);
    Chat.addMessage('user', `📎 ${esc(file.name)} · ${count} claims`);
    const bubble = Chat.addMessage('bot', '', true);
    bubble.classList.add('verify-bubble');

    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('top_k', '5');
      const res = await fetch(`${apiBase}/verify/claims`, { method: 'POST', body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) {
        const d = data.detail;
        throw new Error(typeof d === 'string' ? d : JSON.stringify(d || `HTTP ${res.status}`));
      }
      bubble.innerHTML = render(data);
      bubble.querySelector('.v-download').addEventListener('click', () => download(data));
      App.setStatus('ok', 'Ready');
    } catch (err) {
      bubble.innerHTML = `<span style="color:var(--red)">⚠ ${esc(err.message)}</span>`;
      App.setStatus('error', 'Error');
    } finally {
      busy = false;
      App.setSendDisabled(false);
      const box = document.getElementById('messages');
      box.scrollTop = box.scrollHeight;
    }
  }

  return { run, isBusy: () => busy };
})();
