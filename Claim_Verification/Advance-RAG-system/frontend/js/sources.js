/**
 * frontend/js/sources.js
 * -----------------------
 * Handles the Sources panel:
 *   - POST /chat/retrieve  to get the chunks your pipeline actually retrieved
 *   - Renders each chunk as an expandable card showing file name + text preview
 *   - Wires the "Sources" toggle button in the topbar
 *
 * CONNECTED TO (on the backend):
 *   POST /chat/retrieve
 *       → api/routes/chat.py
 *       → RAGService.retrieve_sources()
 *       → app/ingestion/retrieval/retrieval_pipeline.py → retrieve()
 *           which runs:
 *               HybridRetriever (BM25 + Qdrant dense)
 *               → MMRRetriever  (diversity on dense results)
 *               → merge dense + sparse
 *               → Reranker (CrossEncoder final rerank)
 *           returns List[Document] where each Document has:
 *               .page_content  → chunk text
 *               .metadata      → { file_name, source, file_size }
 *
 * Response JSON:
 *   {
 *     "sources": [
 *       {
 *         "file_name": "report.pdf",
 *         "source":    "data/incoming/report.pdf",
 *         "file_size": 204800,
 *         "preview":   "First 350 chars of the chunk…",
 *         "full_text": "Complete chunk text"
 *       },
 *       ...
 *     ]
 *   }
 */

'use strict';

const Sources = (() => {

  /* ── Panel open/close state ─────────────────────────────────────────────── */
  let panelOpen = false;

  function togglePanel(forceOpen) {
    const panel  = document.getElementById('sourcesPanel');
    const btn    = document.getElementById('sourcesBtn');

    panelOpen = (forceOpen !== undefined) ? forceOpen : !panelOpen;

    panel.classList.toggle('open', panelOpen);
    btn.classList.toggle('active', panelOpen);
  }

  /* ── File icon by extension ─────────────────────────────────────────────── */
  const EXT_ICONS = {
    pdf: '📕', docx: '📘', csv: '📊',
    txt: '📄', jpg: '🖼', jpeg: '🖼', png: '🖼',
  };

  function iconFor(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    return EXT_ICONS[ext] || '📄';
  }

  function escHtml(t) {
    return String(t)
      .replace(/&/g,'&amp;')
      .replace(/</g,'&lt;')
      .replace(/>/g,'&gt;');
  }

  /* ── Render source cards ─────────────────────────────────────────────────── */

  function render(sources) {
    const body  = document.getElementById('sourcesBody');
    const count = document.getElementById('sourcesCount');

    count.textContent = `${sources.length} chunk${sources.length !== 1 ? 's' : ''}`;
    document.getElementById('sourcesBadge').textContent = sources.length;

    if (sources.length === 0) {
      body.innerHTML = '<div class="sources-empty">No relevant chunks were found for this query.</div>';
      return;
    }

    body.innerHTML = '';

    sources.forEach((src) => {
      const card = document.createElement('div');
      card.className = 'source-card';

      card.innerHTML = `
        <div class="source-card-head" role="button" tabindex="0">
          <span class="source-card-icon">${iconFor(src.file_name)}</span>
          <span class="source-file" title="${escHtml(src.file_name)}">${escHtml(src.file_name)}</span>
          <span class="source-arrow">▶</span>
        </div>
        <div class="source-text">${escHtml(src.preview)}</div>
      `;

      // Toggle expand/collapse
      const head  = card.querySelector('.source-card-head');
      const text  = card.querySelector('.source-text');
      const arrow = card.querySelector('.source-arrow');

      function toggle() {
        text.classList.toggle('open');
        arrow.classList.toggle('open');
      }

      head.addEventListener('click', toggle);
      head.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') toggle();
      });

      body.appendChild(card);
    });
  }

  /* ── Fetch from API ─────────────────────────────────────────────────────── */

  /**
   * Called by chat.js in parallel with /chat/stream.
   * Fetches POST /chat/retrieve and renders the results.
   *
   * @param {string} question  The same question sent to /chat/stream
   * @param {string} apiBase   e.g. 'http://localhost:8000'
   */
  async function fetch_(question, apiBase) {
    try {
      const res = await fetch(`${apiBase}/chat/retrieve`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question, top_k: 5 }),
      });

      if (!res.ok) return;

      const data = await res.json();
      render(data.sources || []);

      // Auto-open the panel when results arrive
      if (!panelOpen) togglePanel(true);

    } catch (_) {
      // Sources panel is non-critical — silently fail
    }
  }

  /* ── Init ────────────────────────────────────────────────────────────────── */

  function init() {
    const panel = document.getElementById('sourcesPanel');
    const btn = document.getElementById('sourcesBtn');

    console.log('INIT PANEL:', panel);
    console.log('INIT BUTTON:', btn);

    panelOpen = true;

    panel.classList.add('open');
    btn.classList.add('active');

    btn.addEventListener('click', () => togglePanel());

    console.log('PANEL CLASSES:', panel.className);
  }

  return {
    init,
    fetch: fetch_,     // called by chat.js
    toggle: togglePanel,
  };

})();