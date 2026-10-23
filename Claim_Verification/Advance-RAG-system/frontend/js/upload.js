/**
 * frontend/js/upload.js
 * ----------------------
 * Handles the "Add Document" feature:
 *   - Drag-and-drop onto the drop zone
 *   - Click-to-open file picker
 *   - POST /ingest/upload  for each file
 *   - Show upload progress (loading → success/error/skipped)
 *   - Add the file to the Knowledge Base list in the sidebar
 *
 * CONNECTED TO (on the backend):
 *   POST /ingest/upload
 *       → api/routes/ingest.py  →  IngestService.save_uploaded_file()
 *       → IngestService.process_single_file()
 *       → app/ingestion/queue/worker.py → Worker.process()
 *           which runs: extract → clean → chunk → embed → store in Qdrant + BM25
 *
 * Response JSON:
 *   {
 *     "status":         "success" | "skipped" | "error",
 *     "file":           "document.pdf",
 *     "chunks_created": 42,
 *     "skipped":        false,
 *     "message":        "Ingested 42 chunks from 'document.pdf'."
 *   }
 */

'use strict';

const Upload = (() => {

  /* ── Icon + CSS class mapping by extension ────────────────────────────────── */
  const EXT_META = {
    pdf:  { icon: '📕', cls: 'pdf'  },
    docx: { icon: '📘', cls: 'docx' },
    csv:  { icon: '📊', cls: 'csv'  },
    txt:  { icon: '📄', cls: 'txt'  },
    jpg:  { icon: '🖼',  cls: 'img'  },
    jpeg: { icon: '🖼',  cls: 'img'  },
    png:  { icon: '🖼',  cls: 'img'  },
  };

  function getExt(filename) {
    return filename.split('.').pop().toLowerCase();
  }

  function getMeta(filename) {
    return EXT_META[getExt(filename)] || { icon: '📄', cls: 'txt' };
  }

  /** Format bytes to human-readable string */
  function formatSize(bytes) {
    if (bytes < 1024)       return `${bytes} B`;
    if (bytes < 1024*1024)  return `${(bytes/1024).toFixed(1)} KB`;
    return `${(bytes/1024/1024).toFixed(1)} MB`;
  }

  /* ── Status display ─────────────────────────────────────────────────────── */

  const statusEl = document.getElementById('uploadStatus');

  function showStatus(type, message) {
    statusEl.className = `upload-status ${type}`;
    statusEl.textContent = message;
    if (type !== 'loading') {
      // Auto-hide after 5 seconds
      setTimeout(() => { statusEl.style.display = 'none'; }, 5000);
    }
  }

  /* ── Sidebar docs list ───────────────────────────────────────────────────── */

  function addToDocsList(filename, chunks, skipped) {
    const listEl = document.getElementById('docsList');

    // Remove the "No documents yet" placeholder if present
    const empty = listEl.querySelector('.docs-empty');
    if (empty) empty.remove();

    const { icon, cls } = getMeta(filename);
    const item = document.createElement('div');
    item.className = 'doc-item';
    item.innerHTML = `
      <div class="doc-icon ${cls}">${icon}</div>
      <div class="doc-info">
        <div class="doc-name" title="${filename}">${filename}</div>
        <div class="doc-chunks">
          ${skipped
            ? '⚠ Already indexed'
            : chunks > 0
              ? `${chunks} chunks indexed`
              : 'Processed'
          }
        </div>
      </div>
    `;
    listEl.appendChild(item);
  }

  /* ── Upload a single file ─────────────────────────────────────────────────── */

  async function uploadFile(file, apiBase) {
    showStatus('loading', `Uploading "${file.name}"…`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res  = await fetch(`${apiBase}/ingest/upload`, {
        method: 'POST',
        body:   formData,
        // NOTE: do NOT set Content-Type here — browser sets it with the boundary
      });

      const data = await res.json();

      if (res.ok) {
        if (data.skipped) {
          showStatus('skipped', `⚠ ${data.message}`);
        } else {
          showStatus('success', `✓ ${data.message}`);
        }
        addToDocsList(data.file, data.chunks_created, data.skipped);
      } else {
        showStatus('error', `✗ ${data.detail || 'Upload failed.'}`);
      }

    } catch (err) {
      showStatus('error', '✗ Cannot reach API — is it running?');
    }
  }

  /* ── Init: wire up drop zone + file input ────────────────────────────────── */

  function init(apiBase) {
    const dropZone  = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    // File picker trigger
    fileInput.addEventListener('change', () => {
      Array.from(fileInput.files).forEach(f => uploadFile(f, apiBase));
      fileInput.value = '';   // reset so same file can be re-uploaded if needed
    });

    // Drag-over visual feedback
    dropZone.addEventListener('dragover', e => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('drag-over');
    });

    // Drop
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      Array.from(e.dataTransfer.files).forEach(f => uploadFile(f, apiBase));
    });
  }

  return { init };

})();