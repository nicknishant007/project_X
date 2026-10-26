import os
import pickle
import re
import threading
from pathlib import Path
from typing import List, Optional, Set

from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parents[3]
_TOKEN_RE = re.compile(r"\w+(?:[.\-]\w+)*")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens; keeps decimals (1.63) and hyphenated terms intact."""
    return _TOKEN_RE.findall((text or "").lower())


class BM25Indexer:

    def __init__(self):
        self._lock = threading.RLock()
        self.documents: List = []
        self.tokenized_docs: List = []
        self.bm25: Optional[BM25Okapi] = None
        self._mtime = None

        self.storage_path = BASE_DIR / "storage" / "bm25" / "bm25_store.pkl"
        os.makedirs(self.storage_path.parent, exist_ok=True)
        self.load()

    def _rebuild(self):
        self.bm25 = BM25Okapi(self.tokenized_docs) if self.tokenized_docs else None

    def add_documents(self, chunks):
        with self._lock:
            self._reload_if_changed()
            added = False
            for chunk in chunks:
                tokens = tokenize(chunk.page_content)
                if not tokens:
                    continue
                self.documents.append(chunk)
                self.tokenized_docs.append(tokens)
                added = True
            if added:
                self._rebuild()
                self.save()

    def search(self, query, top_k=7, document_ids: Optional[Set[str]] = None):
        with self._lock:
            self._reload_if_changed()   # picks up documents ingested by another instance
            if self.bm25 is None:
                return []
            scores = self.bm25.get_scores(tokenize(query))
            ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in ranked:
            if score == 0:
                continue
            if document_ids and (chunk.metadata or {}).get("document_id") not in document_ids:
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    def document_ids(self) -> Set[str]:
        with self._lock:
            self._reload_if_changed()
            return {
                d.metadata.get("document_id")
                for d in self.documents
                if d.metadata and d.metadata.get("document_id")
            }

    def save(self):
        with self._lock:
            tmp = self.storage_path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"documents": self.documents}, f)
            os.replace(tmp, self.storage_path)
            self._mtime = self.storage_path.stat().st_mtime_ns

    def _reload_if_changed(self):
        try:
            mtime = self.storage_path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if mtime != self._mtime:
            self.load()

    def load(self):
        with self._lock:
            if not self.storage_path.exists() or self.storage_path.stat().st_size == 0:
                return
            try:
                with open(self.storage_path, "rb") as f:
                    data = pickle.load(f)
                # Tokens are recomputed so tokenizer changes never desync index and queries.
                self.documents = data.get("documents", [])
                self.tokenized_docs = [tokenize(d.page_content) for d in self.documents]
                self._mtime = self.storage_path.stat().st_mtime_ns
                self._rebuild()
            except Exception as e:
                print(f"BM25 load failed: {e}")
