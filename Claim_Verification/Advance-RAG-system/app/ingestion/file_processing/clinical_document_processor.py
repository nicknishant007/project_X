from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.embedding.embedder import Embedder


class EmbeddingWrapper:
    """Adapter for the project's existing embedding model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return Embedder.embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return Embedder.embed([text])[0].tolist()


class ClinicalDocumentProcessor:
    """
    Docling extraction -> cleaning -> evidence metadata
    -> SemanticChunker for prose -> table pieces (header repeated).
    """

    BREAKPOINT_THRESHOLD_TYPE = "percentile"
    BREAKPOINT_THRESHOLD_AMOUNT = 95
    MAX_PROSE_CHARS = 1800
    MAX_TABLE_CHARS = 2500
    SKIP_LABELS = ("page_header", "page_footer", "picture")

    _converter = None
    _converter_lock = threading.Lock()   # Docling's converter is not thread-safe

    # ------------------------------------------------------------ cleaning
    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\x00", " ").replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------- docling
    @staticmethod
    def _get_label(item: Any) -> str:
        label = getattr(item, "label", None)
        return str(getattr(label, "value", label) or "text").lower()

    @staticmethod
    def _get_pages(item: Any) -> list[int]:
        """Raw Docling page numbers (Docling is 1-based; normalised in _extract_blocks)."""
        pages = set()
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                pages.add(int(page_no))
        return sorted(pages)

    @staticmethod
    def _get_item_text(item: Any) -> str:
        return str(getattr(item, "text", "") or "")

    @classmethod
    def _convert(cls, pdf_path: str):
        with cls._converter_lock:
            if cls._converter is None:
                cls._converter = DocumentConverter()
            return cls._converter.convert(pdf_path)

    @classmethod
    def _extract_blocks(cls, pdf_path: str) -> list[dict[str, Any]]:
        docling_doc = cls._convert(pdf_path).document
        blocks, current_section = [], None

        for item, _level in docling_doc.iterate_items():
            label = cls._get_label(item)
            pages = cls._get_pages(item)

            if any(skip in label for skip in cls.SKIP_LABELS):
                continue

            if "section_header" in label or label == "title":
                heading = cls._clean_text(cls._get_item_text(item))
                if heading:
                    current_section = heading
                    blocks.append({"text": heading, "element_type": "heading",
                                   "page_numbers": pages, "section": current_section})
                continue

            if "table" in label:
                try:
                    text = item.export_to_markdown(doc=docling_doc)
                except (AttributeError, TypeError):
                    text = cls._get_item_text(item)
                text = cls._clean_text(text)
                if text:
                    blocks.append({"text": text, "element_type": "table",
                                   "page_numbers": pages, "section": current_section})
                continue

            text = cls._clean_text(cls._get_item_text(item))
            if not text:
                continue

            if "caption" in label:
                element_type = "caption"
            elif "list" in label:
                element_type = "list_item"
            else:
                element_type = "paragraph"
            blocks.append({"text": text, "element_type": element_type,
                           "page_numbers": pages, "section": current_section})

        # Docling pages are 1-based. (The old code added +1, shifting every page.)
        # Safety net: if a 0-based page ever shows up, shift to 1-based.
        all_pages = [p for b in blocks for p in b["page_numbers"]]
        if all_pages and min(all_pages) == 0:
            for b in blocks:
                b["page_numbers"] = [p + 1 for p in b["page_numbers"]]
        return blocks

    # ------------------------------------------------------------ metadata
    @staticmethod
    def _build_metadata(source_metadata, page_numbers, section, element_type,
                        chunking_strategy, block_pages=None) -> dict[str, Any]:
        pages = sorted(set(page_numbers))
        md = {
            **source_metadata,
            "evidence_page": pages[0] if len(pages) == 1 else None,
            "page_numbers": pages,
            "section": section,
            "element_type": element_type,
            "chunking_strategy": chunking_strategy,
        }
        if block_pages:
            # lets the verifier pin a quote to ONE page even when a chunk spans pages
            md["block_pages"] = block_pages
        return md

    # ------------------------------------------------------------- chunking
    @classmethod
    def _semantic_chunk_prose(cls, blocks, source_metadata) -> list[Document]:
        if not blocks:
            return []
        source_text = "\n\n".join(b["text"] for b in blocks).strip()
        if not source_text:
            return []

        splitter = SemanticChunker(
            embeddings=EmbeddingWrapper(),
            breakpoint_threshold_type=cls.BREAKPOINT_THRESHOLD_TYPE,
            breakpoint_threshold_amount=cls.BREAKPOINT_THRESHOLD_AMOUNT,
        )
        size_guard = RecursiveCharacterTextSplitter(
            chunk_size=cls.MAX_PROSE_CHARS - 300, chunk_overlap=150)

        output = []
        for chunk in splitter.create_documents([source_text]):
            chunk_text = cls._clean_text(chunk.page_content)
            if not chunk_text:
                continue

            matched = [b for b in blocks
                       if b["text"] in chunk_text or chunk_text in b["text"]]
            used = matched or blocks   # conservative provenance if matching fails

            pages = sorted({p for b in used for p in b["page_numbers"]})
            sections = list(dict.fromkeys(b["section"] for b in used if b["section"]))
            types = list(dict.fromkeys(b["element_type"] for b in used))
            block_pages = ([{"text": b["text"], "pages": b["page_numbers"]} for b in matched]
                           if matched else None)

            pieces = (size_guard.split_text(chunk_text)
                      if len(chunk_text) > cls.MAX_PROSE_CHARS else [chunk_text])
            for piece in pieces:
                output.append(Document(
                    page_content=piece,
                    metadata=cls._build_metadata(
                        source_metadata, pages,
                        " > ".join(sections) if sections else None,
                        types[0] if len(types) == 1 else "mixed",
                        "langchain_semantic", block_pages),
                ))
        return output

    @classmethod
    def _split_table(cls, text: str) -> list[str]:
        """Split a big markdown table by rows, repeating the header in every piece."""
        if len(text) <= cls.MAX_TABLE_CHARS:
            return [text]
        lines = text.split("\n")
        header = lines[:2] if len(lines) > 2 and set(lines[1].strip()) <= set("|-: ") else lines[:1]
        pieces, current = [], list(header)
        for row in lines[len(header):]:
            if len("\n".join(current + [row])) > cls.MAX_TABLE_CHARS and len(current) > len(header):
                pieces.append("\n".join(current))
                current = list(header)
            current.append(row)
        if len(current) > len(header):
            pieces.append("\n".join(current))
        return pieces or [text]

    # ---------------------------------------------------------------- main
    @classmethod
    def process(cls, pdf_path: str) -> list[Document]:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("ClinicalDocumentProcessor supports PDFs only.")

        blocks = cls._extract_blocks(str(path))
        source_metadata = {
            "source": str(path.resolve()),
            "file_name": path.name,
            "cited_paper": path.name,
            "document_id": path.stem,
            "file_type": "pdf",
            "source_role": "clinical_evidence",
        }

        documents, prose_run = [], []

        def flush_prose():
            nonlocal prose_run
            if prose_run:
                documents.extend(cls._semantic_chunk_prose(prose_run, source_metadata))
                prose_run = []

        for block in blocks:
            if block["element_type"] == "table":
                flush_prose()
                for piece in cls._split_table(block["text"]):
                    documents.append(Document(
                        page_content=piece,
                        metadata=cls._build_metadata(
                            source_metadata, block["page_numbers"], block["section"],
                            "table", "atomic"),
                    ))
            else:
                prose_run.append(block)
        flush_prose()
        return documents
