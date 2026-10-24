
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from app.ingestion.embedding.embedder import Embedder


class EmbeddingWrapper:
    """Adapter for the project's existing embedding model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return Embedder.embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return Embedder.embed([text])[0].tolist()


class ClinicalDocumentProcessor:
    """
    Clinical PDF processing for Fractiva claim verification.

    Pipeline:
        Docling extraction
        -> structure-aware cleaning
        -> evidence metadata
        -> SemanticChunker for prose
        -> atomic table Documents

    Claim verdicts and numerical calculations belong in the verifier.
    """

    BREAKPOINT_THRESHOLD_TYPE = "percentile"
    BREAKPOINT_THRESHOLD_AMOUNT = 95

    # ---------------------------------------------------------
    # 1. Text cleaning
    # ---------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""

        text = text.replace("\x00", " ").replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # ---------------------------------------------------------
    # 2. Docling helpers
    # ---------------------------------------------------------

    @staticmethod
    def _get_label(item: Any) -> str:
        label = getattr(item, "label", None)
        return str(getattr(label, "value", label) or "text").lower()

    @staticmethod
    def _get_pages(item: Any) -> list[int]:
        """Return one-based PDF page numbers from Docling provenance."""
        pages = set()

        for provenance in getattr(item, "prov", []) or []:
            page_no = getattr(provenance, "page_no", None)
            if page_no is not None:
                pages.add(int(page_no) + 1)

        return sorted(pages)

    @staticmethod
    def _get_item_text(item: Any) -> str:
        return str(getattr(item, "text", "") or "")

    # ---------------------------------------------------------
    # 3. Structured extraction
    # ---------------------------------------------------------

    @classmethod
    def _extract_blocks(cls, pdf_path: str) -> list[dict[str, Any]]:
        result = DocumentConverter().convert(pdf_path)
        docling_doc = result.document

        blocks = []
        current_section = None

        for item, _level in docling_doc.iterate_items():
            label = cls._get_label(item)
            pages = cls._get_pages(item)

            # Headings establish section context.
            if "section_header" in label or label == "title":
                heading = cls._clean_text(cls._get_item_text(item))

                if heading:
                    current_section = heading
                    blocks.append({
                        "text": heading,
                        "element_type": "heading",
                        "page_numbers": pages,
                        "section": current_section,
                    })
                continue

            # Preserve table structure.
            if "table" in label:
                try:
                    text = item.export_to_markdown(doc=docling_doc)
                except (AttributeError, TypeError):
                    text = cls._get_item_text(item)

                text = cls._clean_text(text)

                if text:
                    blocks.append({
                        "text": text,
                        "element_type": "table",
                        "page_numbers": pages,
                        "section": current_section,
                    })
                continue

            # Captions may contain essential numerical context.
            if "caption" in label:
                text = cls._clean_text(cls._get_item_text(item))

                if text:
                    blocks.append({
                        "text": text,
                        "element_type": "caption",
                        "page_numbers": pages,
                        "section": current_section,
                    })
                continue

            # Do not invent descriptions for figures.
            if "picture" in label:
                continue

            text = cls._clean_text(cls._get_item_text(item))
            if not text:
                continue

            element_type = "list_item" if "list" in label else "paragraph"

            blocks.append({
                "text": text,
                "element_type": element_type,
                "page_numbers": pages,
                "section": current_section,
            })

        return blocks

    # ---------------------------------------------------------
    # 4. Fractiva evidence metadata
    # ---------------------------------------------------------

    @staticmethod
    def _build_metadata(
        source_metadata: dict[str, Any],
        text: str,
        page_numbers: list[int],
        section: str | None,
        element_type: str,
        chunking_strategy: str,
    ) -> dict[str, Any]:
        pages = sorted(set(page_numbers))

        return {
            **source_metadata,
            "evidence_span": text,
            "evidence_page": pages[0] if len(pages) == 1 else None,
            "page_numbers": pages,
            "section": section,
            "element_type": element_type,
            "chunking_strategy": chunking_strategy,
        }

    # ---------------------------------------------------------
    # 5. Actual LangChain SemanticChunker
    # ---------------------------------------------------------

    @classmethod
    def _semantic_chunk_prose(
        cls,
        blocks: list[dict[str, Any]],
        source_metadata: dict[str, Any],
    ) -> list[Document]:
        if not blocks:
            return []

        # Retain a page/section mapping for each original block.
        # SemanticChunker does not expose exact sentence-to-page
        # alignment in its output, so use conservative provenance.
        source_text = "\n\n".join(
            block["text"] for block in blocks
        ).strip()

        if not source_text:
            return []

        splitter = SemanticChunker(
            embeddings=EmbeddingWrapper(),
            breakpoint_threshold_type=cls.BREAKPOINT_THRESHOLD_TYPE,
            breakpoint_threshold_amount=cls.BREAKPOINT_THRESHOLD_AMOUNT,
        )

        split_docs = splitter.create_documents([source_text])
        output = []

        # Recover each chunk's location by matching its text against
        # the original extracted block text. If exact mapping fails,
        # preserve the full source-run provenance rather than guessing.
        for chunk in split_docs:
            chunk_text = cls._clean_text(chunk.page_content)
            if not chunk_text:
                continue

            matched_blocks = [
                block
                for block in blocks
                if block["text"] in chunk_text
                or chunk_text in block["text"]
            ]

            if matched_blocks:
                pages = sorted({
                    page
                    for block in matched_blocks
                    for page in block["page_numbers"]
                })
                sections = list(dict.fromkeys(
                    block["section"]
                    for block in matched_blocks
                    if block["section"]
                ))
                element_types = list(dict.fromkeys(
                    block["element_type"]
                    for block in matched_blocks
                ))
            else:
                pages = sorted({
                    page
                    for block in blocks
                    for page in block["page_numbers"]
                })
                sections = list(dict.fromkeys(
                    block["section"]
                    for block in blocks
                    if block["section"]
                ))
                element_types = list(dict.fromkeys(
                    block["element_type"]
                    for block in blocks
                ))

            metadata = cls._build_metadata(
                source_metadata=source_metadata,
                text=chunk_text,
                page_numbers=pages,
                section=" > ".join(sections) if sections else None,
                element_type=(
                    element_types[0]
                    if len(element_types) == 1
                    else "mixed"
                ),
                chunking_strategy="langchain_semantic",
            )

            output.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )

        return output

    # ---------------------------------------------------------
    # 6. Main processing entry point
    # ---------------------------------------------------------

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
            "cited_paper": path.name,
            "document_id": path.stem,
            "file_type": "pdf",
            "source_role": "clinical_evidence",
        }

        documents = []
        prose_run = []

        def flush_prose() -> None:
            nonlocal prose_run

            if prose_run:
                documents.extend(
                    cls._semantic_chunk_prose(
                        blocks=prose_run,
                        source_metadata=source_metadata,
                    )
                )
                prose_run = []

        for block in blocks:
            if block["element_type"] == "table":
                flush_prose()

                metadata = cls._build_metadata(
                    source_metadata=source_metadata,
                    text=block["text"],
                    page_numbers=block["page_numbers"],
                    section=block["section"],
                    element_type="table",
                    chunking_strategy="atomic",
                )

                documents.append(
                    Document(
                        page_content=block["text"],
                        metadata=metadata,
                    )
                )
            else:
                prose_run.append(block)

        flush_prose()

        return documents
