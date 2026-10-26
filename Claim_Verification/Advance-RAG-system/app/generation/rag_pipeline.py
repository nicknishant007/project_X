import json
import math
import re
from pathlib import Path
from typing import Any, Generator

from app.generation.extractor import extract_claims, parse_claims
from app.generation.llm import LLM
from app.generation.output_writer import OutputWriter
from app.generation.qa_prompt import build_qa_prompt
from app.generation.response_generator import ResponseGenerator
from app.generation.verifier import ClaimVerifier
from app.ingestion.retrieval.retrieval_pipeline import RetrievalPipeline

CLAIM_BATCH_SIZE = 8   # claims per LLM call (all evidence is retrieved first, then sent in batches)

_PDF_RE = re.compile(r"[^\s()\[\],;]+\.pdf", re.IGNORECASE)
_ALIAS_STOPWORDS = {"plus", "abstract", "prescribing", "information", "paper", "papers", "page"}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


class RAGPipeline:
    def __init__(self):
        self.retrieval_pipeline = RetrievalPipeline()

    # ------------------------------------------------------------------
    # cited_paper -> indexed document ids
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_documents(cited_paper: str, known_ids: list[str]) -> list[str]:
        """
        Map a free-text citation to indexed document_ids (file stems), e.g.
          "SOLIQUA_LBL.pdf (Prescribing Information, page 4)" -> ["SOLIQUA_LBL"]
          "papers/Tabak2020_PSM.pdf (plus LixiLan-L abstract and SoliMix)"
              -> ["Tabak2020_PSM", "LixiLan-L_Aroda2016_abstract", "SoliMix_Rosenstock2021"]
        """
        known = {_norm(i): i for i in known_ids}
        found: list[str] = []

        for match in _PDF_RE.findall(cited_paper or ""):
            stem = _norm(Path(match.replace("\\", "/")).stem)
            if stem in known:
                found.append(known[stem])
            else:  # tolerate suffixes such as "_ab12cd34" added on duplicate uploads
                found.extend(i for n, i in known.items() if n.startswith(stem))

        remainder = _PDF_RE.sub(" ", cited_paper or "")
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", remainder):
            w = _norm(word)
            if len(w) >= 5 and w not in _ALIAS_STOPWORDS:
                found.extend(i for n, i in known.items() if w in n)

        return list(dict.fromkeys(found))

    @staticmethod
    def resolve_for_claim(claim: dict, known_ids: list[str]) -> list[str]:
        """
        Pick the documents that belong to a claim.
          1. explicit claim["document_id"] / claim["document_ids"] (optional override)
          2. claim_id appears as a token in the document id:
                C2 -> "C2C4C6 (1)" ; C7 -> "C5C7 (3)" and "C7" ; C3 -> "C3"
          3. fallback: parse the free-text cited_paper (filename / alias)
        Returns [] when nothing matches -> the claim fails closed.
        """
        explicit = claim.get("document_ids") or claim.get("document_id") or []
        if isinstance(explicit, str):
            explicit = [explicit]
        wanted = {_norm(x) for x in explicit}
        hits = [i for i in known_ids if _norm(i) in wanted]
        if hits:
            return hits

        cid = (claim.get("claim_id") or "").strip().upper()
        if cid:
            matched = []
            for doc_id in known_ids:
                tokens = {t.upper() for t in re.findall(r"[A-Za-z]+\d+", doc_id)}
                if cid in tokens or _norm(doc_id) == _norm(cid):
                    matched.append(doc_id)
            if matched:
                return matched

        return RAGPipeline.resolve_documents(claim.get("cited_paper", ""), known_ids)

    # ------------------------------------------------------------------
    # Evidence retrieval (strictly limited to the cited paper(s))
    # ------------------------------------------------------------------
    def prepare_claim_evidence(self, claims: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        known_ids = self.retrieval_pipeline.list_documents()
        batch: list[dict[str, Any]] = []

        for claim in claims:
            claim_id = claim["claim_id"]
            print(f"\nRetrieving evidence for claim: {claim_id}")
            try:
                doc_ids = self.resolve_for_claim(claim, known_ids)
                if not doc_ids:
                    raise LookupError(
                        f"No indexed document matches claim {claim_id!r} "
                        f"(indexed: {known_ids or 'nothing'}). Name the PDF after the claim id "
                        "(e.g. C2C4C6.pdf) or add a 'document_id' field to the claim. "
                        "Refusing to use evidence from another paper."
                    )
                per_doc = top_k if len(doc_ids) == 1 else max(3, math.ceil(top_k / len(doc_ids)) + 1)
                query = claim["text"]   # the claim text IS the retrieval query

                chunks = []
                for doc_id in doc_ids:
                    chunks.extend(self.retrieval_pipeline.retrieve(
                        query=query, top_k=per_doc, document_ids={doc_id}))
                if not chunks:
                    raise LookupError("No chunks were retrieved from the cited paper.")

                evidence = [
                    {"evidence_id": f"{claim_id}_E{i}",
                     "text": chunk.page_content,
                     "metadata": dict(chunk.metadata or {})}
                    for i, chunk in enumerate(chunks, start=1)
                ]
                batch.append({"claim": claim, "evidence": evidence})
                print(f"Retrieved {len(evidence)} chunks from {doc_ids} for {claim_id}")
            except Exception as exc:
                print(f"Retrieval failed for {claim_id}: {type(exc).__name__}: {exc}")
                batch.append({"claim": claim, "evidence": [], "retrieval_error": True,
                              "retrieval_error_message": f"{type(exc).__name__}: {exc}"})
        return batch

    # ------------------------------------------------------------------
    # Claim verification
    # ------------------------------------------------------------------
    def verify_claims(
        self,
        claims: list[dict[str, Any]],
        top_k: int = 5,
        output_path: str | Path = "outputs/predictions.json",
        analysis_path: str | Path | None = "outputs/llm_analysis.json",
    ) -> list[dict[str, Any]]:
        batch = self.prepare_claim_evidence(claims, top_k=top_k)
        if not batch:
            OutputWriter.write_predictions([], output_path=str(output_path))
            return []

        analyses: list[dict[str, Any]] = []
        answerable = [item for item in batch if item.get("evidence")]

        if answerable and not LLM.has_api_key():
            print("WARNING: GOOGLE_API_KEY is not set - every claim will be 'cannot_verify'.")
        elif answerable:
            for i in range(0, len(answerable), CLAIM_BATCH_SIZE):
                group = answerable[i:i + CLAIM_BATCH_SIZE]
                try:
                    analyses.extend(ResponseGenerator.generate_batch(group))
                except Exception as exc:  # fail closed for this group only
                    ids = [g["claim"]["claim_id"] for g in group]
                    print(f"LLM analysis failed for {ids}; failing closed: {type(exc).__name__}: {exc}")

        if analysis_path is not None:
            path = Path(analysis_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as file:
                json.dump({"predictions": analyses}, file, indent=2, ensure_ascii=False)

        predictions = ClaimVerifier.verify_batch(batch=batch, llm_predictions=analyses)
        OutputWriter.write_predictions(predictions=predictions, output_path=str(output_path))
        print(f"Pipeline complete: {len(predictions)} predictions written.")
        return predictions

    def verify_file(
        self,
        input_path: str | Path = "data/claims.json",
        top_k: int = 5,
        output_path: str | Path = "outputs/predictions.json",
        analysis_path: str | Path | None = "outputs/llm_analysis.json",
    ) -> list[dict[str, Any]]:
        return self.verify_claims(extract_claims(input_path), top_k=top_k,
                                  output_path=output_path, analysis_path=analysis_path)

    def verify_data(self, data: Any, top_k: int = 5, **kwargs) -> list[dict[str, Any]]:
        return self.verify_claims(parse_claims(data), top_k=top_k, **kwargs)

    generate_analysis = verify_file

    # ------------------------------------------------------------------
    # Plain chat Q&A (used by /chat/stream and main.py)
    # ------------------------------------------------------------------
    def stream_query(self, user_query: str, top_k: int = 5) -> Generator[str, None, None]:
        chunks = self.retrieval_pipeline.retrieve(user_query, top_k=top_k)
        if not chunks:
            yield "I couldn't find anything relevant. Upload and index a PDF first."
            return
        if not LLM.has_api_key():
            yield "GOOGLE_API_KEY is not set, so I can't generate an answer. Add it to .env and restart."
            return

        parts = []
        for i, chunk in enumerate(chunks, start=1):
            md = chunk.metadata or {}
            pages = md.get("page_numbers") or []
            parts.append(f"[Source {i}: {md.get('file_name', 'unknown')} | pages {pages}]\n{chunk.page_content}")

        messages = build_qa_prompt().invoke({"question": user_query, "context": "\n\n".join(parts)})
        for piece in LLM.get_llm().stream(messages):
            text = ResponseGenerator._content_as_text(piece)
            if text:
                yield text


if __name__ == "__main__":
    RAGPipeline().verify_file()
