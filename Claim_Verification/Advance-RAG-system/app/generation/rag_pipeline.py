import inspect
import json
import re
from pathlib import Path
from typing import Any

from app.generation.extractor import extract_claims
from app.generation.response_generator import ResponseGenerator
from app.generation.verifier import ClaimVerifier, REQUIRED_CHECKS
from app.generation.output_writer import OutputWriter
from app.ingestion.retrieval.retrieval_pipeline import RetrievalPipeline


class RAGPipeline:
    def __init__(self):
        self.retrieval_pipeline = RetrievalPipeline()

    @staticmethod
    def _normalise_path(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        value = value.replace("\\\\", "/").strip().lower()
        return re.sub(r"/+", "/", value)

    @classmethod
    def _matches_cited_paper(cls, metadata: dict[str, Any], cited_paper: str) -> bool:
        """Match the citation to source metadata; never infer from chunk text."""
        expected = cls._normalise_path(cited_paper)
        expected_name = expected.rsplit("/", 1)[-1]
        expected_stem = expected_name.rsplit(".", 1)[0]
        keys = (
            "cited_paper", "source", "source_path", "file_path", "filepath",
            "filename", "file_name", "document", "document_name", "doc_id",
            "paper", "paper_path",
        )
        for key in keys:
            value = metadata.get(key)
            if not isinstance(value, str):
                continue
            actual = cls._normalise_path(value)
            actual_name = actual.rsplit("/", 1)[-1]
            actual_stem = actual_name.rsplit(".", 1)[0]
            if actual == expected or actual_name == expected_name or actual_stem == expected_stem:
                return True
        return False

    def _retrieve_cited_paper(self, query: str, top_k: int, cited_paper: str):
        """Use a native citation filter if RetrievalPipeline exposes one.

        If not, post-filter returned chunks and fail closed if no matching
        metadata is present. A native metadata filter is preferred because
        post-filtering top-k global results can miss relevant chunks.
        """
        retrieve = self.retrieval_pipeline.retrieve
        try:
            parameters = inspect.signature(retrieve).parameters
        except (TypeError, ValueError):
            parameters = {}

        filter_parameter = next(
            (name for name in ("cited_paper",)
             if name in parameters),
            None,
        )
        if filter_parameter:
            chunks = retrieve(query=query, top_k=top_k, **{filter_parameter: cited_paper})
        else:
            chunks = retrieve(query=query, top_k=max(top_k * 5, 20))

        matched = [
            chunk for chunk in (chunks or [])
            if self._matches_cited_paper(dict(getattr(chunk, "metadata", {}) or {}), cited_paper)
        ]
        if not matched:
            raise LookupError(
                f"No retrieved chunks could be verified as belonging to cited paper "
                f"{cited_paper!r}. Check that the paper is indexed and its metadata "
                "contains a filename/path field. Refusing to use evidence from another paper."
            )
        return matched[:top_k]

    def prepare_claim_evidence(self, input_path: str | Path, top_k: int = 5) -> list[dict[str, Any]]:
        claims = extract_claims(input_path)
        batch: list[dict[str, Any]] = []
        seen_claim_ids = set()

        for claim in claims:
            claim_id = claim["claim_id"]
            if claim_id in seen_claim_ids:
                raise ValueError(f"Duplicate claim_id: {claim_id}")
            seen_claim_ids.add(claim_id)
            print(f"\nRetrieving evidence for claim: {claim_id}")
            query = (
                f"Claim: {claim['text']}\nPopulation: {claim['population']}\n"
                f"Molecule: {claim['molecule']}\nCited paper: {claim['cited_paper']}"
            )
            try:
                retrieved_chunks = self._retrieve_cited_paper(
                    query=query, top_k=top_k, cited_paper=claim["cited_paper"]
                )
                evidence = [
                    {
                        "evidence_id": f"{claim_id}_E{index}",
                        "text": chunk.page_content,
                        "metadata": dict(chunk.metadata or {}),
                    }
                    for index, chunk in enumerate(retrieved_chunks, start=1)
                ]
                batch.append({"claim": claim, "evidence": evidence})
                print(f"Retrieved {len(evidence)} cited-paper chunks for {claim_id}")
            except Exception as exc:
                print(f"Retrieval failed for {claim_id}: {type(exc).__name__}: {exc}")
                batch.append({
                    "claim": claim, "evidence": [], "retrieval_error": True,
                    "retrieval_error_message": f"{type(exc).__name__}: {exc}",
                })
        return batch

    def verify_file(
        self,
        input_path: str | Path = "data/claims.json",
        top_k: int = 5,
        output_path: str | Path = "outputs/predictions.json",
        analysis_path: str | Path | None = "outputs/llm_analysis.json",
    ) -> list[dict[str, Any]]:
        batch = self.prepare_claim_evidence(input_path=input_path, top_k=top_k)
        if not batch:
            OutputWriter.write_predictions([], output_path=str(output_path))
            return []

        try:
            llm_analyses = ResponseGenerator.generate_batch(batch)
        except Exception as exc:
            # Assignment requires graceful failure and prohibits supported
            # verdicts when configuration/model calls fail.
            print(f"LLM analysis failed; failing closed: {type(exc).__name__}: {exc}")
            llm_analyses = []

        if analysis_path is not None:
            path = Path(analysis_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as file:
                json.dump({"predictions": llm_analyses}, file, indent=2, ensure_ascii=False)

        predictions = ClaimVerifier.verify_batch(batch=batch, llm_predictions=llm_analyses)
        OutputWriter.write_predictions(predictions=predictions, output_path=str(output_path))
        print(f"Pipeline complete: {len(predictions)} predictions written.")
        return predictions

    def generate_analysis(
        self,
        input_path: str | Path = "data/claims.json",
        top_k: int = 5,
        output_path: str | Path = "outputs/predictions.json",
    ) -> list[dict[str, Any]]:
        return self.verify_file(input_path=input_path, top_k=top_k, output_path=output_path)


if __name__ == "__main__":
    RAGPipeline().verify_file(
        input_path="data/claims.json", top_k=5, output_path="outputs/predictions.json"
    )
