
import json
from pathlib import Path
from typing import Any

from app.generation.extractor import extract_claims
from app.generation.response_generator import ResponseGenerator
from app.generation.verifier import ClaimVerifier
from app.generation.output_writer import OutputWriter
from app.ingestion.retrieval.retrieval_pipeline import RetrievalPipeline


class RAGPipeline:
    def __init__(self):
        self.retrieval_pipeline = RetrievalPipeline()

    def prepare_claim_evidence(
        self,
        input_path: str | Path,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Extract claims and retrieve evidence separately for each claim.
        Retrieval does not invoke the LLM.
        """
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
                f"Claim: {claim['text']}\n"
                f"Population: {claim['population']}\n"
                f"Molecule: {claim['molecule']}\n"
                f"Cited paper: {claim['cited_paper']}"
            )

            try:
                retrieved_chunks = self.retrieval_pipeline.retrieve(
                    query=query,
                    top_k=top_k,
                )

                evidence = [
                    {
                        "evidence_id": f"{claim_id}_E{index}",
                        "text": chunk.page_content,
                        "metadata": dict(chunk.metadata or {}),
                    }
                    for index, chunk in enumerate(
                        retrieved_chunks,
                        start=1,
                    )
                ]

                batch.append({
                    "claim": claim,
                    "evidence": evidence,
                })

                print(
                    f"Retrieved {len(evidence)} chunks for {claim_id}"
                )

            except Exception as exc:
                # Preserve the claim, but never fabricate evidence.
                print(
                    f"Retrieval failed for {claim_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

                batch.append({
                    "claim": claim,
                    "evidence": [],
                    "retrieval_error": True,
                })

        return batch

    def verify_file(
        self,
        input_path: str | Path = "data/claims.json",
        top_k: int = 5,
        output_path: str | Path = "outputs/predictions.json",
        analysis_path: str | Path | None = "outputs/llm_analysis.json",
    ) -> list[dict[str, Any]]:
        """
        Complete pipeline:

        1. Extract claims.
        2. Retrieve evidence with Python.
        3. Make ONE batched LLM invocation.
        4. Verify the LLM results using Python.
        5. Write final predictions.

        The verifier and output writer must not call an LLM.
        """
        batch = self.prepare_claim_evidence(
            input_path=input_path,
            top_k=top_k,
        )

        if not batch:
            predictions: list[dict[str, Any]] = []
            OutputWriter.write_predictions(
                predictions=predictions,
                output_path=str(output_path),
            )
            return predictions

        # ONE LLM call for the entire batch.
        llm_analyses = ResponseGenerator.generate_batch(batch)

        # Optional: save the intermediate LLM analysis for debugging.
        # This writes a file; it does not make another LLM call.
        if analysis_path is not None:
            analysis_path = Path(analysis_path)
            analysis_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with analysis_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {"predictions": llm_analyses},
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"Saved LLM analysis to {analysis_path.resolve()}"
            )

        # Python-only verification. Pass the original retrieved evidence
        # alongside the LLM analysis so the verifier can validate quotes.
        predictions = ClaimVerifier.verify_batch(
            batch=batch,
            llm_predictions=llm_analyses,
        )

        # Save the FINAL verified predictions.
        OutputWriter.write_predictions(
            predictions=predictions,
            output_path=str(output_path),
        )

        print(
            f"Pipeline complete: {len(predictions)} predictions written."
        )

        return predictions

    def generate_analysis(
        self,
        input_path: str | Path = "data/claims.json",
        top_k: int = 5,
        output_path: str | Path = "outputs/predictions.json",
    ) -> list[dict[str, Any]]:
        """
        Backwards-compatible method name.

        It now runs the complete verification pipeline and writes
        final predictions, rather than only saving raw LLM analysis.
        """
        return self.verify_file(
            input_path=input_path,
            top_k=top_k,
            output_path=output_path,
        )


if __name__ == "__main__":
    pipeline = RAGPipeline()

    pipeline.verify_file(
        input_path="data/claims.json",
        top_k=5,
        output_path="outputs/predictions.json",
    )
