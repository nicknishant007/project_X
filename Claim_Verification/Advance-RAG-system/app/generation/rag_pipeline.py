from pathlib import Path
from typing import Any

from app.generation.extractor import extract_claims
from app.generation.response_generator import ResponseGenerator
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
        Extract claims from JSON and retrieve relevant chunks
        separately for each claim.
        """
        claims = extract_claims(input_path)

        batch = []
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

                evidence = []

                for index, chunk in enumerate(retrieved_chunks, start=1):
                    evidence.append({
                        "evidence_id": f"{claim_id}_E{index}",
                        "text": chunk.page_content,
                        "metadata": dict(chunk.metadata or {}),
                    })

                batch.append({
                    "claim": claim,
                    "evidence": evidence,
                })

                print(
                    f"Retrieved {len(evidence)} chunks for {claim_id}"
                )

            except Exception:
                # Keep the original claim, but do not fabricate evidence.
                print(f"Retrieval failed for claim {claim_id}")

                batch.append({
                    "claim": claim,
                    "evidence": [],
                    "retrieval_error": True,
                })

        return batch

    def generate_analysis(
        self,
        input_path: str | Path = "data/claim.json",
        top_k: int = 5,
        output_path: str | Path = "llm_analysis.json",
    ) -> list[dict[str, Any]]:
        """
        Extract claims, retrieve evidence, call the LLM once
        with the complete batch, and save its analysis.
        """
        batch = self.prepare_claim_evidence(
            input_path=input_path,
            top_k=top_k,
        )

        if not batch:
            analyses = []
        else:
            # One LLM invocation for all claim-evidence pairs.
            analyses = ResponseGenerator.generate_batch(batch)

        # Save raw LLM analyses, not final verified predictions.
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"predictions": analyses},
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Saved {len(analyses)} analyses to {output_path.resolve()}")

        return analyses
