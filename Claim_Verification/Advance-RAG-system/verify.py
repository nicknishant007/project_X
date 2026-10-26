"""Usage: python verify.py data/claims.json [--top-k 5] [--output outputs/predictions.json] [--ingest]"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify claims against indexed clinical papers.")
    parser.add_argument("claims_file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="outputs/predictions.json")
    parser.add_argument("--ingest", action="store_true",
                        help="Index the PDFs in data/incoming before verifying.")
    args = parser.parse_args()

    claims_path = Path(args.claims_file).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    os.chdir(ROOT)               # config / storage paths are relative to the project root
    sys.path.insert(0, str(ROOT))

    try:
        from app.generation.rag_pipeline import RAGPipeline
        from app.generation.llm import LLM

        if not LLM.has_api_key():
            print("WARNING: GOOGLE_API_KEY is not set -> all claims will be 'cannot_verify'.")

        if args.ingest:
            from app.ingestion.registry.registry_manager import RegistryManager
            from app.ingestion.pipeline.ingestion_pipeline import IngestionPipeline
            RegistryManager.initialize()
            IngestionPipeline.run()

        pipeline = RAGPipeline()
        if not pipeline.retrieval_pipeline.list_documents():
            print("WARNING: no documents are indexed. Put the PDFs in data/incoming and "
                  "run with --ingest (or upload them in the UI).")

        predictions = pipeline.verify_file(claims_path,
                                           top_k=args.top_k, output_path=output_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1

    for p in predictions:
        print(f"{p['claim_id']}: {p['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
