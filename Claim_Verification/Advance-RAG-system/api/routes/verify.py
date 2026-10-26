import json
import traceback
from collections import Counter

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.services.rag_service import RAGService

router = APIRouter()
MAX_BYTES = 2_000_000


@router.post("/claims")
def verify_claims(file: UploadFile = File(...), top_k: int = Form(5)):
    """Upload a claims JSON -> run the verifier -> return predictions.
    Also writes outputs/predictions.json (same as `python verify.py`).
    Sync `def` on purpose: FastAPI runs it in a worker thread, so the event
    loop (and the chat UI) stays responsive during the long LLM calls."""
    if not (file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Please upload a .json claims file.")

    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Claims file is too large (max 2 MB).")

    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    try:
        claims, predictions = RAGService.verify_claims(data, top_k=max(1, min(top_k, 15)))
    except ValueError as exc:                      # bad claim structure
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}")

    return {
        "success": True,
        "claims": {c["claim_id"]: c["text"] for c in claims},
        "predictions": predictions,
        "summary": dict(Counter(p["verdict"] for p in predictions)),
    }
