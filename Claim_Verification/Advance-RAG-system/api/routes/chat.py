import json
import traceback
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from api.services.rag_service import RAGService
from guardrails.input_guard import InputGuard

router = APIRouter()


# ============================================================
# Request Model
# ============================================================

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


# ============================================================
# Helper
# ============================================================

def sse(data: dict):
    """
    Convert dict -> Server Sent Event
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def validate_question(question: str):
    """
    Validate user input using Input Guardrail.

    Returns:
        None -> Question is safe.
        JSONResponse -> Validation failed.
    """

    is_safe, reason = InputGuard.validate(question)

    if not is_safe:

        def blocked():

            yield sse({
                "type": "error",
                "content": reason,
            })

            yield sse({
                "type": "done",
            })

        return StreamingResponse(
            blocked(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return None


# ============================================================
# Stream Endpoint
# ============================================================

@router.post("/stream")
async def stream_chat(req: ChatRequest):

    question = req.question.strip()

    error = validate_question(question)

    if error:
        return error

    def generate():

        try:

            for token in RAGService.stream_answer(
                question=question,
                top_k=req.top_k,
            ):

                if not token:
                    continue

                yield sse({
                    "type": "token",
                    "content": token,
                })

            yield sse({
                "type": "done",
            })

        except GeneratorExit:
            return

        except Exception as e:

            print(traceback.format_exc())

            yield sse({
                "type": "error",
                "content": str(e),
            })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# Retrieve Sources
# ============================================================

@router.post("/retrieve")
async def retrieve(req: ChatRequest):

    question = req.question.strip()

    error = validate_question(question)

    if error:
        return error

    try:

        docs = RAGService.retrieve_sources(
            question=question,
            top_k=req.top_k,
        )

        return JSONResponse({
            "success": True,
            "sources": docs,
        })

    except Exception as e:

        print(traceback.format_exc())

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(e),
            },
        )