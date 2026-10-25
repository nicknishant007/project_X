import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.generation.llm import LLM
from app.generation.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ResponseGenerator:
    LOG_PATH = Path("outputs/llm_log.jsonl")
    MODEL_PARAMETERS = {
        "model": "gemini-2.5-flash",
        "temperature": 0.3,
        "max_tokens": 8000,
        "max_retries": 3,
        "timeout": 60,
    }

    @classmethod
    def _append_log(cls, record: dict[str, Any]) -> None:
        """Append one audit event; never include environment variables/API keys."""
        path = Path(os.getenv("LLM_LOG_PATH", str(cls.LOG_PATH)))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _content_as_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
        if not isinstance(content, str):
            raise ValueError("LLM returned an unexpected response type.")
        return content

    @classmethod
    def generate_batch(cls, batch: list[dict]) -> list[dict]:
        """Make one batched LLM request and audit the request and raw response."""
        if not batch:
            return []

        payload = [
            {
                "claim": item["claim"],
                "evidence": item["evidence"],
                "retrieval_error": item.get("retrieval_error"),
            }
            for item in batch
        ]
        inputs = {
            "claim": json.dumps([item["claim"] for item in batch], ensure_ascii=False),
            "context": json.dumps(payload, ensure_ascii=False, default=str),
        }
        prompt = PromptBuilder.build()
        prompt_value = prompt.invoke(inputs)
        messages = [
            {"role": getattr(message, "type", "unknown"),
             "content": getattr(message, "content", str(message))}
            for message in prompt_value.to_messages()
        ]
        request_id = datetime.now(timezone.utc).isoformat()

        try:
            llm = LLM.get_llm()
            # Invoke the model with the exact rendered messages so the logged
            # request matches what was submitted.
            response = llm.invoke(prompt_value.to_messages())
            raw_content = cls._content_as_text(response)
            cls._append_log({
                "timestamp": request_id,
                "event": "llm_response",
                "request": {"messages": messages},
                "parameters": dict(cls.MODEL_PARAMETERS),
                "raw_response": raw_content,
            })
        except Exception as exc:
            logger.exception("Batch LLM generation failed")
            try:
                cls._append_log({
                    "timestamp": request_id,
                    "event": "llm_error",
                    "request": {"messages": messages},
                    "parameters": dict(cls.MODEL_PARAMETERS),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            except Exception:
                logger.exception("Failed to write LLM error audit record")
            raise

        try:
            result = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON.") from exc
        if not isinstance(result, dict) or not isinstance(result.get("predictions"), list):
            raise ValueError("LLM JSON must contain a 'predictions' array.")

        predictions = result["predictions"]
        expected_ids = {item["claim"]["claim_id"] for item in batch}
        returned_ids = [
            prediction.get("claim_id")
            for prediction in predictions
            if isinstance(prediction, dict)
        ]
        if (
            len(returned_ids) != len(predictions)
            or len(returned_ids) != len(expected_ids)
            or set(returned_ids) != expected_ids
            or len(set(returned_ids)) != len(returned_ids)
        ):
            raise ValueError("LLM predictions do not match the input claim IDs.")
        return predictions
