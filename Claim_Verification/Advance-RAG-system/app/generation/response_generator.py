import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.generation.llm import LLM
from app.generation.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ResponseGenerator:
    LOG_PATH = Path("outputs/llm_log.jsonl")

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
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        if not isinstance(content, str):
            raise ValueError("LLM returned an unexpected response type.")
        return content

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Parse JSON, tolerating ```json fences / stray prose around the object."""
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start != -1 and end > start:
                return json.loads(cleaned[start:end + 1])
            raise

    @staticmethod
    def _slim_evidence(evidence: list[dict]) -> list[dict]:
        """Send the LLM only what it needs (no duplicated text / internal fields)."""
        slim = []
        for item in evidence:
            md = item.get("metadata") or {}
            slim.append({
                "evidence_id": item.get("evidence_id"),
                "text": item.get("text"),
                "metadata": {
                    "source": md.get("file_name") or md.get("cited_paper"),
                    "evidence_page": md.get("evidence_page"),
                    "page_numbers": md.get("page_numbers"),
                    "section": md.get("section"),
                    "element_type": md.get("element_type"),
                },
            })
        return slim

    @classmethod
    def generate_batch(cls, batch: list[dict]) -> list[dict]:
        """One batched LLM request; logs the request and the raw response."""
        if not batch:
            return []

        payload = [
            {
                "claim": item["claim"],
                "evidence": cls._slim_evidence(item.get("evidence", [])),
                "retrieval_error": item.get("retrieval_error"),
            }
            for item in batch
        ]
        inputs = {
            "claim": json.dumps([item["claim"] for item in batch], ensure_ascii=False),
            "context": json.dumps(payload, ensure_ascii=False, default=str),
        }
        prompt_value = PromptBuilder.build().invoke(inputs)
        messages = [
            {"role": getattr(m, "type", "unknown"),
             "content": getattr(m, "content", str(m))}
            for m in prompt_value.to_messages()
        ]
        request_id = datetime.now(timezone.utc).isoformat()

        try:
            llm = LLM.get_llm()
            response = llm.invoke(prompt_value.to_messages())
            raw_content = cls._content_as_text(response)
            cls._append_log({
                "timestamp": request_id,
                "event": "llm_response",
                "request": {"messages": messages},
                "parameters": dict(LLM.PARAMS),
                "raw_response": raw_content,
            })
        except Exception as exc:
            logger.exception("Batch LLM generation failed")
            try:
                cls._append_log({
                    "timestamp": request_id,
                    "event": "llm_error",
                    "request": {"messages": messages},
                    "parameters": dict(LLM.PARAMS),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            except Exception:
                logger.exception("Failed to write LLM error audit record")
            raise

        try:
            result = cls._parse_json(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON.") from exc
        if not isinstance(result, dict) or not isinstance(result.get("predictions"), list):
            raise ValueError("LLM JSON must contain a 'predictions' array.")

        # Lenient on shape, strict on content: keep only well-formed predictions for
        # known claim ids (first occurrence). Anything missing is failed closed
        # (cannot_verify) by the verifier instead of discarding the whole batch.
        expected = {item["claim"]["claim_id"] for item in batch}
        seen, valid = set(), []
        for prediction in result["predictions"]:
            if not isinstance(prediction, dict):
                continue
            cid = prediction.get("claim_id")
            if cid in expected and cid not in seen:
                seen.add(cid)
                valid.append(prediction)
        if not valid:
            raise ValueError("LLM predictions did not contain any known claim IDs.")
        return valid
