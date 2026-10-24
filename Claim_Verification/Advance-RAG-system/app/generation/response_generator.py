import json
import logging

from app.generation.llm import LLM
from app.generation.prompt_builder import PromptBuilder


logger = logging.getLogger(__name__)


class ResponseGenerator:

    @staticmethod
    def generate_batch(batch: list[dict]) -> list[dict]:
        """
        Send all claims and their associated evidence to the LLM
        in one request. Return one analysis object per claim.
        """

        if not batch:
            return []

        payload = []

        for item in batch:
            claim = item["claim"]
            evidence = item["evidence"]

            payload.append({
                "claim": claim,
                "evidence": evidence,
                "retrieval_error": item.get("retrieval_error"),
            })

        batch_context = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )

        prompt = PromptBuilder.build()
        llm = LLM.get_llm()

        # Ask Gemini for a JSON response rather than streamed prose.
        chain = prompt | llm

        logger.info(
            "Submitting batch of %d claims to the LLM",
            len(batch),
        )

        try:
            response = chain.invoke({
                "claim": json.dumps(
                    [item["claim"] for item in batch],
                    ensure_ascii=False,
                ),
                "context": batch_context,
            })
        except Exception:
            logger.exception("Batch LLM generation failed")
            raise

        content = getattr(response, "content", response)

        # Some chat models return content as a list of blocks.
        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )

        if not isinstance(content, str):
            raise ValueError("LLM returned an unexpected response type.")

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON: %s", content)
            raise ValueError(
                "LLM response was not valid JSON."
            ) from exc

        # Expect {"predictions": [...]} from the batch prompt.
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object from the LLM.")

        predictions = result.get("predictions")

        if not isinstance(predictions, list):
            raise ValueError(
                "LLM JSON must contain a 'predictions' array."
            )

        expected_ids = {
            item["claim"]["claim_id"]
            for item in batch
        }

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
            raise ValueError(
                "LLM predictions do not match the input claim IDs."
            )

        logger.info(
            "Received %d predictions from one batch LLM call",
            len(predictions),
        )

        return predictions
