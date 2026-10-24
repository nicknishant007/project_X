
import re
from typing import Any


REQUIRED_CHECKS = (
    "figure",
    "population",
    "drug_indication",
    "actually_stated",
)

VALID_STATUSES = {"match", "mismatch", "unclear"}


class ClaimVerifier:
    """
    Lightweight, deterministic validation of LLM claim-verification results.

    This class:
      - Makes ZERO LLM calls.
      - Validates the four check objects returned by the LLM.
      - Confirms evidence quotations occur in retrieved chunks.
      - Uses retrieved metadata for evidence page numbers.
      - Computes numerical ratios when source values are unambiguous.
      - Produces the final prediction schema expected by OutputWriter.

    The LLM remains responsible for semantic reasoning.
    Python validates traceability, structure, and arithmetic.
    """

    @staticmethod
    def _normalise_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _page_number(evidence_item: dict[str, Any]) -> int | None:
        """Return a page number only when supported by source metadata."""
        metadata = evidence_item.get("metadata") or {}

        for key in ("evidence_page", "page_number", "page_no", "page"):
            value = metadata.get(key)

            if isinstance(value, bool):
                continue

            if isinstance(value, int):
                # Common PDF loaders use zero-based page metadata.
                if key == "page":
                    return value + 1 if value >= 0 else None
                return value if value > 0 else None

            if isinstance(value, str) and value.isdigit():
                number = int(value)
                if key == "page":
                    return number + 1
                return number if number > 0 else None

        for key in ("page_numbers", "pages"):
            pages = metadata.get(key)
            if isinstance(pages, list) and len(pages) == 1:
                value = pages[0]
                if isinstance(value, int) and value > 0:
                    return value
                if isinstance(value, str) and value.isdigit():
                    number = int(value)
                    if number > 0:
                        return number

        return None

    @classmethod
    def _find_evidence(
        cls,
        evidence_span: Any,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Find a verbatim quotation in the retrieved chunks.

        Whitespace differences are tolerated, but the quotation must
        otherwise match the source text.
        """
        if not isinstance(evidence_span, str) or not evidence_span.strip():
            return None

        quote = cls._normalise_text(evidence_span)

        for item in evidence:
            source_text = item.get("text", "")
            if not isinstance(source_text, str):
                continue

            if quote in cls._normalise_text(source_text):
                return item

        return None

    @classmethod
    def _validate_check(
        cls,
        raw_check: Any,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate one LLM check without repeating semantic reasoning."""
        if not isinstance(raw_check, dict):
            return {
                "status": "unclear",
                "evidence_page": None,
                "evidence_span": None,
                "source_values": None,
                "reason": "Missing or invalid check object from the LLM.",
            }

        status = raw_check.get("status")
        reason = raw_check.get("reason", "")

        if status not in VALID_STATUSES:
            return {
                "status": "unclear",
                "evidence_page": None,
                "evidence_span": None,
                "source_values": None,
                "reason": "The LLM returned an invalid check status.",
            }

        if not isinstance(reason, str):
            reason = ""

        span = raw_check.get("evidence_span")
        evidence_item = cls._find_evidence(span, evidence)

        # A semantic status without traceable evidence cannot be trusted.
        if evidence_item is None:
            return {
                "status": "unclear",
                "evidence_page": None,
                "evidence_span": None,
                "source_values": None,
                "reason": (
                    "The supporting quotation is missing or could not "
                    "be located in the retrieved evidence."
                ),
            }

        source_values = raw_check.get("source_values")

        # Accept only JSON-compatible source values.
        if not isinstance(source_values, (list, dict, type(None))):
            source_values = None

        return {
            "status": status,
            "evidence_page": cls._page_number(evidence_item),
            "evidence_span": span,
            "source_values": source_values,
            "reason": reason,
        }

    @staticmethod
    def _claimed_multiplier(claim_text: str) -> float | None:
        """Extract an explicit multiplier such as '~2x' or '2-fold'."""
        if not isinstance(claim_text, str):
            return None

        patterns = (
            r"~\s*(\d+(?:\.\d+)?)\s*x\b",
            r"(\d+(?:\.\d+)?)\s*[- ]?fold\b",
            r"(\d+(?:\.\d+)?)\s*x\b",
            r"(\d+(?:\.\d+)?)\s*times\b",
        )

        for pattern in patterns:
            match = re.search(pattern, claim_text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if value > 0:
                    return value

        return None

    @staticmethod
    def _numeric_value(item: Any) -> float | None:
        """Read a number from a structured source-value object."""
        if not isinstance(item, dict):
            return None

        value = item.get("value")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

        return float(value)

    @staticmethod
    def _label(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    @classmethod
    def _arm_role(cls, value: Any) -> str | None:
        """
        Identify an arm only when its label explicitly identifies its role.
        Ambiguous treatment names are not guessed.
        """
        label = cls._label(value)

        treatment_labels = {
            "treatment arm",
            "treatment group",
            "intervention arm",
            "intervention group",
            "experimental arm",
            "experimental group",
        }

        comparator_labels = {
            "comparator arm",
            "comparator group",
            "control arm",
            "control group",
            "comparison arm",
            "comparison group",
            "placebo arm",
            "placebo group",
        }

        if label in treatment_labels:
            return "treatment"

        if label in comparator_labels:
            return "comparator"

        return None

    @classmethod
    def _compute_figure(
        cls,
        claim_text: str,
        source_values: Any,
    ) -> dict[str, Any] | None:
        """
        Calculate treatment/comparator ratio using Python.

        Expected source_values format:
        [
            {
                "value": 1.63,
                "measure": "HbA1c reduction",
                "unit": "percentage points",
                "arm": "treatment arm",
                "timepoint": "Week 30"
            },
            {
                "value": 1.34,
                "measure": "HbA1c reduction",
                "unit": "percentage points",
                "arm": "comparator arm",
                "timepoint": "Week 30"
            }
        ]

        Returns None if the values cannot safely be compared.
        Does not apply an arbitrary percentage tolerance.
        """
        claimed = cls._claimed_multiplier(claim_text)

        if claimed is None:
            return None

        if not isinstance(source_values, list) or len(source_values) != 2:
            return None

        first, second = source_values

        first_value = cls._numeric_value(first)
        second_value = cls._numeric_value(second)

        if first_value is None or second_value is None:
            return None

        if first_value <= 0 or second_value <= 0:
            return None

        # Require the same outcome, units, and timepoint.
        for key in ("measure", "unit", "timepoint"):
            first_label = cls._label(first.get(key))
            second_label = cls._label(second.get(key))

            if not first_label or first_label != second_label:
                return None

        first_role = cls._arm_role(first.get("arm"))
        second_role = cls._arm_role(second.get("arm"))

        if {first_role, second_role} != {"treatment", "comparator"}:
            return None

        treatment_value = (
            first_value if first_role == "treatment" else second_value
        )
        comparator_value = (
            first_value if first_role == "comparator" else second_value
        )

        ratio = treatment_value / comparator_value

        return {
            "claimed": f"~{claimed:g}x",
            "from_source": [treatment_value, comparator_value],
            "result": f"{ratio:.2f}x",
        }

    @classmethod
    def verify_batch(
        cls,
        batch: list[dict[str, Any]],
        llm_predictions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Verify a complete batch.

        Compatible with:
            ClaimVerifier.verify_batch(
                batch=batch,
                llm_predictions=llm_analyses,
            )

        Returns final predictions for OutputWriter.
        """
        if not isinstance(batch, list):
            raise TypeError("batch must be a list.")

        if not isinstance(llm_predictions, list):
            raise TypeError("llm_predictions must be a list.")

        analyses_by_id: dict[str, dict[str, Any]] = {}

        for analysis in llm_predictions:
            if not isinstance(analysis, dict):
                continue

            claim_id = analysis.get("claim_id")

            if not isinstance(claim_id, str) or not claim_id:
                continue

            if claim_id in analyses_by_id:
                raise ValueError(
                    f"Duplicate LLM analysis for claim_id: {claim_id}"
                )

            analyses_by_id[claim_id] = analysis

        results = []

        for item in batch:
            claim = item.get("claim", {})
            claim_id = claim.get("claim_id")
            claim_text = claim.get("text", "")
            evidence = item.get("evidence", [])

            if not isinstance(claim_id, str) or not claim_id:
                raise ValueError("A batch item is missing claim.claim_id.")

            if not isinstance(evidence, list):
                evidence = []

            analysis = analyses_by_id.get(claim_id)

            # Missing model output or evidence must never become supported.
            if (
                analysis is None
                or item.get("retrieval_error")
                or not evidence
            ):
                results.append({
                    "claim_id": claim_id,
                    "verdict": "cannot_verify",
                    "checks": {
                        name: "unclear" for name in REQUIRED_CHECKS
                    },
                    "evidence_page": None,
                    "evidence_span": None,
                    "computed": None,
                    "reason": (
                        "Missing LLM analysis or usable retrieved evidence."
                    ),
                })
                continue

            raw_checks = analysis.get("checks", {})
            if not isinstance(raw_checks, dict):
                raw_checks = {}

            validated = {
                name: cls._validate_check(
                    raw_checks.get(name),
                    evidence,
                )
                for name in REQUIRED_CHECKS
            }

            computed = cls._compute_figure(
                claim_text=claim_text,
                source_values=validated["figure"]["source_values"],
            )

            # If the claim makes a numerical multiplier assertion, failure
            # to calculate it safely means the figure cannot be verified.
            if (
                cls._claimed_multiplier(claim_text) is not None
                and computed is None
                and validated["figure"]["status"] == "match"
            ):
                validated["figure"]["status"] = "unclear"
                validated["figure"]["reason"] = (
                    "The numerical claim could not be independently "
                    "calculated from compatible, clearly labelled "
                    "source values."
                )

            statuses = [
                validated[name]["status"]
                for name in REQUIRED_CHECKS
            ]

            # The LLM supplies semantic judgments. Python enforces
            # evidence traceability and fails closed on uncertainty.
            if "unclear" in statuses:
                verdict = "cannot_verify"
            elif "mismatch" in statuses:
                verdict = "overstated"
            else:
                verdict = "supported"

            # Prefer the figure evidence, then another validated quote.
            primary = validated["figure"]

            if not primary["evidence_span"]:
                for name in REQUIRED_CHECKS:
                    if validated[name]["evidence_span"]:
                        primary = validated[name]
                        break

            reasons = [
                f"{name}: {validated[name]['reason']}"
                for name in REQUIRED_CHECKS
                if validated[name]["reason"]
            ]

            results.append({
                "claim_id": claim_id,
                "verdict": verdict,
                "checks": {
                    name: validated[name]["status"]
                    for name in REQUIRED_CHECKS
                },
                "evidence_page": primary["evidence_page"],
                "evidence_span": primary["evidence_span"],
                "computed": computed,
                "reason": " ".join(reasons) or (
                    "The LLM's checks were validated against retrieved "
                    "evidence."
                ),
            })

        return results
