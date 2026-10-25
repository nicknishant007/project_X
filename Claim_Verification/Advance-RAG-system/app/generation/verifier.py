import re
from typing import Any

REQUIRED_CHECKS = ("figure", "population", "drug_indication", "actually_stated")
VALID_STATUSES = {"match", "mismatch", "unclear"}

# For claims written as "~2x", accept ratios within ±20% of the claimed
# multiplier. This is a policy choice; tighten/loosen only with an explicit
# evaluation rationale. A ratio of 1.22 is therefore not a match for ~2x.
MULTIPLIER_RELATIVE_TOLERANCE = 0.20


class ClaimVerifier:
    """Deterministic validation of LLM checks, evidence traceability and arithmetic."""

    @staticmethod
    def _normalise_text(value: Any) -> str:
        return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""

    @staticmethod
    def _page_number(evidence_item: dict[str, Any]) -> int | None:
        metadata = evidence_item.get("metadata") or {}
        for key in ("evidence_page", "page_number", "page_no", "page"):
            value = metadata.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                number = int(value)
                if key == "page":
                    return number + 1 if number >= 0 else None
                return number if number > 0 else None
        for key in ("page_numbers", "pages"):
            pages = metadata.get(key)
            if isinstance(pages, list) and len(pages) == 1:
                value = pages[0]
                if isinstance(value, int) and value > 0:
                    return value
                if isinstance(value, str) and value.isdigit() and int(value) > 0:
                    return int(value)
        return None

    @classmethod
    def _find_evidence(cls, evidence_span: Any, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not isinstance(evidence_span, str) or not evidence_span.strip():
            return None
        quote = cls._normalise_text(evidence_span)
        for item in evidence:
            source_text = item.get("text", "")
            if isinstance(source_text, str) and quote in cls._normalise_text(source_text):
                return item
        return None

    @staticmethod
    def _number_occurs_in_text(value: int | float, text: str) -> bool:
        """Require every asserted numeric value to occur in the quoted source."""
        numeric = float(value)
        candidates = {str(value), str(int(value)) if numeric.is_integer() else "", f"{numeric:g}"}
        candidates.discard("")
        # Match as a numeric token; permit commas in thousands and optional %.
        for candidate in candidates:
            pattern = rf"(?<![\w.]){re.escape(candidate)}(?![\w.])"
            if re.search(pattern, text):
                return True
        return False

    @classmethod
    def _source_values_supported(cls, source_values: Any, quote: str) -> bool:
        """Reject structured source values whose numeric assertions aren't quoted."""
        if source_values is None:
            return True
        if isinstance(source_values, dict):
            for key, value in source_values.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    if not cls._number_occurs_in_text(value, quote):
                        return False
                elif isinstance(value, (dict, list)):
                    if not cls._source_values_supported(value, quote):
                        return False
            return True
        if isinstance(source_values, list):
            return all(cls._source_values_supported(value, quote) for value in source_values)
        return True

    @classmethod
    def _validate_check(cls, raw_check: Any, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        unclear = {
            "status": "unclear", "evidence_page": None, "evidence_span": None,
            "source_values": None, "reason": ""
        }
        if not isinstance(raw_check, dict):
            return {**unclear, "reason": "Missing or invalid check object from the LLM."}
        status = raw_check.get("status")
        if status not in VALID_STATUSES:
            return {**unclear, "reason": "The LLM returned an invalid check status."}
        reason = raw_check.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        span = raw_check.get("evidence_span")
        evidence_item = cls._find_evidence(span, evidence)
        if evidence_item is None:
            return {**unclear, "reason": "The supporting quotation is missing or could not be located in retrieved evidence."}

        source_values = raw_check.get("source_values")
        if not isinstance(source_values, (list, dict, type(None))):
            source_values = None
        if not cls._source_values_supported(source_values, span):
            return {
                **unclear,
                "reason": "One or more numeric source_values are not present in the quoted evidence.",
            }
        return {
            "status": status,
            "evidence_page": cls._page_number(evidence_item),
            "evidence_span": span,
            "source_values": source_values,
            "reason": reason,
        }

    @staticmethod
    def _claimed_multiplier(claim_text: str) -> float | None:
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
        if not isinstance(item, dict):
            return None
        value = item.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _label(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip() if isinstance(value, str) else ""

    @classmethod
    def _arm_role(cls, value: Any) -> str | None:
        label = cls._label(value)
        treatment = {"treatment arm", "treatment group", "intervention arm",
                     "intervention group", "experimental arm", "experimental group"}
        comparator = {"comparator arm", "comparator group", "control arm",
                      "control group", "comparison arm", "comparison group",
                      "placebo arm", "placebo group"}
        if label in treatment:
            return "treatment"
        if label in comparator:
            return "comparator"
        return None

    @classmethod
    def _compute_figure(cls, claim_text: str, source_values: Any) -> dict[str, Any] | None:
        claimed = cls._claimed_multiplier(claim_text)
        if claimed is None or not isinstance(source_values, list) or len(source_values) != 2:
            return None
        first, second = source_values
        first_value, second_value = cls._numeric_value(first), cls._numeric_value(second)
        if first_value is None or second_value is None or first_value <= 0 or second_value <= 0:
            return None
        for key in ("measure", "unit", "timepoint"):
            first_label, second_label = cls._label(first.get(key)), cls._label(second.get(key))
            if not first_label or first_label != second_label:
                return None
        first_role, second_role = cls._arm_role(first.get("arm")), cls._arm_role(second.get("arm"))
        if {first_role, second_role} != {"treatment", "comparator"}:
            return None
        treatment_value = first_value if first_role == "treatment" else second_value
        comparator_value = first_value if first_role == "comparator" else second_value
        ratio = treatment_value / comparator_value
        relative_error = abs(ratio - claimed) / claimed
        matches_claim = relative_error <= MULTIPLIER_RELATIVE_TOLERANCE
        return {
            "claimed": f"~{claimed:g}x",
            "from_source": [treatment_value, comparator_value],
            "result": f"{ratio:.2f}x",
            "matches_claim": matches_claim,
            "relative_error": round(relative_error, 4),
            "tolerance": MULTIPLIER_RELATIVE_TOLERANCE,
        }

    @classmethod
    def verify_batch(cls, batch: list[dict[str, Any]], llm_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                raise ValueError(f"Duplicate LLM analysis for claim_id: {claim_id}")
            analyses_by_id[claim_id] = analysis

        results = []
        for item in batch:
            claim = item.get("claim", {})
            claim_id, claim_text = claim.get("claim_id"), claim.get("text", "")
            evidence = item.get("evidence", [])
            if not isinstance(claim_id, str) or not claim_id:
                raise ValueError("A batch item is missing claim.claim_id.")
            if not isinstance(evidence, list):
                evidence = []
            analysis = analyses_by_id.get(claim_id)
            if analysis is None or item.get("retrieval_error") or not evidence:
                results.append({
                    "claim_id": claim_id, "verdict": "cannot_verify",
                    "checks": {name: "unclear" for name in REQUIRED_CHECKS},
                    "evidence_page": None, "evidence_span": None, "computed": None,
                    "reason": "Missing LLM analysis or usable retrieved evidence.",
                })
                continue

            raw_checks = analysis.get("checks", {})
            if not isinstance(raw_checks, dict):
                raw_checks = {}
            validated = {name: cls._validate_check(raw_checks.get(name), evidence)
                         for name in REQUIRED_CHECKS}

            computed = cls._compute_figure(claim_text, validated["figure"]["source_values"])
            if cls._claimed_multiplier(claim_text) is not None:
                figure = validated["figure"]
                if computed is None:
                    if figure["status"] == "match":
                        figure["status"] = "unclear"
                        figure["reason"] = (
                            "The multiplier could not be independently calculated from "
                            "traceable, compatible source values."
                        )
                elif not computed["matches_claim"]:
                    # Deterministic numeric contradiction overrides an LLM 'match'.
                    figure["status"] = "mismatch"
                    figure["reason"] = (
                        f"Python calculated {computed['result']} from the quoted source "
                        f"values, which is outside the ±{MULTIPLIER_RELATIVE_TOLERANCE:.0%} "
                        f"tolerance for {computed['claimed']}."
                    )
                elif figure["status"] == "unclear":
                    # Arithmetic cannot establish semantic meaning by itself.
                    figure["reason"] = (
                        "The ratio is numerically consistent with the claimed multiplier, "
                        "but the figure check was not semantically established."
                    )

            statuses = [validated[name]["status"] for name in REQUIRED_CHECKS]
            verdict = "cannot_verify" if "unclear" in statuses else (
                "overstated" if "mismatch" in statuses else "supported"
            )
            primary = validated["figure"]
            if not primary["evidence_span"]:
                primary = next((validated[name] for name in REQUIRED_CHECKS
                                if validated[name]["evidence_span"]), primary)
            reasons = [f"{name}: {validated[name]['reason']}" for name in REQUIRED_CHECKS
                       if validated[name]["reason"]]
            results.append({
                "claim_id": claim_id, "verdict": verdict,
                "checks": {name: validated[name]["status"] for name in REQUIRED_CHECKS},
                "evidence_page": primary["evidence_page"],
                "evidence_span": primary["evidence_span"],
                "computed": computed,
                "reason": " ".join(reasons) or "Checks were validated against retrieved evidence.",
            })
        return results
