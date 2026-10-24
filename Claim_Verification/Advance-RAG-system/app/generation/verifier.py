
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
    """Validate LLM evidence and build assignment-format predictions."""

    @staticmethod
    def _find_evidence(
        span: Any,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Find a retrieved chunk containing the quoted text."""

        if not isinstance(span, str) or not span.strip():
            return None

        # Exact substring match; no fuzzy quote validation.
        for item in evidence:
            text = item.get("text", "")
            if isinstance(text, str) and span in text:
                return item

        return None

    @classmethod
    def _validate_check(
        cls,
        check: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate a check's status, quotation, and page metadata."""

        status = check.get("status", "unclear")
        span = check.get("evidence_span")
        page = check.get("evidence_page")

        if status not in VALID_STATUSES:
            status = "unclear"

        source = cls._find_evidence(span, evidence)

        # A match/mismatch needs a traceable quotation.
        if source is None:
            return {
                "status": "unclear",
                "evidence_page": None,
                "evidence_span": None,
                "source_values": None,
                "reason": "The quoted evidence could not be validated.",
            }

        metadata = source.get("metadata") or {}
        page_numbers = metadata.get("page_numbers") or []
        metadata_page = metadata.get("evidence_page")

        valid_pages = set(page_numbers)
        if metadata_page is not None:
            valid_pages.add(metadata_page)

        if page is not None and valid_pages and page not in valid_pages:
            return {
                "status": "unclear",
                "evidence_page": None,
                "evidence_span": None,
                "source_values": None,
                "reason": "The cited page conflicts with evidence metadata.",
            }

        if page is None:
            if metadata_page is not None:
                page = metadata_page
            elif len(page_numbers) == 1:
                page = page_numbers[0]

        # If the source has page metadata, do not accept an unlisted page.
        if page is not None and valid_pages and page not in valid_pages:
            page = None

        return {
            "status": status,
            "evidence_page": page,
            "evidence_span": span,
            "source_values": check.get("source_values"),
            "reason": str(
                check.get("reason")
                or "No explanation was supplied by the LLM."
            ),
        }

    @staticmethod
    def _extract_claimed_multiplier(claim_text: str) -> float | None:
        """Extract claims such as '2x', '~2×', or '1.5-fold'."""

        patterns = (
            r"~?\s*(\d+(?:\.\d+)?)\s*[xX×]\b",
            r"~?\s*(\d+(?:\.\d+)?)\s*[- ]?fold\b",
        )

        for pattern in patterns:
            match = re.search(pattern, claim_text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None

    @staticmethod
    def _compute_figure(
        claim: dict[str, Any],
        figure_check: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Calculate a ratio from exactly two positive source values.

        This calculation is valid only when the LLM has extracted the
        two values in the intended numerator/denominator order from
        validated evidence. It does not independently prove their meaning.
        """

        values = figure_check.get("source_values")

        if not isinstance(values, list) or len(values) != 2:
            return None

        # Booleans are integers in Python; explicitly reject them.
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in values
        ):
            return None

        numerator, denominator = values

        if (
            numerator <= 0
            or denominator <= 0
        ):
            return None

        claimed_multiplier = ClaimVerifier._extract_claimed_multiplier(
            claim.get("text", "")
        )

        if claimed_multiplier is None:
            return None

        result = numerator / denominator

        return {
            "claimed": f"~{claimed_multiplier:g}x",
            "from_source": [numerator, denominator],
            "result": f"{result:.2f}x",
        }

    @staticmethod
    def _format_reason(
        verdict: str,
        checks: dict[str, str],
        computed: dict[str, Any] | None,
    ) -> str:
        """Create a concise reviewer-facing explanation."""

        if verdict == "supported":
            return (
                "All four checks passed evidence validation, and any "
                "applicable numerical comparison was calculated."
            )

        mismatches = [
            name for name, status in checks.items()
            if status == "mismatch"
        ]

        if verdict == "overstated":
            details = ", ".join(mismatches)
            if computed:
                return (
                    f"The evidence indicates that the claim exceeds "
                    f"the source result ({computed['result']} versus "
                    f"{computed['claimed']}); check(s) marked mismatch: "
                    f"{details}."
                )
            return (
                f"Validated evidence indicates that the claim exceeds "
                f"the source; check(s) marked mismatch: {details}."
            )

        return (
            "The claim could not be verified reliably because evidence, "
            "page attribution, or a required comparison was insufficient "
            "or ambiguous."
        )

    @classmethod
    def verify_batch(
        cls,
        batch: list[dict[str, Any]],
        llm_predictions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return predictions using the assignment's output schema."""

        predictions_by_id = {
            item.get("claim_id"): item
            for item in llm_predictions
            if isinstance(item, dict)
        }

        final_predictions = []

        for item in batch:
            claim = item["claim"]
            claim_id = claim["claim_id"]
            evidence = item.get("evidence", [])
            llm_result = predictions_by_id.get(claim_id)

            # Safe default: no claim is supported without analysis/evidence.
            if (
                item.get("retrieval_error")
                or not evidence
                or llm_result is None
            ):
                final_predictions.append({
                    "claim_id": claim_id,
                    "verdict": "cannot_verify",
                    "checks": {
                        name: "unclear" for name in REQUIRED_CHECKS
                    },
                    "evidence_page": None,
                    "evidence_span": None,
                    "computed": None,
                    "reason": (
                        "Evidence retrieval or LLM analysis failed; "
                        "the claim cannot be verified."
                    ),
                })
                continue

            raw_checks = llm_result.get("checks", {})
            if not isinstance(raw_checks, dict):
                raw_checks = {}

            validated = {}

            for name in REQUIRED_CHECKS:
                raw_check = raw_checks.get(name, {})
                if not isinstance(raw_check, dict):
                    raw_check = {}

                validated[name] = cls._validate_check(
                    raw_check,
                    evidence,
                )

            # The public schema requires simple status strings.
            checks = {
                name: validated[name]["status"]
                for name in REQUIRED_CHECKS
            }

            # Use the figure check's validated quote as primary evidence.
            primary = validated["figure"]

            if not primary["evidence_span"]:
                # Fall back to another validated check's quotation.
                primary = next(
                    (
                        validated[name]
                        for name in REQUIRED_CHECKS
                        if validated[name]["evidence_span"]
                    ),
                    {
                        "evidence_page": None,
                        "evidence_span": None,
                    },
                )

            computed = cls._compute_figure(
                claim,
                validated["figure"],
            )

            # A figure mismatch must not be accepted if a required
            # numerical calculation could not be performed.
            if (
                checks["figure"] != "unclear"
                and cls._extract_claimed_multiplier(claim.get("text", ""))
                is not None
                and computed is None
            ):
                checks["figure"] = "unclear"

            if any(status == "unclear" for status in checks.values()):
                verdict = "cannot_verify"
            elif any(status == "mismatch" for status in checks.values()):
                verdict = "overstated"
            else:
                verdict = "supported"

            # When a numeric comparison is available, ensure it agrees
            # with the claimed multiplier before allowing "supported".
            if computed is not None:
                claimed = cls._extract_claimed_multiplier(claim["text"])
                source_result = float(computed["result"].removesuffix("x"))

                # A 20% relative tolerance is an explicit provisional
                # rule for approximate multiplier claims.
                if abs(source_result - claimed) / claimed > 0.20:
                    checks["figure"] = "mismatch"
                    verdict = "overstated"

            reason = cls._format_reason(verdict, checks, computed)

            final_predictions.append({
                "claim_id": claim_id,
                "verdict": verdict,
                "checks": checks,
                "evidence_page": primary["evidence_page"],
                "evidence_span": primary["evidence_span"],
                "computed": computed,
                "reason": reason,
            })

        return final_predictions
