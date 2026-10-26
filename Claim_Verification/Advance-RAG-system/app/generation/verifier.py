import re
from typing import Any

REQUIRED_CHECKS = ("figure", "population", "drug_indication", "actually_stated")
VALID_STATUSES = {"match", "mismatch", "unclear"}

# "~2x" is accepted when the Python-computed ratio is within +/-20% of the claim.
MULTIPLIER_RELATIVE_TOLERANCE = 0.20

_NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|\.\d+")
# Numbers in a claim that must be traceable to the source: decimals and percentages.
_CLAIM_NUM_RE = re.compile(r"(?<![\d.])(?:\d+\.\d+|\d+(?=\s*%))")


class ClaimVerifier:
    """Deterministic validation of LLM checks, evidence traceability and arithmetic."""

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _normalise_text(value: Any) -> str:
        return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""

    @staticmethod
    def _numbers(text: str) -> set[float]:
        out = set()
        for token in _NUM_RE.findall(text or ""):
            try:
                out.add(round(abs(float(token.replace(",", ""))), 6))
            except ValueError:
                pass
        return out

    @classmethod
    def _claim_numbers(cls, claim_text: str) -> set[float]:
        return {round(float(t), 6) for t in _CLAIM_NUM_RE.findall(claim_text or "")}

    @classmethod
    def _page_number(cls, evidence_item: dict[str, Any], quote: str | None = None) -> int | None:
        metadata = evidence_item.get("metadata") or {}

        # Best: the quote sits inside a single original block with a single page.
        nq = cls._normalise_text(quote)
        if nq:
            for block in metadata.get("block_pages") or []:
                if not isinstance(block, dict):
                    continue
                if nq in cls._normalise_text(block.get("text")):
                    pages = {p for p in (block.get("pages") or []) if isinstance(p, int)}
                    if len(pages) == 1:
                        return pages.pop()

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

    @classmethod
    def _source_values_supported(cls, source_values: Any, quote: str) -> bool:
        """Every reported numeric `value` must occur in the quoted span."""
        if source_values is None:
            return True
        quote_numbers = cls._numbers(quote)
        items = source_values if isinstance(source_values, list) else [source_values]
        for entry in items:
            if not isinstance(entry, dict):
                return False
            value = entry.get("value")
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if round(abs(float(value)), 6) not in quote_numbers:
                return False
        return True

    # -------------------------------------------------------- single check
    @classmethod
    def _validate_check(cls, raw_check: Any, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        unclear = {"status": "unclear", "evidence_page": None, "evidence_span": None,
                   "source_values": None, "reason": ""}
        if not isinstance(raw_check, dict):
            return {**unclear, "reason": "Missing or invalid check object from the LLM."}
        status = raw_check.get("status")
        if status not in VALID_STATUSES:
            return {**unclear, "reason": "The LLM returned an invalid check status."}
        reason = raw_check.get("reason")
        reason = reason.strip() if isinstance(reason, str) else ""

        span = raw_check.get("evidence_span")
        item = cls._find_evidence(span, evidence)

        if status == "unclear":
            out = {**unclear, "reason": reason or "Evidence is missing, ambiguous or insufficient."}
            if item is not None:
                out["evidence_span"] = span
                out["evidence_page"] = cls._page_number(item, span)
            return out

        if item is None:
            return {**unclear, "reason": (
                f"LLM reported '{status}' but its quotation is missing or not found "
                "verbatim in the retrieved evidence, so the check is not accepted.")}

        source_values = raw_check.get("source_values")
        if not isinstance(source_values, (list, dict, type(None))):
            source_values = None
        if not cls._source_values_supported(source_values, span):
            return {**unclear, "reason":
                    "Numeric source_values are not present in the quoted evidence."}

        return {"status": status, "evidence_page": cls._page_number(item, span),
                "evidence_span": span, "source_values": source_values, "reason": reason}

    # ----------------------------------------------------------- arithmetic
    @staticmethod
    def _claimed_multiplier(claim_text: str) -> float | None:
        if not isinstance(claim_text, str):
            return None
        patterns = (
            r"~\s*(\d+(?:\.\d+)?)\s*[x\u00d7]",
            r"(\d+(?:\.\d+)?)\s*[- ]?fold\b",
            r"(\d+(?:\.\d+)?)\s*[x\u00d7](?![a-z])",
            r"(\d+(?:\.\d+)?)\s*times\b",
        )
        for pattern in patterns:
            match = re.search(pattern, claim_text, re.IGNORECASE)
            if match and float(match.group(1)) > 0:
                return float(match.group(1))
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
        if any(w in label for w in ("comparator", "control", "placebo", "comparison")):
            return "comparator"
        if any(w in label for w in ("treatment", "intervention", "experimental")):
            return "treatment"
        return None

    @classmethod
    def _compute_figure(cls, claim_text: str, source_values: Any) -> dict[str, Any] | None:
        claimed = cls._claimed_multiplier(claim_text)
        if claimed is None or not isinstance(source_values, list) or len(source_values) != 2:
            return None
        first, second = source_values
        a, b = cls._numeric_value(first), cls._numeric_value(second)
        if a is None or b is None or a == 0 or b == 0 or (a > 0) != (b > 0):
            return None
        for key in ("measure", "unit", "timepoint"):
            # units such as "%" normalise to "" -> fall back to the raw string
            la = cls._label(first.get(key)) or str(first.get(key) or "").strip()
            lb = cls._label(second.get(key)) or str(second.get(key) or "").strip()
            if not la or la != lb:
                return None
        ra, rb = cls._arm_role(first.get("arm")), cls._arm_role(second.get("arm"))
        if {ra, rb} != {"treatment", "comparator"}:
            return None
        treatment = abs(a) if ra == "treatment" else abs(b)
        comparator = abs(a) if ra == "comparator" else abs(b)
        ratio = treatment / comparator
        error = abs(ratio - claimed) / claimed
        return {
            "claimed": f"~{claimed:g}x",
            "from_source": [treatment, comparator],
            "result": f"{ratio:.2f}x",
            "matches_claim": error <= MULTIPLIER_RELATIVE_TOLERANCE,
            "relative_error": round(error, 4),
            "tolerance": MULTIPLIER_RELATIVE_TOLERANCE,
        }

    # ----------------------------------------------------------- main entry
    @classmethod
    def verify_batch(cls, batch: list[dict[str, Any]], llm_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(batch, list):
            raise TypeError("batch must be a list.")
        if not isinstance(llm_predictions, list):
            raise TypeError("llm_predictions must be a list.")

        analyses: dict[str, dict[str, Any]] = {}
        for analysis in llm_predictions:
            if isinstance(analysis, dict) and isinstance(analysis.get("claim_id"), str):
                analyses.setdefault(analysis["claim_id"], analysis)

        results = []
        for item in batch:
            claim = item.get("claim", {})
            claim_id, claim_text = claim.get("claim_id"), claim.get("text", "")
            evidence = item.get("evidence", [])
            if not isinstance(claim_id, str) or not claim_id:
                raise ValueError("A batch item is missing claim.claim_id.")
            if not isinstance(evidence, list):
                evidence = []

            analysis = analyses.get(claim_id)
            if analysis is None or item.get("retrieval_error") or not evidence:
                why = item.get("retrieval_error_message") or "Missing LLM analysis or usable retrieved evidence."
                results.append({
                    "claim_id": claim_id, "verdict": "cannot_verify",
                    "checks": {n: "unclear" for n in REQUIRED_CHECKS},
                    "evidence_page": None, "evidence_span": None, "computed": None,
                    "reason": why,
                })
                continue

            raw_checks = analysis.get("checks")
            raw_checks = raw_checks if isinstance(raw_checks, dict) else {}
            validated = {n: cls._validate_check(raw_checks.get(n), evidence) for n in REQUIRED_CHECKS}
            figure = validated["figure"]

            # Every decimal / percentage in the claim must exist somewhere in the evidence.
            claim_numbers = cls._claim_numbers(claim_text)
            if claim_numbers and figure["status"] == "match":
                evidence_numbers = set()
                for ev in evidence:
                    evidence_numbers |= cls._numbers(ev.get("text", ""))
                missing = sorted(claim_numbers - evidence_numbers)
                if missing:
                    figure["status"] = "unclear"
                    figure["reason"] = (
                        "Claim numbers not found in the retrieved evidence: "
                        + ", ".join(f"{m:g}" for m in missing) + "."
                    )

            computed = cls._compute_figure(claim_text, figure["source_values"])
            if cls._claimed_multiplier(claim_text) is not None:
                if computed is None:
                    if figure["status"] == "match":
                        figure["status"] = "unclear"
                        figure["reason"] = ("The multiplier could not be independently calculated "
                                            "from traceable, compatible source values.")
                elif not computed["matches_claim"]:
                    figure["status"] = "mismatch"
                    figure["reason"] = (
                        f"Python calculated {computed['result']} from the quoted source values, "
                        f"outside the +/-{MULTIPLIER_RELATIVE_TOLERANCE:.0%} tolerance for "
                        f"{computed['claimed']}.")
                elif figure["status"] == "unclear":
                    figure["reason"] = ("Ratio is consistent with the claim but the figure check "
                                        "was not semantically established.")

            statuses = [validated[n]["status"] for n in REQUIRED_CHECKS]
            if "mismatch" in statuses:
                verdict = "overstated"
            elif "unclear" in statuses:
                verdict = "cannot_verify"
            else:
                verdict = "supported"

            # Pick the most informative evidence for the top level.
            def pick():
                if verdict == "overstated":
                    for n in REQUIRED_CHECKS:
                        if validated[n]["status"] == "mismatch" and validated[n]["evidence_span"]:
                            return validated[n]
                for n in REQUIRED_CHECKS:
                    if validated[n]["evidence_span"]:
                        return validated[n]
                return figure
            primary = pick()

            flagged = [n for n in REQUIRED_CHECKS if validated[n]["status"] != "match"] or list(REQUIRED_CHECKS)
            reasons = [f"{n}: {validated[n]['reason']}" for n in flagged if validated[n]["reason"]]

            results.append({
                "claim_id": claim_id, "verdict": verdict,
                "checks": {n: validated[n]["status"] for n in REQUIRED_CHECKS},
                "evidence_page": primary["evidence_page"],
                "evidence_span": primary["evidence_span"],
                "computed": computed,
                "reason": " ".join(reasons) or "All checks were validated against retrieved evidence.",
            })
        return results
