import json
from pathlib import Path
from typing import Any

# Only these are mandatory. population / molecule are optional so that
# differently-shaped claim files still run.
REQUIRED_FIELDS = {"claim_id", "text", "cited_paper"}
OPTIONAL_STRING_FIELDS = ("population", "molecule")


def parse_claims(data: Any) -> list[dict[str, Any]]:
    """Validate already-parsed JSON (list of claims, or {"claims": [...]})."""
    claims = data.get("claims") if isinstance(data, dict) else data

    if not isinstance(claims, list):
        raise ValueError(
            "Expected a JSON array or an object containing a 'claims' array."
        )

    validated, seen = [], set()

    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(f"Claim at index {index} must be a JSON object.")

        missing = REQUIRED_FIELDS - claim.keys()
        if missing:
            raise ValueError(
                f"Claim at index {index} is missing fields: {sorted(missing)}"
            )

        for field in REQUIRED_FIELDS:
            if not isinstance(claim[field], str) or not claim[field].strip():
                raise ValueError(
                    f"Claim at index {index}: '{field}' must be a non-empty string."
                )

        item = dict(claim)  # keep extra fields
        for field in OPTIONAL_STRING_FIELDS:
            value = item.get(field)
            item[field] = value.strip() if isinstance(value, str) else ""

        item["claim_id"] = item["claim_id"].strip()
        if item["claim_id"] in seen:
            raise ValueError(f"Duplicate claim_id: {item['claim_id']}")
        seen.add(item["claim_id"])
        validated.append(item)

    return validated


def extract_claims(file_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate claims from a JSON file."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError("Input file must have a .json extension.")

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    return parse_claims(data)
