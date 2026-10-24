import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "claim_id",
    "text",
    "population",
    "molecule",
    "cited_paper",
}


def extract_claims(file_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate claims from a JSON file."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() != ".json":
        raise ValueError("Input file must have a .json extension.")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    # Accept either a list of claims or {"claims": [...]}.
    if isinstance(data, dict):
        claims = data.get("claims")
    else:
        claims = data

    if not isinstance(claims, list):
        raise ValueError(
            "Expected a JSON array or an object containing a 'claims' array."
        )

    validated_claims = []
    seen_ids = set()

    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(f"Claim at index {index} must be a JSON object.")

        missing_fields = REQUIRED_FIELDS - claim.keys()

        if missing_fields:
            raise ValueError(
                f"Claim at index {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        for field in REQUIRED_FIELDS:
            value = claim[field]

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Claim at index {index}: '{field}' must be a "
                    "non-empty string."
                )

        claim_id = claim["claim_id"].strip()

        if claim_id in seen_ids:
            raise ValueError(f"Duplicate claim_id: {claim_id}")

        seen_ids.add(claim_id)

        # Preserve extra input fields, if any.
        validated_claim = dict(claim)
        validated_claim["claim_id"] = claim_id
        validated_claims.append(validated_claim)

    return validated_claims
