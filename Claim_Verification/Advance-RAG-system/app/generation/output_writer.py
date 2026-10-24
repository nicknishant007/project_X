
import json
from pathlib import Path
from typing import Any


class OutputWriter:

    @staticmethod
    def write_predictions(
        predictions: list[dict[str, Any]],
        output_path: str | Path = "predictions.json",
    ) -> None:
        """Write predictions to a JSON file."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                {"predictions": predictions},
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Saved {len(predictions)} predictions to {path.resolve()}"
        )
