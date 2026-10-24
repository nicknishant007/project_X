
import os
from pathlib import Path

import yaml


# Project root: app/ingestion/scanner/local_scanner.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ingestion.yaml"


def load_config():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Ingestion config not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("ingestion.yaml must contain a YAML mapping.")

    if not config.get("data_dir"):
        raise ValueError("'data_dir' is missing from ingestion.yaml.")

    if not config.get("supported_extensions"):
        raise ValueError(
            "'supported_extensions' is missing or empty in ingestion.yaml."
        )

    return config


_CONFIG = load_config()

DATA_DIR = Path(_CONFIG["data_dir"])
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_ROOT / DATA_DIR

SUPPORTED_FORMATS = {
    ext.lower() if ext.startswith(".") else f".{ext.lower()}"
    for ext in _CONFIG["supported_extensions"]
}


class LocalScanner:
    @staticmethod
    def scan():
        """Yield absolute paths for supported files, recursively."""

        if not DATA_DIR.is_dir():
            raise FileNotFoundError(
                f"Data directory does not exist: {DATA_DIR}"
            )

        for root, dirs, files in os.walk(DATA_DIR):
            # Stable traversal order makes runs reproducible.
            dirs.sort()

            for file_name in sorted(files):
                extension = Path(file_name).suffix.lower()

                if extension in SUPPORTED_FORMATS:
                    yield str(Path(root) / file_name)
