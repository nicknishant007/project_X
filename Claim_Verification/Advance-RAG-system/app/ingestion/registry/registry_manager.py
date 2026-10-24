
from app.ingestion.registry.db import get_connection


class RegistryManager:

    @staticmethod
    def initialize() -> None:
        """Create the file registry table if it doesn't exist."""

        conn = get_connection()

        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE,
                    file_hash TEXT NOT NULL,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

        finally:
            conn.close()

    @staticmethod
    def is_processed(file_hash: str) -> bool:
        """Check whether a file hash already exists in the registry."""

        # Defensive initialization prevents a missing-table error.
        RegistryManager.initialize()

        conn = get_connection()

        try:
            result = conn.execute(
                """
                SELECT id
                FROM file_registry
                WHERE file_hash = ?
                LIMIT 1
                """,
                (file_hash,),
            ).fetchone()

            return result is not None

        finally:
            conn.close()

    @staticmethod
    def mark_processed(file_path: str, file_hash: str) -> None:
        """Save a successfully processed file in the registry."""

        RegistryManager.initialize()

        conn = get_connection()

        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO file_registry
                    (file_path, file_hash)
                VALUES (?, ?)
                """,
                (str(file_path), file_hash),
            )

            conn.commit()

        finally:
            conn.close()
