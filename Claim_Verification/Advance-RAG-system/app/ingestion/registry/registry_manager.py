from app.ingestion.registry.db import get_connection

class RegistryManager:

    @staticmethod
    def initialize():

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                file_hash TEXT,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    @staticmethod
    def is_processed(file_hash:str)-> bool:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM file_registry WHERE file_hash=?',(file_hash,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    @staticmethod
    def mark_processed(file_path:str,file_hash:str):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''INSERT OR IGNORE INTO file_registry (file_path,file_hash) VALUES (?,?)''',(file_path,file_hash))
        conn.commit()
        conn.close()
#Read about it how it works and how to use it in the main.py file
"""File Hash
    ↓
Check registry
    ↓
Already processed?
    ↓
YES → Skip
NO  → Process + Save"""