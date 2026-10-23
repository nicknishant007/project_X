import sqlite3

DB_PATH = "storage/registry.db"


def get_connection():

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False
    )

    conn.execute(
        "PRAGMA journal_mode=WAL;"
    )

    return conn


#What and Why we did this 
#DATABASE LOCKING ISSUE:
"""Database locking is a mechanism used by databases to maintain data consistency and 
prevent corruption when multiple processes or threads try to access the same database
 simultaneously. In your ingestion pipeline, multiple parallel workers were processing
documents at the same time, and each worker attempted to update the SQLite registry 
database with processed file information. Since SQLite allows only limited concurrent 
write operations, when one worker started writing to the database, SQLite temporarily 
locked the database file. If another worker tried to write during that time, SQLite 
raised a database is locked error to prevent conflicting writes and protect the database from corruption."""
"""Worker 1
   ↓
Writing to registry.db
   ↓
SQLite locks database

Worker 2
   ↓
Tries writing simultaneously
   ↓
Database locked error"""



"""Your ingestion pipeline uses multiple parallel workers to process files efficiently,
which means several threads may try to access the same SQLite database (registry.db) 
at the same time. Initially, the database connection was created using the default 
SQLite configuration, but SQLite is designed primarily for lightweight single-process 
usage. When one worker thread was writing processed file information into the registry 
and another thread attempted to write simultaneously, SQLite locked the database to
maintain consistency, resulting in the database is locked error. To solve this, we
updated the database connection with timeout=30, check_same_thread=False, and 
enabled WAL (Write Ahead Logging) mode. The timeout=30 allows a thread to wait 
for the database to become available instead of failing immediately, while 
check_same_thread=False enables safe multi-threaded access from parallel ingestion workers.

We also enabled WAL mode using PRAGMA journal_mode=WAL;, which changes how
SQLite handles writes internally. Instead of directly modifying the database 
file during every write operation, SQLite first stores changes inside a temporary 
WAL log and later merges them into the main database. This significantly improves
concurrent read/write performance and reduces locking conflicts. The overall workflow
now becomes: parallel ingestion workers process files → workers attempt to update 
the registry database → if one worker is already writing, others wait instead of
crashing → writes are temporarily stored in the WAL log → SQLite safely merges changes 
into the main database. This makes the ingestion pipeline far more stable, scalable, and 
suitable for parallel document processing."""