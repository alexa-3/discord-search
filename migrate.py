"""
migrate.py — one-time database migrations

Safe to run multiple times — each migration is skipped if already applied.

Usage:
    python3.14 migrate.py
"""

import sqlite3

DB_PATH = "messages.db"


def migrate(conn: sqlite3.Connection) -> None:
    # Migration 1: add author_id column
    # Existing rows will have NULL for author_id; new syncs will populate it.
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN author_id TEXT")
        conn.commit()
        print("✓ Added author_id column to messages table")
    except sqlite3.OperationalError:
        print("- author_id column already exists, skipping")


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    except sqlite3.OperationalError:
        print("Error: messages table not found. Run sync.py first.")
        conn.close()
        return

    print(f"Database: {count:,} messages\n")
    migrate(conn)
    print("\nMigration complete.")
    conn.close()


if __name__ == "__main__":
    main()
