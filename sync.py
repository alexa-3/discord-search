"""
sync.py — Discord message sync

Fetches all messages from every text channel in a Discord server and stores
them in messages.db with FTS5 full-text search and vector embeddings.

Features:
  - Resumes per-channel from last saved position (survives interruptions)
  - 60-second timeout per Discord API page fetch
  - Skips a channel after 3 consecutive timeouts
  - Progress indicator every 100 new messages

Usage:
    python sync.py
"""

import os
import sys
import ssl
import sqlite3
import asyncio
import aiohttp
import numpy as np
import certifi
import discord
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ssl_context = ssl.create_default_context(cafile=certifi.where())

load_dotenv()

BOT_TOKEN    = os.getenv("DISCORD_BOT_TOKEN")
GUILD_ID_RAW = os.getenv("DISCORD_GUILD_ID")
DB_PATH      = "messages.db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE     = 64    # messages per embedding batch
FETCH_TIMEOUT  = 30.0  # seconds to wait for one Discord API page
MAX_TIMEOUTS   = 3     # consecutive timeouts before skipping a channel
PROGRESS_EVERY = 100   # print a progress line every N new messages


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id           TEXT PRIMARY KEY,
            channel_name TEXT NOT NULL,
            author       TEXT NOT NULL,
            author_id    TEXT,
            timestamp    TEXT NOT NULL,
            content      TEXT NOT NULL,
            embedding    BLOB
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
        USING fts5(id UNINDEXED, content)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_progress (
            channel_name    TEXT PRIMARY KEY,
            last_message_id TEXT NOT NULL
        )
    """)
    # Index on channel_name speeds up the MAX(id) per-channel startup query
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_name)")
    # Migration: add author_id to existing databases that predate this column
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN author_id TEXT")
        print("Migration: added author_id column to messages table")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()


def get_existing_ids(conn: sqlite3.Connection) -> set:
    rows = conn.execute("SELECT id FROM messages").fetchall()
    return {row[0] for row in rows}


def insert_batch(conn: sqlite3.Connection, batch: list, embeddings) -> None:
    emb_blobs = [emb.astype(np.float32).tobytes() for emb in embeddings]
    conn.executemany(
        "INSERT INTO messages (id, channel_name, author, author_id, timestamp, content, embedding)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (m["id"], m["channel_name"], m["author"], m["author_id"],
             m["timestamp"], m["content"], blob)
            for m, blob in zip(batch, emb_blobs)
        ],
    )
    conn.executemany(
        "INSERT INTO messages_fts (id, content) VALUES (?, ?)",
        [(m["id"], m["content"]) for m in batch],
    )
    conn.commit()


def get_channel_progress(conn: sqlite3.Connection, channel_name: str) -> str | None:
    """Return the last saved message ID for this channel, or None."""
    row = conn.execute(
        "SELECT last_message_id FROM sync_progress WHERE channel_name = ?",
        (channel_name,),
    ).fetchone()
    return row[0] if row else None


def save_channel_progress(conn: sqlite3.Connection, channel_name: str, last_message_id: str) -> None:
    """Upsert the resume position for this channel."""
    conn.execute(
        "INSERT OR REPLACE INTO sync_progress (channel_name, last_message_id) VALUES (?, ?)",
        (channel_name, last_message_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Channel sync
# ---------------------------------------------------------------------------

async def sync_channel(
    channel: discord.TextChannel,
    existing_ids: set,
    conn: sqlite3.Connection,
    model: SentenceTransformer,
) -> None:
    # --- Diagnostic: confirm we entered the function before any I/O ---
    print(f"  #{channel.name}: entered sync_channel", flush=True)

    # Determine resume position.
    # Priority 1: saved sync_progress entry from a previous run.
    # Priority 2: MAX(id) already in the DB for this channel — avoids
    #             silently re-scanning millions of already-synced messages
    #             when the sync_progress table has no entry yet.
    after_id = get_channel_progress(conn, channel.name)
    if not after_id:
        print(f"  #{channel.name}: no progress entry — querying MAX(id) from DB...", flush=True)
        row = conn.execute(
            "SELECT MAX(id) FROM messages WHERE channel_name = ?", (channel.name,)
        ).fetchone()
        if row and row[0]:
            after_id = row[0]
            print(f"  #{channel.name}: seeding start position from DB max id {after_id}", flush=True)

    after = discord.Object(id=int(after_id)) if after_id else None
    if after_id:
        print(f"  #{channel.name}: fetching messages after {after_id}", flush=True)
    else:
        print(f"  #{channel.name}: no existing messages — fetching from start", flush=True)

    buffer: list = []
    new_count         = 0
    skipped_count     = 0
    consecutive_timeouts = 0
    last_seen_id      = after_id

    history = channel.history(limit=None, oldest_first=True, after=after)
    print(f"  #{channel.name}: history iterator created, starting fetch...", flush=True)

    while True:
        try:
            message = await asyncio.wait_for(history.__anext__(), timeout=FETCH_TIMEOUT)
            consecutive_timeouts = 0
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            consecutive_timeouts += 1
            print(
                f"\n  #{channel.name}: fetch timeout "
                f"({consecutive_timeouts}/{MAX_TIMEOUTS})...",
                flush=True,
            )
            if consecutive_timeouts >= MAX_TIMEOUTS:
                print(f"  #{channel.name}: {MAX_TIMEOUTS} consecutive timeouts — skipping", flush=True)
                if last_seen_id:
                    save_channel_progress(conn, channel.name, last_seen_id)
                return
            continue

        last_seen_id = str(message.id)
        content = message.content.strip()
        if not content:
            continue

        # When resuming (after_id is set), every message Discord returns is
        # guaranteed to be newer than our last saved position — no dupe check needed.
        if not after_id and last_seen_id in existing_ids:
            skipped_count += 1
            if skipped_count % PROGRESS_EVERY == 0:
                print(f"  #{channel.name}: skipped {skipped_count} already-synced...", end="\r", flush=True)
            continue

        buffer.append({
            "id":           last_seen_id,
            "channel_name": channel.name,
            "author":       str(message.author),
            "author_id":    str(message.author.id),
            "timestamp":    message.created_at.isoformat(),
            "content":      content,
        })
        existing_ids.add(last_seen_id)
        new_count += 1

        if new_count % PROGRESS_EVERY == 0:
            print(f"  #{channel.name}: {new_count} new messages...", end="\r", flush=True)

        if len(buffer) >= BATCH_SIZE:
            texts = [m["content"] for m in buffer]
            embeddings = model.encode(texts, show_progress_bar=False)
            insert_batch(conn, buffer, embeddings)
            save_channel_progress(conn, channel.name, last_seen_id)
            buffer = []

    # Flush any remaining messages
    if buffer:
        texts = [m["content"] for m in buffer]
        embeddings = model.encode(texts, show_progress_bar=False)
        insert_batch(conn, buffer, embeddings)
        if last_seen_id:
            save_channel_progress(conn, channel.name, last_seen_id)

    status = f"{new_count} new"
    if skipped_count:
        status += f", {skipped_count} already synced"
    if after_id:
        status += " (resumed)"
    print(f"  #{channel.name}: {status}" + " " * 40, flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    if not BOT_TOKEN:
        print("Error: DISCORD_BOT_TOKEN not set in .env")
        sys.exit(1)
    if not GUILD_ID_RAW:
        print("Error: DISCORD_GUILD_ID not set in .env")
        sys.exit(1)

    guild_id = int(GUILD_ID_RAW)

    print("Loading embedding model (downloads ~90 MB on first run)...")
    model = SentenceTransformer(EMBED_MODEL)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    existing_ids = get_existing_ids(conn)
    print(f"Database ready — {len(existing_ids):,} existing messages\n")

    intents = discord.Intents.default()
    intents.message_content = True
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    client = discord.Client(intents=intents, connector=connector)

    @client.event
    async def on_ready() -> None:
        print(f"Logged in as {client.user}\n")

        guild = client.get_guild(guild_id)
        if not guild:
            print(f"Error: Guild ID {guild_id} not found. Check DISCORD_GUILD_ID in .env")
            await client.close()
            return

        text_channels = [
            c for c in await guild.fetch_channels()
            if isinstance(c, discord.TextChannel)
        ]
        print(f"Server: '{guild.name}' — {len(text_channels)} text channels\n")

        for channel in text_channels:
            try:
                print(f"Syncing #{channel.name}...")
                await sync_channel(channel, existing_ids, conn, model)
            except discord.Forbidden:
                print(f"  #{channel.name}: no access, skipping")
            except Exception as exc:
                print(f"  #{channel.name}: error — {exc}")

        total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        print(f"\nSync complete! Total messages in database: {total:,}")
        conn.close()
        await client.close()

    await client.start(BOT_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
