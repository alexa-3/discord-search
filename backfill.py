"""
backfill.py — fill in author_id for messages synced before that column existed

For each channel that has rows with NULL author_id, fetches Discord messages
in the exact ID range needed and updates matching rows. Saves per-channel
progress so it can be safely interrupted and resumed.

Usage:
    python3.14 backfill.py
"""

import os
import sys
import ssl
import sqlite3
import asyncio
import aiohttp
import certifi
import discord
from dotenv import load_dotenv

ssl_context = ssl.create_default_context(cafile=certifi.where())

load_dotenv()

BOT_TOKEN    = os.getenv("DISCORD_BOT_TOKEN")
GUILD_ID_RAW = os.getenv("DISCORD_GUILD_ID")
DB_PATH      = "messages.db"

FETCH_TIMEOUT  = 60.0   # seconds per Discord API page
MAX_TIMEOUTS   = 3      # consecutive timeouts before skipping a channel
PROGRESS_EVERY = 1000   # print a line every N messages scanned from Discord
COMMIT_EVERY   = 500    # UPDATE rows per DB commit


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_progress_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backfill_progress (
            channel_name    TEXT PRIMARY KEY,
            last_message_id TEXT NOT NULL
        )
    """)
    conn.commit()


def get_null_ids_by_channel(conn: sqlite3.Connection) -> dict:
    """Return {channel_name: set_of_message_ids} for all NULL author_id rows."""
    rows = conn.execute(
        "SELECT channel_name, id FROM messages WHERE author_id IS NULL"
    ).fetchall()
    by_channel: dict = {}
    for channel_name, msg_id in rows:
        by_channel.setdefault(channel_name, set()).add(msg_id)
    return by_channel


def get_backfill_progress(conn: sqlite3.Connection, channel_name: str) -> str | None:
    row = conn.execute(
        "SELECT last_message_id FROM backfill_progress WHERE channel_name = ?",
        (channel_name,),
    ).fetchone()
    return row[0] if row else None


def save_backfill_progress(conn: sqlite3.Connection, channel_name: str, last_message_id: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO backfill_progress (channel_name, last_message_id) VALUES (?, ?)",
        (channel_name, last_message_id),
    )
    conn.commit()


def flush_updates(conn: sqlite3.Connection, updates: list, channel_name: str, last_seen_id: str) -> None:
    conn.executemany(
        "UPDATE messages SET author_id = ? WHERE id = ?", updates
    )
    conn.commit()
    save_backfill_progress(conn, channel_name, last_seen_id)


# ---------------------------------------------------------------------------
# Per-channel backfill
# ---------------------------------------------------------------------------

async def backfill_channel(
    channel: discord.TextChannel,
    null_ids: set,
    conn: sqlite3.Connection,
) -> int:
    """
    Scan Discord messages in [min_null_id, max_null_id] for this channel,
    updating author_id for every matching row. Returns count of rows filled.
    """
    min_null_id = min(null_ids)
    max_null_id = max(null_ids)

    # Resume or start just before the first NULL message
    after_id = get_backfill_progress(conn, channel.name)
    if after_id:
        after = discord.Object(id=int(after_id))
        print(f"  #{channel.name}: resuming after {after_id} ({len(null_ids):,} still NULL)")
    else:
        # Start one ID before the first NULL so it's included in the range
        after = discord.Object(id=int(min_null_id) - 1)

    # Stop once we've passed the last NULL message
    before = discord.Object(id=int(max_null_id) + 1)

    remaining = set(null_ids)   # IDs still needing author_id
    pending_updates: list = []  # (author_id, msg_id) pairs not yet committed
    scanned  = 0
    filled   = 0
    consecutive_timeouts = 0
    last_seen_id: str | None = after_id

    history = channel.history(
        limit=None,
        oldest_first=True,
        after=after,
        before=before,
    )

    while remaining:
        try:
            message = await asyncio.wait_for(history.__anext__(), timeout=FETCH_TIMEOUT)
            consecutive_timeouts = 0
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            consecutive_timeouts += 1
            print(
                f"\n  #{channel.name}: fetch timeout ({consecutive_timeouts}/{MAX_TIMEOUTS})...",
                flush=True,
            )
            if consecutive_timeouts >= MAX_TIMEOUTS:
                print(f"  #{channel.name}: {MAX_TIMEOUTS} consecutive timeouts — saving and skipping")
                if last_seen_id:
                    if pending_updates:
                        flush_updates(conn, pending_updates, channel.name, last_seen_id)
                        pending_updates = []
                    else:
                        save_backfill_progress(conn, channel.name, last_seen_id)
                return filled
            continue

        last_seen_id = str(message.id)
        scanned += 1

        if last_seen_id in remaining:
            pending_updates.append((str(message.author.id), last_seen_id))
            remaining.discard(last_seen_id)
            filled += 1

        if scanned % PROGRESS_EVERY == 0:
            pct = (len(null_ids) - len(remaining)) / len(null_ids) * 100
            print(
                f"  #{channel.name}: scanned {scanned:,} | "
                f"filled {filled:,}/{len(null_ids):,} ({pct:.1f}%)",
                end="\r", flush=True,
            )

        if len(pending_updates) >= COMMIT_EVERY:
            flush_updates(conn, pending_updates, channel.name, last_seen_id)
            pending_updates = []

    # Commit any leftover updates
    if pending_updates:
        flush_updates(conn, pending_updates, channel.name, last_seen_id)

    unfilled = len(remaining)
    note = f" ({unfilled:,} not found in Discord — deleted messages?)" if unfilled else ""
    print(
        f"  #{channel.name}: filled {filled:,}/{len(null_ids):,} "
        f"(scanned {scanned:,} messages){note}" + " " * 20
    )
    return filled


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

    conn = sqlite3.connect(DB_PATH)
    try:
        total_null = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE author_id IS NULL"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        print("Error: messages table not found or author_id column missing.")
        print("Run migrate.py first, then sync.py.")
        conn.close()
        sys.exit(1)

    if total_null == 0:
        print("Nothing to do — all rows already have author_id.")
        conn.close()
        return

    print(f"Found {total_null:,} messages with NULL author_id\n")
    init_progress_table(conn)
    null_by_channel = get_null_ids_by_channel(conn)

    intents = discord.Intents.default()
    intents.message_content = True
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    client = discord.Client(intents=intents, connector=connector)

    @client.event
    async def on_ready() -> None:
        print(f"Logged in as {client.user}\n")

        guild = client.get_guild(guild_id)
        if not guild:
            print(f"Error: Guild ID {guild_id} not found.")
            await client.close()
            return

        channel_map = {c.name: c for c in guild.channels if isinstance(c, discord.TextChannel)}

        total_filled = 0
        for channel_name, null_ids in sorted(null_by_channel.items()):
            channel = channel_map.get(channel_name)
            if not channel:
                print(f"  #{channel_name}: channel not found in server, skipping ({len(null_ids):,} rows)")
                continue

            print(f"Backfilling #{channel_name} — {len(null_ids):,} NULL rows ...")
            try:
                filled = await backfill_channel(channel, null_ids, conn)
                total_filled += filled
            except discord.Forbidden:
                print(f"  #{channel_name}: no access, skipping")
            except Exception as exc:
                print(f"  #{channel_name}: error — {exc}")

        remaining_null = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE author_id IS NULL"
        ).fetchone()[0]
        print(f"\nBackfill complete — filled {total_filled:,} rows, {remaining_null:,} still NULL")
        conn.close()
        await client.close()

    await client.start(BOT_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
