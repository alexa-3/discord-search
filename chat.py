"""
chat.py — Claude-powered Discord search chat

Terminal chat loop that finds relevant Discord messages via hybrid search
(FTS5 + vector embeddings) and answers questions using Claude.

Usage:
    python chat.py
"""

import json
import os
import re
import time
import sqlite3
import numpy as np
from anthropic import Anthropic
from anthropic import APIStatusError
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_PATH = "messages.db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 60            # final messages sent to Claude
RESULTS_PER_QUERY = 20  # per variation, per search method

OFFICIAL_CHANNELS = {
    "dao-announcements", "community-updates", "announcements", "farming-alerts"
}

# Triggers the OFFICIAL_CHANNELS channel filter (no explicit channel needed)
OFFICIAL_CHANNEL_KEYWORDS = {
    "official", "announcement", "announcements",
}

# Discord user IDs for known team members — triggers the TEAM_AUTHOR filter
TEAM_AUTHORS = {
    "832906877775642644",   # Krysia
    "1303022320503492650",  # chillandgreen
    "689634122305372270",   # mcarbon
    "1000836969863073893",  # aspiringhacker
    "826005593721667585",   # gshador
}

# Words that identify team-member intent (used for author filter, not channel filter)
TEAM_AUTHOR_KEYWORDS = {"team", "mods", "moderators", "admin", "admins", "staff"}

# Phrases that unambiguously request team-author filtering
TEAM_AUTHOR_PHRASES = [
    "from mods", "from admins", "from team", "from the team", "from staff",
    "what did the team", "what did mods", "what did admins",
    "mods said", "admins said", "team said",
]


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

FALLBACK_MODEL = {
    "claude-sonnet-4-6":        "claude-haiku-4-5-20251001",
    "claude-haiku-4-5-20251001": "claude-sonnet-4-6",
}

def api_create(client: Anthropic, **kwargs):
    """Call client.messages.create with exponential backoff on 529, then model fallback."""
    delays = [5, 15, 30]  # seconds between retries on the primary model
    for attempt, delay in enumerate(delays, 1):
        try:
            return client.messages.create(**kwargs)
        except APIStatusError as e:
            if e.status_code == 529 and attempt <= len(delays):
                print(f"API overloaded — retrying in {delay}s (attempt {attempt}/{len(delays)})...")
                time.sleep(delay)
            else:
                raise

    # All retries exhausted — try the fallback model once
    primary = kwargs.get("model", "")
    fallback = FALLBACK_MODEL.get(primary)
    if fallback:
        print(f"Switching model: {primary} → {fallback}")
        return client.messages.create(**{**kwargs, "model": fallback})
    raise RuntimeError(f"All retries failed and no fallback defined for model '{primary}'")


# ---------------------------------------------------------------------------
# Embedding index
# ---------------------------------------------------------------------------

def load_embeddings(conn: sqlite3.Connection):
    """Load all message embeddings from DB into a normalized numpy matrix."""
    print("Loading embeddings into memory...")
    rows = conn.execute(
        "SELECT id, embedding FROM messages WHERE embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        return [], np.empty((0, 384), dtype=np.float32)

    ids = [r[0] for r in rows]
    matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    # Pre-normalize rows for fast cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)

    print(f"Loaded {len(ids):,} embeddings")
    return ids, matrix


def vector_search(
    query_text: str,
    ids: list,
    matrix: np.ndarray,
    model: SentenceTransformer,
    top_k: int,
    allowed_ids: set | None = None,
) -> list:
    """Return [(id, cosine_score), ...] sorted best-first.

    If allowed_ids is provided, only consider embeddings whose id is in that set.
    """
    if not ids:
        return []
    if allowed_ids is not None:
        pairs = [(i, mid) for i, mid in enumerate(ids) if mid in allowed_ids]
        if not pairs:
            return []
        sub_indices = [p[0] for p in pairs]
        sub_ids     = [p[1] for p in pairs]
        sub_matrix  = matrix[sub_indices]
    else:
        sub_ids    = list(ids)
        sub_matrix = matrix
    qvec = model.encode([query_text])[0].astype(np.float32)
    qvec /= max(float(np.linalg.norm(qvec)), 1e-8)
    sim = sub_matrix @ qvec
    top_idx = np.argsort(sim)[::-1][:top_k]
    return [(sub_ids[int(i)], float(sim[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Full-text search
# ---------------------------------------------------------------------------

def sanitize_fts(query: str) -> str:
    """Strip FTS5 operator characters; return plain token string."""
    tokens = re.findall(r"\b\w+\b", query)
    return " ".join(tokens)


def fts_search(
    query_text: str,
    conn: sqlite3.Connection,
    top_k: int,
    filter_sql: str = "",
    filter_params: tuple = (),
) -> list:
    """Return [(id, normalized_score), ...] from FTS5 BM25 search."""
    safe = sanitize_fts(query_text)
    if not safe:
        return []
    id_subquery = (
        f" AND id IN (SELECT id FROM messages WHERE {filter_sql})"
        if filter_sql else ""
    )
    try:
        rows = conn.execute(
            f"SELECT id, rank FROM messages_fts WHERE content MATCH ?{id_subquery} ORDER BY rank LIMIT ?",
            (safe, *filter_params, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    if not rows:
        return []

    # FTS5 rank is negative BM25 (lower = more relevant). Flip and normalize.
    raw_scores = [-r[1] for r in rows]
    lo, hi = min(raw_scores), max(raw_scores)
    denom = max(hi - lo, 1e-8)
    return [(rows[i][0], (raw_scores[i] - lo) / denom) for i in range(len(rows))]


# ---------------------------------------------------------------------------
# Exact-match search for URLs / claim links
# ---------------------------------------------------------------------------

def exact_match_search(
    question: str,
    conn: sqlite3.Connection,
    top_k: int,
    filter_sql: str = "",
    filter_params: tuple = (),
) -> list:
    """
    Return [(id, 1.0), ...] for messages that contain a URL or claim-related
    keyword alongside any project-name token found in the question.
    """
    # Extract candidate project-name tokens: capitalised words or all-caps
    tokens = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b|\b[A-Z]{2,}\b", question)
    if not tokens:
        return []

    link_keywords = ["http", "claim", "airdrop"]
    results = []
    seen: set = set()

    for token in tokens:
        for kw in link_keywords:
            where = "content LIKE ? AND content LIKE ?"
            params: list = [f"%{token}%", f"%{kw}%"]
            if filter_sql:
                where += f" AND {filter_sql}"
                params.extend(filter_params)
            params.append(top_k)
            try:
                rows = conn.execute(
                    f"SELECT id FROM messages WHERE {where} LIMIT ?", params
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            for (msg_id,) in rows:
                if msg_id not in seen:
                    seen.add(msg_id)
                    results.append((msg_id, 1.0))

    return results


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_query(question: str, client: Anthropic) -> list:
    """Ask Claude to generate 3-5 search variations of the question."""
    response = api_create(client,
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=(
            "This is a crypto/Web3 Discord server called Cookie DAO. "
            "All queries relate to crypto projects, NFTs, tokens, airdrops, campaigns, and Web3 concepts. "
            "Never interpret terms as anything other than crypto/Web3 related. "
            "For example, 'Cookie' refers to the Cookie DAO project/token, 'Tria' is a crypto project, etc."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Generate 3-5 distinct search query variations for the question below. "
                    "Capture synonyms, related terms, and alternative phrasings. "
                    "Return ONLY a numbered list, one variation per line, no other text.\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )
    variations = []
    for line in response.content[0].text.strip().splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            variation = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
            if variation:
                variations.append(variation)
    return variations if variations else [question]


# ---------------------------------------------------------------------------
# Filter extraction
# ---------------------------------------------------------------------------

def parse_filters(question: str, client: Anthropic) -> dict:
    """Use Claude to extract any date range and channel names from the question.

    Returns a dict with keys:
        date_from  — ISO date string "YYYY-MM-DD" or null
        date_to    — ISO date string "YYYY-MM-DD" or null
        channels   — list of channel name strings (without #) or null
    """
    response = api_create(client,
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=(
            "You extract search filters from questions about a crypto/Web3 Discord server. "
            "Today's date is 2026-04-02. "
            "Return ONLY valid JSON with keys: date_from, date_to, channels. "
            "date_from and date_to are ISO date strings (YYYY-MM-DD) or null. "
            "For open-ended ranges like 'since October 2025' set date_to to null. "
            "channels is a list of explicit Discord channel name strings (without the # prefix) or null. "
            "Only set channels when the user names a specific channel like '#dao-talks' or 'in announcements'. "
            "NEVER put generic words like 'official', 'team', 'staff', 'admin', 'announcement' in channels — "
            "these describe intent, not channel names. "
            "If no explicit channel is mentioned, set channels to null. "
            "If no filter is mentioned, return {\"date_from\": null, \"date_to\": null, \"channels\": null}."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract date range and channel filters from this question. "
                    "Return only JSON, nothing else.\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )
    try:
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?|\n?```$", "", raw).strip()
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return {"date_from": None, "date_to": None, "channels": None}


def build_filter_sql(filters: dict) -> tuple:
    """Convert a filters dict into a (where_clause, params) tuple.

    The returned where_clause is suitable for appending after WHERE or AND.
    Returns ("", ()) when no filters are active.
    """
    conditions: list[str] = []
    params: list = []
    if filters.get("date_from"):
        conditions.append("DATE(timestamp) >= ?")
        params.append(filters["date_from"][:10])  # ensure YYYY-MM-DD
    if filters.get("date_to"):
        conditions.append("DATE(timestamp) <= ?")
        params.append(filters["date_to"][:10])    # ensure YYYY-MM-DD
    if filters.get("channels"):
        like_clauses = " OR ".join("channel_name LIKE ?" for _ in filters["channels"])
        conditions.append(f"({like_clauses})")
        params.extend(f"%{ch}%" for ch in filters["channels"])
    if filters.get("author_ids"):
        placeholders = ",".join("?" * len(filters["author_ids"]))
        conditions.append(f"author_id IN ({placeholders})")
        params.extend(filters["author_ids"])
    return " AND ".join(conditions), tuple(params)


def get_filtered_ids(
    conn: sqlite3.Connection, filter_sql: str, filter_params: tuple
) -> set | None:
    """Return the set of message IDs matching the filter, or None if no filter."""
    if not filter_sql:
        return None
    rows = conn.execute(
        f"SELECT id FROM messages WHERE {filter_sql}", filter_params
    ).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def _has_project_name(question: str) -> bool:
    """True if the question contains a capitalised token (likely a project name)."""
    return bool(re.search(r"\b[A-Z][a-zA-Z0-9]{2,}\b|\b[A-Z]{2,}\b", question))


def _uses_official_intent(question: str, filters: dict) -> bool:
    """True if the query implies official-channel content and no explicit channel was given.

    Triggers on: 'official', 'announcement', or the phrase 'team update'.
    Does NOT trigger if the user already named a specific channel.
    """
    if filters.get("channels"):
        return False  # user named a specific channel — respect that instead
    q_lower = question.lower()
    words = set(re.findall(r"\b\w+\b", q_lower))
    if words & OFFICIAL_CHANNEL_KEYWORDS:
        return True
    # "team update" → look for official announcements, not author-filtered results
    if "team" in words and "update" in words:
        return True
    return False


def _uses_team_author_intent(question: str, filters: dict) -> bool:
    """True if the query implies filtering by known team-member authors.

    Triggers on explicit phrases like 'from mods', 'what did the team say', etc.,
    OR when team/staff keywords appear alongside an explicitly named channel
    (e.g. 'official team updates in #cookie-talks').
    """
    q_lower = question.lower()
    # Phrase-level match — always triggers team-author filter
    if any(phrase in q_lower for phrase in TEAM_AUTHOR_PHRASES):
        return True
    # Word-level match only when the user also named a specific channel
    if filters.get("channels"):
        words = set(re.findall(r"\b\w+\b", q_lower))
        if words & TEAM_AUTHOR_KEYWORDS:
            return True
    return False


def hybrid_search(
    question: str,
    ids: list,
    matrix: np.ndarray,
    model: SentenceTransformer,
    conn: sqlite3.Connection,
    client: Anthropic,
) -> list:
    """
    Parse filters → expand query → run FTS5 + vector + exact-match search per
    variation → merge, deduplicate, boost announcement channels when a project
    name is present, return top-60 rows sorted chronologically.
    """
    # --- Filters ---
    filters = parse_filters(question, client)

    # Official channels intent: restrict to announcement channels when no
    # explicit channel was named and the query implies official content.
    if _uses_official_intent(question, filters):
        filters["channels"] = list(OFFICIAL_CHANNELS)
        print(f"Official intent detected — restricting to: {sorted(OFFICIAL_CHANNELS)}")

    # Team author intent: filter by known team-member Discord IDs.
    # Works independently of (and can combine with) the channel filter.
    if _uses_team_author_intent(question, filters):
        filters["author_ids"] = list(TEAM_AUTHORS)
        print(f"Team author intent detected — filtering by {len(TEAM_AUTHORS)} team members")

    filter_sql, filter_params = build_filter_sql(filters)
    if filter_sql:
        active = {k: v for k, v in filters.items() if v}
        print(f"Active filters: {active}")
    allowed_ids = get_filtered_ids(conn, filter_sql, filter_params)

    # --- Query expansion ---
    variations = expand_query(question, client)
    print(f"Query variations:")
    for i, v in enumerate(variations, 1):
        print(f"  {i}. {v}")
    print()

    scores: dict = {}  # message_id -> best score seen

    for variation in variations:
        # FTS5 (keyword) search — weight slightly lower than vector
        for msg_id, score in fts_search(variation, conn, RESULTS_PER_QUERY, filter_sql, filter_params):
            scores[msg_id] = max(scores.get(msg_id, 0.0), score * 0.65)

        # Vector (semantic) search
        for msg_id, score in vector_search(variation, ids, matrix, model, RESULTS_PER_QUERY, allowed_ids):
            scores[msg_id] = max(scores.get(msg_id, 0.0), score)

    # Exact-match search for URLs / claim links — scored at 1.0 (top priority)
    for msg_id, score in exact_match_search(question, conn, RESULTS_PER_QUERY, filter_sql, filter_params):
        scores[msg_id] = max(scores.get(msg_id, 0.0), score)

    if not scores:
        return []

    # Fetch all candidates so we can apply channel boost before trimming
    candidate_ids = list(scores.keys())
    placeholders = ",".join("?" * len(candidate_ids))
    rows = conn.execute(
        f"SELECT id, channel_name, author, timestamp, content"
        f"  FROM messages WHERE id IN ({placeholders})",
        candidate_ids,
    ).fetchall()

    # Boost official channels when the query names a specific project
    if _has_project_name(question):
        boosted: dict = {}
        for row in rows:
            msg_id, channel = row[0], row[1]
            ch = (channel or "").lower()
            boost = 0.15 if any(oc in ch for oc in OFFICIAL_CHANNELS) else 0.0
            boosted[msg_id] = scores[msg_id] + boost
    else:
        boosted = scores

    # Take top-60 by score, then sort chronologically for Claude
    top_ids = set(sorted(boosted, key=lambda x: boosted[x], reverse=True)[:TOP_K])
    top_rows = [r for r in rows if r[0] in top_ids]
    top_rows.sort(key=lambda r: r[3] or "")  # sort by timestamp ascending
    return top_rows


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def build_answer(question: str, messages: list, client: Anthropic) -> str:
    """Format retrieved messages into context and ask Claude for an answer."""
    if not messages:
        context = "(No relevant messages were found in the database.)"
    else:
        parts = []
        for _, channel, author, timestamp, content in messages:
            date = timestamp[:10] if timestamp else "unknown date"
            parts.append(f"[#{channel} | {author} | {date}]\n{content}")
        context = "\n\n---\n\n".join(parts)

    response = api_create(client,
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are answering questions about a Discord server based on its message history.\n"
                    "Use ONLY the messages below as your source — do not add outside knowledge.\n"
                    "Cite sources inline as (#{channel}, {author}, {date}).\n"
                    "If the answer cannot be determined from the messages, say so explicitly.\n\n"
                    f"=== Discord Messages ===\n{context}\n\n"
                    f"=== Question ===\n{question}"
                ),
            }
        ],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    try:
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    except sqlite3.OperationalError:
        print("Database not found or not initialized. Run sync.py first.")
        conn.close()
        return

    if count == 0:
        print("No messages in database. Run sync.py first.")
        conn.close()
        return

    print(f"Database: {count:,} messages\n")

    model = SentenceTransformer(EMBED_MODEL)
    ids, matrix = load_embeddings(conn)

    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

    print("\nDiscord Search — type a question, or 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nSearching...")
        messages = hybrid_search(question, ids, matrix, model, conn, anthropic_client)
        print(f"Found {len(messages)} relevant messages. Generating answer...\n")

        answer = build_answer(question, messages, anthropic_client)
        print(f"Claude:\n{answer}")
        print("\n" + "=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
