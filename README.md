# Discord Channel Search

Search your Discord server's message history by chatting with Claude. Run a sync to pull all messages into a local SQLite database, then ask questions in a terminal chat loop.

---
## Live demo
[▶ Try the interactive demo](https://alexa-3.github.io/discord-search/demo.html)

---

## How it works

1. **`sync.py`** — connects to Discord with a bot token, fetches every message from every text channel, stores them in `messages.db` with FTS5 full-text search and vector embeddings.
2. **`chat.py`** — takes your question, expands it into multiple search variations via Claude, runs hybrid keyword + semantic search, and sends the top 40 results back to Claude to compose a cited answer.

---

## Step 1 — Create a Discord bot with read-only permissions

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and click **New Application**. Give it a name.
2. In the left sidebar, click **Bot**.
   - Click **Add Bot** (or **Reset Token** if a bot already exists) and copy the token — you'll need it for `.env`.
   - Under **Privileged Gateway Intents**, enable **Message Content Intent**. Save changes.
3. In the left sidebar, click **OAuth2 → URL Generator**.
   - Under **Scopes**, check `bot`.
   - Under **Bot Permissions**, check only `Read Messages/View Channels` and `Read Message History`.
   - Copy the generated URL, open it in a browser, and invite the bot to your server.
4. Enable **Developer Mode** in Discord (User Settings → Advanced → Developer Mode), then right-click your server icon and choose **Copy Server ID**.

---

## Step 2 — Install dependencies

Python 3.10+ is required. A virtual environment is recommended.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` pulls in PyTorch (~600 MB on first install) and will download the `all-MiniLM-L6-v2` model (~90 MB) on first run. Both are cached locally afterward.

---

## Step 3 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your bot token, guild ID, and Anthropic API key.

---

## Step 4 — Sync messages

```bash
python sync.py
```

The script connects to Discord, iterates through every text channel it can access, and stores each message with a vector embedding. Progress is printed per channel.

**Expected time:**

| Server size | Approximate sync time |
|-------------|----------------------|
| Small (< 50k messages) | 5–20 minutes |
| Medium (50k–500k messages) | 30 minutes – 3 hours |
| Large (500k+ messages) | Several hours or more |

The bottleneck is Discord's rate limits (50 messages per API call, ~50 calls/sec). Embedding generation adds a small additional cost per batch.

**Re-runs are fast.** Already-synced messages are skipped by message ID, so subsequent syncs only fetch new messages.

---

## Step 5 — Start chatting

```bash
python chat.py
```

Type any question about things discussed in your server and press Enter. Claude will search the message history and answer with citations.

```
You: Was there ever a discussion about switching to Postgres?

Searching...
Query variations:
  1. Postgres migration database switch
  2. switching from MySQL to PostgreSQL
  ...

Found 40 relevant messages. Generating answer...

Claude:
Yes — there was a detailed discussion in #backend on 2024-03-14 where @alice proposed
migrating from MySQL to Postgres for better JSON support (#backend, alice, 2024-03-14).
@bob raised concerns about migration complexity (#backend, bob, 2024-03-14), and the
thread concluded with no decision made pending a proof-of-concept.
```

Type `quit` or press `Ctrl+C` to exit.

---

## Notes

- **First sync is slow** due to Discord rate limits and embedding generation for each message. Re-runs only process new messages and are much faster.
- **Memory usage**: all embeddings are loaded into RAM when `chat.py` starts (~1.5 MB per 1,000 messages). For very large servers (1M+ messages), this may require 1–2 GB of RAM.
- The bot only needs read access. It never sends messages or modifies anything.
- Messages with empty content (images, embeds only) are skipped during sync.
