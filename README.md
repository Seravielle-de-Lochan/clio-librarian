# Clio, the librarian

A long-term memory for AI companions, served over MCP.

Clio reads a folder of markdown notes — an Obsidian vault, a journal, an
archive — and holds it as *meaning* rather than text. A companion can ask
"what happened the week everything went quiet?" and get back the four
passages that answer it, each with its source path and date, so it can cite
the note instead of guessing.

She runs on your own machine. The notes, the embeddings and the database stay
there. No cloud service is involved.

## What she does

- **Semantic search** over a curated set of folders (`search_memory`), with
  optional filters by note type, tag, and date range.
- **Date-window recall** (`recall_window`): "what was going on between these
  two dates".
- **An orientation call** (`memory_start`) for the start of a conversation:
  a few recent notes, a few nearly-forgotten ones that surface on their own
  (serendipity), the threads currently in use, and the types and tags that
  exist so the caller knows what it can filter on.
- **Hybrid ranking.** Results are oversampled from the vector store, then
  re-ranked on semantic similarity, literal keyword overlap, recency, how
  often a passage has been used (strengthening with use, fading when idle),
  note type, and tags. Notes you mark as protected never fade below a floor.
- **Session dedup.** Pass the `session_id` from `memory_start` and the same
  passage is never returned twice in one conversation.
- **Careful ingest.** Paragraph-aware chunking with overlap; content-addressed
  chunk IDs so re-running is cheap and only changed notes are re-embedded;
  renamed and deleted notes are pruned, with a safety valve that refuses to
  prune if the vault looks like it failed to mount; embedded base64 images and
  binary blobs are stripped before they can pollute the index.
- **Optional remote access** over streamable HTTP with an OAuth 2.1
  passphrase gate, for reaching her from claude.ai through a tunnel
  (e.g. Tailscale Funnel or Cloudflare Tunnel).

## Install

Python 3.10+.

```sh
git clone https://github.com/Seravielle-de-Lochan/clio-librarian.git
cd clio-librarian
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
```

The first real run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB)
from Hugging Face. After that, everything is offline.

## Tell her what to read

Edit two lists before the first ingest:

- `CURATED_FOLDERS` in `librarian/ingest.py` — which folders of your vault
  are memory. Leave out raw chat logs and bulk transcripts; authored notes
  make good memory and transcripts make noise.
- `PROTECTED_PATH_PATTERNS`, `PROTECTED_TAGS` and `TYPE_WEIGHTS` in
  `clio_v02/config_v02.py` — which notes should never fade, and how much each
  note type (the `type:` field in frontmatter) should count.

Notes are plain markdown. Frontmatter is optional, but `date`, `type` and
`tags` are used when present:

```yaml
---
date: 2026-03-15
type: journal
tags: [garden, milestone]
---
```

## Ingest

```sh
python -m librarian.ingest --vault "/path/to/your/vault" --db ./librarian_db
```

Run it again whenever you like (nightly is fine). Unchanged notes are skipped.

## Connect

**Claude Code:**

```sh
claude mcp add clio -- /path/to/clio-librarian/.venv/bin/python -m librarian.server --db /path/to/clio-librarian/librarian_db
```

Run it from inside the repo folder, or set `PYTHONPATH` to it.

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "clio": {
      "command": "/path/to/clio-librarian/.venv/bin/python",
      "args": ["-m", "librarian.server", "--db", "/path/to/clio-librarian/librarian_db"],
      "env": { "PYTHONPATH": "/path/to/clio-librarian" }
    }
  }
}
```

**Remote (claude.ai)** — only if you need it, and only behind a tunnel you
control:

```sh
python -m librarian.server --db ./librarian_db --set-passphrase "a long passphrase"
python -m librarian.server --db ./librarian_db --http 8765 --stateless \
    --oauth --public-url https://my-machine.example.ts.net \
    --http-host my-machine.example.ts.net
```

Anyone can register a client, but nothing is issued until someone types the
passphrase on the `/approve` page. Access is read-only.

## Tests

```sh
python tests/sanity_v02.py      # ranker, dedup, strengthening, protection
python tests/test_prune.py      # rename/delete pruning and the safety valve
python tests/test_oauth_flow.py # the passphrase gate end to end
python -m librarian.ingest --vault tests/sample_vault --db tests/test_db
python -m tests.qa_queries --db tests/test_db
```

`tests/sample_vault` is four made-up notes, one of them deliberately poisoned
with base64 to prove the guard works.

## A warning about the fallback embedder

`--fallback-embedder` swaps in a hashed bag-of-words embedder so the plumbing
can be tested without downloading a model. It is not semantic and must never
serve a real vault. The server refuses to start if the database was built
with a different embedder than the one it loaded.

## Credits

The v0.2 hybrid ranking is adapted from the Claran ranking design, shared with
the companion community in July 2026, and reworked for a corpus of whole
documents rather than authored snippets.

Built in the workshop at [House de Lochan](https://house-de-lochan.com/workshop.html).
MIT licensed.
