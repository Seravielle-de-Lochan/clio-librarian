"""
Clio, the librarian — MCP server.

One librarian, many readers: exposes a vault's vector memory to every
assistant you connect (Claude Desktop, Claude Code, claude.ai via a tunnel)
over the Model Context Protocol. Local by default. No cloud.

Tools:
  - memory_start()
      Orientation call, once per conversation: recents + serendipity +
      active threads + corpus vocab. Returns a session_id for dedup.
  - search_memory(query, n_results=6, type_filter="", tag_filter="", after="", before="", session_id="")
      Semantic search over the curated archive, re-ranked by the v0.2
      hybrid scorer (semantic/keyword/recency/access/type/tags).
  - recall_window(after, before, n_results=10, session_id="")
      "What was happening between these dates" — date-bounded recall.

v0.2: read-path upgrade adapted from the Claran hybrid-ranking design —
oversample from Chroma, re-rank with clio_v02.ranker, strengthen retrieved
chunks (Bjork dynamics, JSON sidecar next to the DB). Ingest untouched.

Run:
    pip install mcp chromadb sentence-transformers
    python -m librarian.server --db ./librarian_db

Claude Desktop / Claude Code config snippet (see README.md):
    "clio": {
      "command": "/path/to/clio-librarian/.venv/bin/python",
      "args": ["-m", "librarian.server", "--db", "/path/to/clio-librarian/librarian_db"],
      "env": {"PYTHONPATH": "/path/to/clio-librarian"}
    }
"""

from __future__ import annotations
import argparse
from pathlib import Path

import chromadb
from mcp.server.fastmcp import FastMCP

from .embeddings import get_embedder
from .ingest import COLLECTION_NAME, date_to_num

from clio_v02 import config_v02 as cfg
from clio_v02.access_store import AccessStore
from clio_v02.ranker import rank, SessionStore
from clio_v02.memory_start_impl import build_memory_start

mcp = FastMCP("clio")

# Set by main() when --oauth is on. The approval routes below read it.
_oauth = None

_db_path: Path = Path("./librarian_db")
_client = None
_coll = None
_embedder = None
_access = None
_sessions = SessionStore()


def _ensure():
    global _client, _coll, _embedder, _access
    if _coll is None:
        _client = chromadb.PersistentClient(path=str(_db_path))
        _coll = _client.get_or_create_collection(COLLECTION_NAME)
        _embedder = get_embedder()
        # The fallback embedder must never silently serve a real-model DB:
        # both are 384-dim, so a broken install would return quiet nonsense
        # instead of an error. Compare against what the ingester recorded.
        stored = (_coll.metadata or {}).get("embedder", "")
        if stored and stored != _embedder.name:
            raise RuntimeError(
                f"Embedder mismatch: this database was built with '{stored}' "
                f"but the server loaded '{_embedder.name}'. Refusing to serve "
                f"wrong-brain results — fix the environment or re-ingest."
            )
        _access = AccessStore(str(_db_path / "access_strengths.json"))
    return _coll, _embedder


class _StoreAdapter:
    """The two reads memory_start needs, backed by the Chroma collection."""

    def __init__(self, coll):
        self._coll = coll

    def all_ids_with_meta(self):
        got = self._coll.get(include=["metadatas"])
        return list(zip(got["ids"], got["metadatas"]))

    def fetch(self, ids):
        got = self._coll.get(ids=list(ids), include=["documents", "metadatas"])
        return [{"id": i, "text": d, "metadata": m}
                for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]


def _candidates(results):
    """Chroma query result -> the ranker's candidate dict shape."""
    if not results["ids"] or not results["ids"][0]:
        return []
    return [{"id": _id, "text": doc, "metadata": meta or {}, "distance": dist}
            for _id, doc, meta, dist in zip(
                results["ids"][0], results["documents"][0],
                results["metadatas"][0], results["distances"][0])]


def _format_ranked(ranked) -> str:
    if not ranked:
        return "The librarian found nothing for that query. Try broader wording — semantic search rewards concepts over keywords."
    lines = []
    for r in ranked:
        meta = r["metadata"] or {}
        header = f"— {meta.get('path', r['id'])}"
        if meta.get("date"):
            header += f" | {meta['date']}"
        if meta.get("type"):
            header += f" | {meta['type']}"
        header += f" | score {r['score']:.2f}"
        if r["protected"]:
            header += " | protected"
        lines.append(f"{header}\n{r['text'].strip()}")
    return "\n\n====\n\n".join(lines)


def _rank_and_record(results, query, session_id, n_results):
    """Shared v0.2 read path: candidates -> rank -> strengthen -> format."""
    cands = _candidates(results)
    session = _sessions.get(session_id) if session_id else None
    ranked = rank(cands, query, _access, session=session, n_results=n_results)
    if ranked:
        _access.record_retrieval([r["id"] for r in ranked],
                                 {r["id"]: r["protected"] for r in ranked})
    return _format_ranked(ranked)


@mcp.tool()
def memory_start() -> dict:
    """Orientation call — use once at the start of a conversation. Returns
    recent memories, a few nearly-forgotten ones (serendipity — worth a
    glance, they surfaced for a reason), currently-active threads, the
    corpus vocabulary (types/tags for filtering), and a session_id. Pass
    that session_id to search_memory/recall_window so this conversation
    never gets the same chunk twice."""
    coll, _ = _ensure()
    return build_memory_start(_StoreAdapter(coll), _access, _sessions)


@mcp.tool()
def search_memory(query: str, n_results: int = 6, type_filter: str = "",
                  tag_filter: str = "", after: str = "", before: str = "",
                  session_id: str = "") -> str:
    """Semantic search over the archive (journals, summaries, letters,
    notes, core documents). Filters: type_filter (e.g. 'summary', 'journal'),
    tag_filter (substring match on tags), after/before (ISO dates, e.g.
    '2026-03-01'). session_id (from memory_start) dedups results within a
    conversation. Returns sourced, dated excerpts — always cite the path when
    using a result, and prefer the source note for anything load-bearing."""
    coll, embedder = _ensure()
    where = _build_where(type_filter, after, before)
    kwargs = dict(
        query_embeddings=embedder.embed([query]),
        n_results=cfg.SEARCH_CANDIDATES,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where
    results = coll.query(**kwargs)
    if tag_filter:
        results = _filter_tags(results, tag_filter)
    return _rank_and_record(results, query, session_id,
                            max(1, min(int(n_results), 20)))


@mcp.tool()
def recall_window(after: str, before: str, n_results: int = 10,
                  session_id: str = "") -> str:
    """Date-bounded recall: 'what was happening between these dates'. ISO dates
    (YYYY-MM-DD). Returns the period's chunks, most representative first."""
    coll, embedder = _ensure()
    query = "significant events, decisions, discoveries, and emotional milestones"
    results = coll.query(
        query_embeddings=embedder.embed([query]),
        n_results=cfg.SEARCH_CANDIDATES,
        include=["documents", "metadatas", "distances"],
        where=_build_where("", after, before),
    )
    return _rank_and_record(results, query, session_id,
                            max(1, min(int(n_results), 30)))


def _build_where(type_filter: str, after: str, before: str):
    clauses = []
    if type_filter:
        clauses.append({"type": {"$eq": type_filter}})
    # ChromaDB only supports $gte/$lte on numbers — filter on date_num
    # (YYYYMMDD int, written at ingest), never on the date string.
    if after:
        clauses.append({"date_num": {"$gte": date_to_num(after)}})
    if before:
        clauses.append({"date_num": {"$lte": date_to_num(before)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _filter_tags(results, tag_filter: str):
    keep = [i for i, meta in enumerate(results["metadatas"][0])
            if tag_filter.lower() in (meta or {}).get("tags", "").lower()]
    for key in ("ids", "documents", "metadatas", "distances"):
        results[key][0] = [results[key][0][i] for i in keep]
    return results


# ----------------------------------------------------- the approval gate ----
_APPROVE_PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clio — approve access</title>
<style>
 :root{{color-scheme:dark}}
 body{{margin:0;min-height:100vh;display:grid;place-items:center;
   background:#14181a;color:#e8e6e3;
   font:16px/1.55 ui-sans-serif,system-ui,"Segoe UI",sans-serif}}
 .card{{width:min(94vw,26rem);background:#1b2124;border:1px solid #2b3438;
   border-radius:14px;padding:1.9rem 1.7rem;box-shadow:0 12px 40px #0008}}
 h1{{margin:0 0 .2rem;font-size:1.22rem;color:#5a8f9b;letter-spacing:.01em}}
 p{{margin:.55rem 0;color:#a8b2b6;font-size:.93rem}}
 .who{{color:#e8e6e3;font-weight:600}}
 input{{width:100%;box-sizing:border-box;margin:1.1rem 0 .9rem;padding:.7rem .8rem;
   background:#111517;border:1px solid #33403f;border-radius:8px;
   color:#e8e6e3;font-size:1rem}}
 input:focus{{outline:2px solid #5a8f9b;outline-offset:1px;border-color:#5a8f9b}}
 button{{width:100%;padding:.72rem;background:#5a8f9b;color:#0d1113;border:0;
   border-radius:8px;font-size:1rem;font-weight:650;cursor:pointer}}
 button:hover{{background:#6ba3b0}}
 .err{{color:#e2887f;font-size:.9rem;margin:.4rem 0 0}}
 .foot{{margin:1.15rem 0 0;font-size:.8rem;color:#6c7a7f}}
</style>
<div class="card">
  <h1>Clio — the librarian</h1>
  <p><span class="who">{client}</span> is asking to read the archive.</p>
  <form method="post">
    <input type="hidden" name="request" value="{request}">
    <input type="password" name="passphrase" placeholder="Passphrase"
           autofocus autocomplete="current-password">
    <button type="submit">Open the door</button>
  </form>
  {error}
  <p class="foot">Memories stay on this machine. This grants read access only.</p>
</div>"""


def _approve_html(request_id: str, client: str, error: str = "") -> str:
    from html import escape
    err = f'<p class="err">{escape(error)}</p>' if error else ""
    return _APPROVE_PAGE.format(client=escape(client), request=escape(request_id), error=err)


@mcp.custom_route("/approve", methods=["GET", "POST"])
async def approve(request):
    """Passphrase gate. Registration is open; this is what actually guards Clio."""
    from starlette.responses import HTMLResponse, RedirectResponse

    if _oauth is None:                                    # pragma: no cover
        return HTMLResponse("OAuth is not enabled on this server.", status_code=404)

    if request.method == "GET":
        rid = request.query_params.get("request", "")
        pending = _oauth.peek_pending(rid)
        if not pending:
            return HTMLResponse(
                _approve_html("", "Unknown or expired request",
                              "That approval link has expired. Start the connection again."),
                status_code=400)
        return HTMLResponse(_approve_html(rid, pending.get("client_name", "A client")))

    form = await request.form()
    rid = str(form.get("request", ""))
    pending = _oauth.peek_pending(rid)
    if not pending:
        return HTMLResponse(
            _approve_html("", "Unknown or expired request",
                          "That approval link has expired. Start the connection again."),
            status_code=400)
    if not _oauth.check_passphrase(str(form.get("passphrase", ""))):
        return HTMLResponse(
            _approve_html(rid, pending.get("client_name", "A client"),
                          "That passphrase doesn't match."),
            status_code=401)

    target = _oauth.complete_approval(rid)
    if not target:                                        # pragma: no cover
        return HTMLResponse(
            _approve_html("", "Expired", "The request expired while you were typing."),
            status_code=400)
    return RedirectResponse(target, status_code=302)


def main():
    global _db_path, _oauth
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=Path("./librarian_db"), type=Path)
    ap.add_argument("--http", type=int, default=0, metavar="PORT",
                    help="Serve streamable HTTP on 127.0.0.1:PORT instead of "
                         "stdio — for remote access via a tunnel. "
                         "Run as a second instance; Desktop keeps stdio.")
    ap.add_argument("--http-path", default="/mcp", metavar="PATH",
                    help="URL path of the HTTP endpoint. When the port is "
                         "tunnelled to the public internet, use a long random "
                         "path — it acts as a shared secret; everything else "
                         "404s.")
    ap.add_argument("--http-host", default="", metavar="HOSTNAME",
                    help="Public hostname the tunnel serves us under (e.g. "
                         "my-machine.example.ts.net). Needed because the SDK's "
                         "DNS-rebinding protection otherwise rejects any Host "
                         "header that isn't localhost.")
    ap.add_argument("--oauth", action="store_true",
                    help="Serve OAuth 2.1 (discovery + dynamic client registration "
                         "+ passphrase-gated authorization). Required by claude.ai's "
                         "connector layer, which refuses no-auth servers.")
    ap.add_argument("--public-url", default="", metavar="URL",
                    help="Public base URL this server is reached at, e.g. "
                         "https://my-machine.example.ts.net — the OAuth issuer.")
    ap.add_argument("--set-passphrase", default="", metavar="PHRASE",
                    help="Set the approval passphrase and exit.")
    ap.add_argument("--stateless", action="store_true",
                    help="Hold no per-connection session state. This server is "
                         "restarted nightly by the re-ingest and again on every "
                         "boot; with sessions, each restart orphans the client's "
                         "session id and it gets 'Bad Request: Missing session "
                         "ID' until it re-registers. Our three tools are plain "
                         "ask-and-answer, so there is nothing a session buys us.")
    args = ap.parse_args()
    _db_path = args.db

    if args.set_passphrase:
        from .oauth import ClioOAuthProvider
        ClioOAuthProvider(str(args.db / "oauth_store.json"),
                          args.public_url or "http://localhost").set_passphrase(args.set_passphrase)
        print("Passphrase set.")
        return

    if args.oauth:
        if not args.public_url:
            ap.error("--oauth requires --public-url (the OAuth issuer must be "
                     "the address clients actually reach).")
        from mcp.server.auth.provider import ProviderTokenVerifier
        from mcp.server.auth.settings import (
            AuthSettings, ClientRegistrationOptions, RevocationOptions)
        from .oauth import ClioOAuthProvider, SCOPE

        _oauth = ClioOAuthProvider(str(args.db / "oauth_store.json"), args.public_url)
        if not _oauth.has_passphrase():
            ap.error("No approval passphrase set. Run once with --set-passphrase first.")
        # FastMCP takes these in its constructor, but the tool decorators need a
        # module-level instance. The app is assembled at run() time, so attaching
        # here is equivalent — and keeps the tools registered.
        mcp._auth_server_provider = _oauth
        mcp._token_verifier = ProviderTokenVerifier(_oauth)
        mcp.settings.auth = AuthSettings(
            issuer_url=args.public_url,
            resource_server_url=args.public_url.rstrip("/") + args.http_path,
            client_registration_options=ClientRegistrationOptions(
                enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE]),
            revocation_options=RevocationOptions(enabled=True),
            required_scopes=[SCOPE],
        )

    # Eager-load model + DB at spawn: a cold sentence-transformers load on
    # the FIRST QUERY sits exactly where MCP client timeouts live.
    _ensure()
    if args.http:
        from mcp.server.transport_security import TransportSecuritySettings
        allowed_hosts = [f"127.0.0.1:{args.http}", f"localhost:{args.http}"]
        allowed_origins = []
        if args.http_host:
            allowed_hosts += [args.http_host, f"{args.http_host}:443"]
            allowed_origins.append(f"https://{args.http_host}")
        mcp.settings.transport_security = TransportSecuritySettings(
            allowed_hosts=allowed_hosts, allowed_origins=allowed_origins)
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = args.http
        mcp.settings.streamable_http_path = args.http_path
        mcp.settings.stateless_http = args.stateless
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
