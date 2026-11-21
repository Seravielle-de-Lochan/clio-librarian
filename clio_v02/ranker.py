"""
Clio v0.2 — hybrid re-ranker + session dedup.

Flow (wraps the existing v0.1 search, does not replace it):
  1. Chroma query with n_results = SEARCH_CANDIDATES (oversample)
  2. rank(candidates, query, session_id) → scored, deduped, top N
  3. AccessStore.record_retrieval() on what was actually returned

Scoring (weights in config_v02):
  semantic + keyword + recency + access_strength + type_weight + tag_match
Superseded chunks are demoted by SUPERSEDED_MULTIPLIER.

Candidates are dicts:
  {id, text, metadata: {date_num?, type?, tags?, source?, superseded_by?},
   distance}   # cosine distance from Chroma
"""

import math
import re
import uuid
from datetime import date

from . import config_v02 as cfg


# ── protection check ───────────────────────────────────────────
def is_protected(meta):
    src = (meta or {}).get("source", "") or (meta or {}).get("path", "")
    tags = (meta or {}).get("tags", "") or ""
    if any(p in src for p in cfg.PROTECTED_PATH_PATTERNS):
        return True
    tagset = {t.strip().lower() for t in str(tags).split(",") if t.strip()}
    return any(t in tagset for t in (x.lower() for x in cfg.PROTECTED_TAGS))


# ── individual signals ─────────────────────────────────────────
def semantic_score(distance):
    """Chroma cosine distance (0=identical, 2=opposite) → similarity 0..1."""
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (distance / 2.0)))


_WORD = re.compile(r"[a-zA-Z0-9']{3,}")

def keyword_score(query, text):
    """Literal term overlap: exact names/terms surface reliably even when
    the embedding underweights them."""
    terms = {t.lower() for t in _WORD.findall(query or "")}
    if not terms:
        return 0.0
    lowered = (text or "").lower()
    hits = sum(1 for t in terms if t in lowered)
    return hits / len(terms)


def recency_score(meta):
    """Exponential decay on the numeric date shadow field .
    Date-blind chunks (notes without a frontmatter date) get a neutral constant."""
    d = (meta or {}).get("date_num")
    if not d:
        return cfg.RECENCY_MISSING
    try:
        d = int(d)
        y, m, day = d // 10000, (d // 100) % 100, d % 100
        days_old = max(0, (date.today() - date(y, m, day)).days)
    except (ValueError, TypeError):
        return cfg.RECENCY_MISSING
    return math.exp(-cfg.RECENCY_LAMBDA * days_old)


def type_score(meta):
    t = (meta or {}).get("type", "")
    return cfg.TYPE_WEIGHTS.get(t, cfg.TYPE_DEFAULT)


def tag_score(query, meta):
    tags = (meta or {}).get("tags", "") or ""
    tagset = set()
    for t in str(tags).split(","):
        tagset.update(_WORD.findall(t.lower()))
    if not tagset:
        return 0.0
    terms = {t.lower() for t in _WORD.findall(query or "")}
    if not terms:
        return 0.0
    return len(terms & tagset) / len(terms)


# ── the ranker ─────────────────────────────────────────────────
def rank(candidates, query, access_store, session=None, n_results=None):
    """Score, demote superseded, dedup against session, return top N.
    Caller should then access_store.record_retrieval([ids returned])."""
    n_results = n_results or cfg.SEARCH_RESULTS
    seen = session.seen if session else set()
    scored = []
    for c in candidates:
        if c["id"] in seen:
            continue
        meta = c.get("metadata") or {}
        prot = is_protected(meta)
        s = (cfg.W_SEMANTIC * semantic_score(c.get("distance"))
             + cfg.W_KEYWORD * keyword_score(query, c.get("text", ""))
             + cfg.W_RECENCY * recency_score(meta)
             + cfg.W_ACCESS * access_store.current_strength(c["id"], prot)
             + cfg.W_TYPE * type_score(meta)
             + cfg.W_TAG * tag_score(query, meta))
        if meta.get("superseded_by"):
            s *= cfg.SUPERSEDED_MULTIPLIER
        scored.append((s, prot, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n_results]
    if session:
        session.mark_seen(c["id"] for _, _, c in top)
    results = []
    for score, prot, c in top:
        results.append({
            "id": c["id"],
            "text": c.get("text", ""),
            "metadata": c.get("metadata"),
            "score": round(score, 4),
            "protected": prot,
        })
    return results


# ── session store (in-memory, resets with server) ──────────────
class SessionStore:
    def __init__(self):
        self._sessions = {}

    def start(self):
        sid = uuid.uuid4().hex[:16]
        self._sessions[sid] = set()
        return sid

    def get(self, sid):
        """Unknown/expired ids get a fresh live session under the same id —
        graceful after server restarts mid-conversation."""
        if sid not in self._sessions:
            self._sessions[sid] = set()
        return _SessionView(self._sessions[sid])


class _SessionView:
    def __init__(self, seen_set):
        self.seen = seen_set

    def mark_seen(self, ids):
        self.seen.update(ids)
