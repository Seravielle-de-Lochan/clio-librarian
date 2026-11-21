"""
Clio v0.2 — memory_start(): the orientation call.

Called once at the start of a context window. Returns:
  - session_id            (pass to all subsequent searches for dedup)
  - recent                (latest chunks by document date)
  - serendipity           (weighted-random pulls toward the nearly
                           forgotten — the shoulder tap)
  - active_threads        (high current strength + recent = what's alive)
  - corpus_vocab          (known types + a tag sample, so the caller
                           knows what filters exist)

Integration note: this module is transport-agnostic — wire
`build_memory_start(store, access_store, sessions)` into a FastMCP
@tool the same way search_memory is wired. `store` needs two reads:
  store.all_ids_with_meta() -> [(id, metadata), ...]   (Chroma .get())
  store.fetch(ids)          -> [{id, text, metadata}, ...]
"""

import random
from . import config_v02 as cfg
from .ranker import is_protected


def _date_key(meta):
    try:
        return int((meta or {}).get("date_num") or 0)
    except (TypeError, ValueError):
        return 0


def _doc_path(cid, meta):
    return (meta or {}).get("path") or (meta or {}).get("source") or cid


def build_memory_start(store, access_store, sessions):
    session_id = sessions.start()
    all_meta = store.all_ids_with_meta()          # [(id, meta)]

    # Every section works at DOCUMENT level —
    # one chunk per note. Chunk-level draws returned two paragraphs of the
    # same note as two "memories" (seen in first production smoke test).

    # ── recent: latest documents, one opening chunk each ───────
    dated = [(cid, m) for cid, m in all_meta if _date_key(m) > 0]
    dated.sort(key=lambda x: (-_date_key(x[1]),
                              int((x[1] or {}).get("chunk", 0) or 0)))
    recent_ids, recent_docs = [], set()
    for cid, m in dated:
        p = _doc_path(cid, m)
        if p in recent_docs:
            continue
        recent_docs.add(p)
        recent_ids.append(cid)
        if len(recent_ids) == cfg.START_RECENT:
            break

    # ── serendipity: the shoulder tap (one candidate per doc) ──
    by_doc = {}
    for cid, m in all_meta:
        p = _doc_path(cid, m)
        if p in recent_docs:
            continue
        by_doc.setdefault(p, []).append(cid)
    pool = [random.choice(cids) for cids in by_doc.values()]
    if len(pool) > cfg.SERENDIPITY_POOL:
        pool = random.sample(pool, cfg.SERENDIPITY_POOL)
    prot = {cid: is_protected(m) for cid, m in all_meta}
    serendipity_ids = access_store.serendipity_sample(
        pool, cfg.START_SERENDIPITY, protected_lookup=prot)

    # ── active threads: strong AND recent, one per doc ─────────
    meta_by_id = dict(all_meta)
    taken = recent_docs | {_doc_path(c, meta_by_id.get(c)) for c in serendipity_ids}
    strong = []
    for cid, m in dated[:200]:                    # recent window only
        s = access_store.current_strength(cid, prot.get(cid, False))
        if s > cfg.ACCESS_DEFAULT:                # has actually been used
            strong.append((s, cid))
    strong.sort(reverse=True)
    active_ids, active_docs = [], set()
    for s, cid in strong:
        p = _doc_path(cid, meta_by_id.get(cid))
        if p in taken or p in active_docs:
            continue
        active_docs.add(p)
        active_ids.append(cid)
        if len(active_ids) == cfg.START_ACTIVE:
            break

    # ── fetch texts, mark seen, record retrieval ───────────────
    wanted = recent_ids + serendipity_ids + active_ids
    fetched = {c["id"]: c for c in store.fetch(wanted)}
    session = sessions.get(session_id)
    session.mark_seen(wanted)
    access_store.record_retrieval(wanted, protected_lookup=prot)

    def pack(ids):
        return [{"id": i,
                 "text": fetched[i]["text"],
                 "metadata": fetched[i].get("metadata")}
                for i in ids if i in fetched]

    # ── vocab: what filters exist ──────────────────────────────
    types, tags = set(), set()
    for _, m in all_meta:
        if m and m.get("type"):
            types.add(m["type"])
        for t in str((m or {}).get("tags", "")).split(","):
            if t.strip():
                tags.add(t.strip())

    return {
        "session_id": session_id,
        "recent": pack(recent_ids),
        "serendipity": pack(serendipity_ids),
        "active_threads": pack(active_ids),
        "corpus_vocab": {
            "types": sorted(types),
            "tag_sample": sorted(random.sample(
                sorted(tags), min(40, len(tags)))) if tags else [],
        },
    }
