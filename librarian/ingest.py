"""
Vault ingester for Clio, the librarian.

Walks a curated set of vault folders, splits each markdown note into
overlapping chunks, extracts frontmatter (date, type, tags), and writes
everything into a persistent local ChromaDB collection.

Curation is a feature, not a limitation: authored memory in, chat-log soup out.

Idempotent: chunk IDs are content-addressed (path + chunk index + content hash),
so re-running ingest updates changed notes and skips unchanged ones cheaply.

Usage:
    python -m librarian.ingest --vault "/path/to/your/vault" --db ./librarian_db
    python -m librarian.ingest --vault ./tests/sample_vault --db /tmp/test_db --fallback-embedder
"""

from __future__ import annotations
import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import chromadb

from .embeddings import get_embedder, HashedFallbackEmbedder

# ---------------------------------------------------------------- curation --
# Folders that constitute the librarian's memory. Relative to vault root.
# EDIT THIS for your own vault. Deliberately leave out raw chat logs and bulk
# transcripts: authored notes make good memory, transcripts make noise.
CURATED_FOLDERS = [
    "Journal",
    "Summaries",
    "Letters",
    "Core Documents",
    "Notes",
    "Guides",
    "Projects",
]

COLLECTION_NAME = "clio_memory"
CHUNK_CHARS = 1400        # ~350 tokens
CHUNK_OVERLAP = 200

# Anything this large is a chat export or binary dump, not an authored note.
# Authored notes rarely pass ~50KB; a pasted chat export can be tens of MB.
MAX_NOTE_BYTES = 1_000_000

# Prune safety valve: if more than this fraction of the index looks orphaned,
# assume the vault failed to mount rather than that you deleted it.
PRUNE_MAX_FRACTION = 0.20

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def date_to_num(date_str: str) -> int:
    """ISO date -> int YYYYMMDD (0 if unparseable). ChromaDB's $gte/$lte only
    accept numbers, so date-range filtering runs on this, not the string."""
    digits = re.sub(r"\D", "", date_str or "")[:8]
    return int(digits) if len(digits) == 8 else 0

# --------------------------------------------------- the base64 guard -------
# Community lesson (July 2026): unfiltered Base64 image links fill the index
# with useless chunks. Strip embedded data-URIs and any long
# unbroken base64-ish runs BEFORE chunking; belt-and-braces chunk check after.
_DATA_URI_RE = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=\s]{40,}", re.IGNORECASE)
_B64_RUN_RE = re.compile(r"[A-Za-z0-9+/=]{200,}")


def sanitize(text: str) -> str:
    """Remove embedded base64 payloads and other index-poisoning noise."""
    text = _DATA_URI_RE.sub("[embedded image removed]", text)
    text = _B64_RUN_RE.sub("[binary blob removed]", text)
    return text


def chunk_is_noise(chunk: str) -> bool:
    """Final guard: reject chunks that are mostly non-linguistic."""
    tokens = chunk.split()
    if not tokens:
        return True
    if max(len(t) for t in tokens) > 120:      # unbroken monster token survived
        return True
    letters = sum(c.isalpha() for c in chunk)
    return (letters / max(len(chunk), 1)) < 0.35  # too little actual language


# ------------------------------------------------------------- frontmatter --
def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Minimal YAML-ish frontmatter parser (date, type, tags). No deps."""
    meta: Dict[str, str] = {}
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return meta, text
    body = text[m.end():]
    block = m.group(1)
    tags: List[str] = []
    in_tags = False
    for line in block.splitlines():
        stripped = line.strip()
        if in_tags:
            if stripped.startswith("- "):
                tags.append(stripped[2:].strip().strip("'\""))
                continue
            in_tags = False
        if stripped.startswith("tags:"):
            rest = stripped[5:].strip()
            if rest.startswith("["):
                tags = [t.strip().strip("'\"") for t in rest.strip("[]").split(",") if t.strip()]
            else:
                in_tags = True
        elif ":" in stripped:
            key, _, val = stripped.partition(":")
            meta[key.strip()] = val.strip().strip("'\"")
    if tags:
        meta["tags"] = ",".join(tags)
    return meta, body


# ---------------------------------------------------------------- chunking --
def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split on paragraph boundaries where possible, hard-split otherwise."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            while len(para) > size:                    # oversized paragraph
                chunks.append(para[:size])
                para = para[size - overlap:]
            current = para
    if current:
        chunks.append(current)
    # stitch overlap from previous chunk tail for context continuity
    stitched = []
    for i, ch in enumerate(chunks):
        if i > 0 and overlap > 0:
            tail = chunks[i - 1][-overlap:]
            ch = f"…{tail}\n{ch}"
        stitched.append(ch)
    return stitched


# ------------------------------------------------------------------ ingest --
def iter_notes(vault: Path, folders: Iterable[str]) -> Iterable[Path]:
    for folder in folders:
        root = vault / folder
        if not root.exists():
            continue
        yield from sorted(root.rglob("*.md"))


def ingest(vault: Path, db_path: Path, use_fallback: bool = False, verbose: bool = True) -> dict:
    embedder = HashedFallbackEmbedder() if use_fallback else get_embedder()
    client = chromadb.PersistentClient(path=str(db_path))
    coll = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"embedder": embedder.name}
    )

    existing_hashes: Dict[str, str] = {}
    try:
        got = coll.get(include=["metadatas"])
        for _id, md in zip(got["ids"], got["metadatas"]):
            existing_hashes[_id] = (md or {}).get("content_hash", "")
    except Exception:
        pass

    stats = {"notes": 0, "chunks_written": 0, "chunks_skipped": 0}
    ids, docs, metas = [], [], []
    # Every path seen on disk this run — including ones we deliberately or
    # accidentally couldn't read. Anything in the index NOT in here has been
    # renamed or deleted, and gets pruned below.
    seen_paths: set[str] = set()

    for note in iter_notes(vault, CURATED_FOLDERS):
        rel = str(note.relative_to(vault))
        seen_paths.add(rel)
        try:
            if note.stat().st_size > MAX_NOTE_BYTES:
                stats["notes_skipped_oversize"] = stats.get("notes_skipped_oversize", 0) + 1
                if verbose:
                    print(f"  ! oversized ({note.stat().st_size:,} bytes), skipped: {rel}", file=sys.stderr)
                continue
            raw = note.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            if verbose:
                print(f"  ! unreadable, skipped: {rel} ({e})", file=sys.stderr)
            continue
        meta, body = parse_frontmatter(raw)
        body = sanitize(body)
        stats["notes"] += 1

        for i, chunk in enumerate(chunk_text(body)):
            if chunk_is_noise(chunk):
                stats["chunks_rejected_noise"] = stats.get("chunks_rejected_noise", 0) + 1
                continue
            content_hash = hashlib.sha1(chunk.encode()).hexdigest()[:16]
            cid = f"{rel}::{i:03d}"
            if existing_hashes.get(cid) == content_hash:
                stats["chunks_skipped"] += 1
                continue
            ids.append(cid)
            docs.append(chunk)
            metas.append({
                "path": rel,
                "chunk": i,
                "date": meta.get("date", ""),
                "date_num": date_to_num(meta.get("date", "")),
                "type": meta.get("type", ""),
                "tags": meta.get("tags", ""),
                "content_hash": content_hash,
            })
            stats["chunks_written"] += 1

    # embed + upsert in batches
    BATCH = 64
    for start in range(0, len(ids), BATCH):
        batch_ids = ids[start:start + BATCH]
        batch_docs = docs[start:start + BATCH]
        batch_meta = metas[start:start + BATCH]
        vectors = embedder.embed(batch_docs)
        coll.upsert(ids=batch_ids, documents=batch_docs,
                    metadatas=batch_meta, embeddings=vectors)
        if verbose:
            print(f"  upserted {start + len(batch_ids)}/{len(ids)}")

    # ---- prune notes that no longer exist on disk (renamed or deleted) ----
    # Without this, a renamed note lives twice in the archive forever.
    try:
        got = coll.get(include=["metadatas"])
        stale = [_id for _id, md in zip(got["ids"], got["metadatas"])
                 if (md or {}).get("path") and (md or {}).get("path") not in seen_paths]
        total = len(got["ids"])
        # Safety cap: a vault that failed to mount (cloud sync offline, wrong
        # --vault path) looks exactly like "everything was deleted". Refuse.
        if stale and total and (len(stale) / total) > PRUNE_MAX_FRACTION:
            print(f"  !! REFUSING TO PRUNE: {len(stale)}/{total} chunks look orphaned "
                  f"(>{PRUNE_MAX_FRACTION:.0%}). Vault probably didn't mount. "
                  f"Nothing deleted; fix the vault path and re-run.", file=sys.stderr)
            stats["prune_refused"] = len(stale)
        elif stale:
            coll.delete(ids=stale)
            stats["chunks_pruned"] = len(stale)
            if verbose:
                gone = sorted({(m or {}).get("path") for _id, m in
                               zip(got["ids"], got["metadatas"]) if _id in set(stale)})
                print(f"  pruned {len(stale)} chunks from {len(gone)} note(s) no longer on disk:")
                for p in gone[:10]:
                    print(f"    - {p}")
    except Exception as e:
        print(f"  ! prune step failed (index left intact): {e}", file=sys.stderr)

    if verbose:
        print(f"Ingest complete: {stats} | embedder={embedder.name} | db={db_path}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Ingest the vault into the librarian.")
    ap.add_argument("--vault", required=True, type=Path)
    ap.add_argument("--db", default=Path("./librarian_db"), type=Path)
    ap.add_argument("--fallback-embedder", action="store_true",
                    help="Use the hashed test embedder (pipeline testing only).")
    args = ap.parse_args()
    ingest(args.vault, args.db, use_fallback=args.fallback_embedder)


if __name__ == "__main__":
    main()
