"""QA battery for Clio — run against a populated DB.

Usage: python -m tests.qa_queries --db tests/test_db
Covers the handoff's checklist items 4 (three semantic routes on the real
embedder), the metadata filter, the ChromaDB string-date-range question,
and the base64 guard audit (no base64 in stored chunks).
"""
import argparse
import re
import sys

import chromadb

sys.path.insert(0, ".")
from librarian.embeddings import get_embedder
from librarian.ingest import COLLECTION_NAME

ROUTES = [
    ("feeling flattened, repetitive responses, presence drifting", "Flu_Days"),
    ("choosing jewellery as a gift, daisy earrings", "Market_Day"),
    ("crediting decisions to the wrong person in the records", "Archive_Correction"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="tests/test_db")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.db)
    coll = client.get_collection(COLLECTION_NAME)
    emb = get_embedder()
    passed = failed = 0

    print(f"collection metadata: {coll.metadata}")

    for query, expect in ROUTES:
        res = coll.query(query_embeddings=emb.embed([query]), n_results=1,
                         include=["metadatas", "distances"])
        top = res["metadatas"][0][0]["path"]
        ok = expect in top
        passed, failed = passed + ok, failed + (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] route '{query[:45]}…' -> {top} "
              f"(dist {res['distances'][0][0]:.3f})")

    res = coll.query(query_embeddings=emb.embed(["anything"]), n_results=5,
                     where={"type": {"$eq": "journal"}}, include=["metadatas"])
    types = {m["type"] for m in res["metadatas"][0]}
    ok = types == {"journal"}
    passed, failed = passed + ok, failed + (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] type filter -> {types}")

    try:
        from librarian.server import _build_where
        res = coll.query(query_embeddings=emb.embed(["anything"]), n_results=10,
                         where=_build_where("", "2026-03-01", "2026-03-31"),
                         include=["metadatas"])
        dates = sorted(m["date"] for m in res["metadatas"][0])
        ok = bool(dates) and all("2026-03-01" <= d <= "2026-03-31" for d in dates)
        passed, failed = passed + ok, failed + (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] string date range -> {dates}")
    except Exception as e:
        failed += 1
        print(f"[FAIL] string date range raised: {e}")

    got = coll.get(include=["documents"])
    b64 = re.compile(r"[A-Za-z0-9+/=]{100,}")
    dirty = [d for d in got["documents"] if b64.search(d) or "data:image" in d]
    ok = not dirty
    passed, failed = passed + ok, failed + (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] base64 guard: {len(got['documents'])} "
          f"stored chunks, {len(dirty)} contain base64/data-URIs")

    poisoned = [d for d in got["documents"] if "Poisoned" in d or "survive ingestion" in d]
    surviving = [d for d in poisoned if "closes the note like nothing happened" in d]
    ok = bool(surviving)
    passed, failed = passed + ok, failed + (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] poisoned note prose survived: {len(surviving)} chunk(s)")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
