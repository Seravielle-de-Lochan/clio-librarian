"""Prune-step tests: renamed/deleted notes leave the index; a vanished vault does not."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import chromadb
from librarian import ingest as ing

tmp = Path(tempfile.mkdtemp())
vault, db = tmp / "vault", tmp / "db"
shutil.copytree("tests/sample_vault", vault)
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def paths_in_index():
    coll = chromadb.PersistentClient(path=str(db)).get_collection(ing.COLLECTION_NAME)
    return {(m or {}).get("path") for m in coll.get(include=["metadatas"])["metadatas"]}


ing.ingest(vault, db, use_fallback=True, verbose=False)
before = paths_in_index()
check("baseline ingest", len(before) == 4, f"{len(before)} notes")

# --- rename a note: old path must leave, new path must arrive ---
old = vault / "Journal" / "2026-02-20_The_Flu_Days.md"
new = old.with_name("2026-02-20_Renamed_Flu_Days.md")
old.rename(new)
ing.PRUNE_MAX_FRACTION = 0.9          # 1-of-4 exceeds the real cap; test the mechanism
s = ing.ingest(vault, db, use_fallback=True, verbose=False)
after = paths_in_index()
check("renamed note's old path pruned",
      not any("The_Flu_Days" in (p or "") for p in after), f"pruned={s.get('chunks_pruned')}")
check("renamed note's new path present",
      any("Renamed_Flu_Days" in (p or "") for p in after))
check("no note count drift", len(after) == 4, f"{len(after)} notes")

# --- safety valve: an empty vault must NOT empty the index ---
ing.PRUNE_MAX_FRACTION = 0.20
empty = tmp / "empty_vault"
empty.mkdir()
s2 = ing.ingest(empty, db, use_fallback=True, verbose=False)
survived = paths_in_index()
check("vanished vault refused, index intact",
      len(survived) == 4 and s2.get("prune_refused"), f"refused={s2.get('prune_refused')}")

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
