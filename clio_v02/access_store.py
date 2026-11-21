"""
Clio v0.2 — access-strength store (Bjork dual-strength model).

Memories that get retrieved consolidate; memories that don't, fade.
Protected memories never fade below a floor.

Implemented as a JSON sidecar keyed by chunk ID — no ChromaDB schema
change, no migration, safe to bolt onto the existing store. Atomic
writes (tmp file + os.replace) so a crash mid-save can't corrupt it.
"""

import json
import math
import os
import random
import tempfile
import threading
from datetime import date, datetime

from . import config_v02 as cfg


class AccessStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._data = {}
        self._load()

    # ── persistence ────────────────────────────────────────────
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # A corrupt sidecar must never take Clio down.
                # Strengths regrow from use; start fresh and log loudly.
                print(f"[access_store] WARNING: could not read {self.path}; "
                      f"starting with empty strengths.")
                self._data = {}

    def _save(self):
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp_path, self.path)
        except OSError:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ── core model ─────────────────────────────────────────────
    def _entry(self, chunk_id):
        return self._data.get(chunk_id, {
            "strength": cfg.ACCESS_DEFAULT,
            "retrieval_count": 0,
            "last_retrieved": None,
        })

    @staticmethod
    def _days_since(iso_str):
        if not iso_str:
            return None
        try:
            then = datetime.fromisoformat(iso_str).date()
            return max(0, (date.today() - then).days)
        except ValueError:
            return None

    def current_strength(self, chunk_id, protected=False):
        """Strength right now: stored value decayed by idle time."""
        e = self._entry(chunk_id)
        s = e["strength"]
        idle = self._days_since(e["last_retrieved"])
        if idle:
            s *= math.exp(-cfg.ACCESS_DECAY_RATE * idle)
        if protected:
            s = max(s, cfg.PROTECTION_FLOOR)
        return max(0.0, min(1.0, s))

    def record_retrieval(self, chunk_ids, protected_lookup=None):
        """Call AFTER results are returned to the model. Strengthens each
        retrieved chunk with diminishing returns (can't exceed 1.0)."""
        protected_lookup = protected_lookup or {}
        with self._lock:
            # Two server instances share this sidecar (stdio for Desktop,
            # HTTP for the tunnel). Reload before applying so one process's
            # save doesn't clobber strengths the other recorded meanwhile.
            self._load()
            today = date.today().isoformat()
            for cid in chunk_ids:
                base = self.current_strength(
                    cid, protected=protected_lookup.get(cid, False))
                new = base + cfg.ACCESS_LEARNING_RATE * (1.0 - base)
                e = self._entry(cid)
                self._data[cid] = {
                    "strength": round(new, 6),
                    "retrieval_count": e["retrieval_count"] + 1,
                    "last_retrieved": today,
                }
            self._save()

    # ── serendipity ────────────────────────────────────────────
    def serendipity_sample(self, candidate_ids, k, protected_lookup=None):
        """Weighted-random draw favouring LOW-strength (nearly forgotten)
        chunks — the shoulder tap. weight = (1 - strength) + epsilon."""
        protected_lookup = protected_lookup or {}
        if not candidate_ids:
            return []
        weights = []
        for cid in candidate_ids:
            s = self.current_strength(
                cid, protected=protected_lookup.get(cid, False))
            weights.append((1.0 - s) + 0.05)
        k = min(k, len(candidate_ids))
        # random.choices can repeat; draw without replacement manually.
        chosen = []
        pool = list(zip(candidate_ids, weights))
        for _ in range(k):
            total = sum(w for _, w in pool)
            r = random.uniform(0, total)
            acc = 0.0
            for i, (cid, w) in enumerate(pool):
                acc += w
                if r <= acc:
                    chosen.append(cid)
                    pool.pop(i)
                    break
        return chosen
