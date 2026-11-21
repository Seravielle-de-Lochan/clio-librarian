"""
Clio v0.2 — ranking & memory-dynamics configuration.
All tunables in one place. Change, restart server, behaviour shifts.

Adapted from the Claran hybrid-ranking design (shared with the companion
community, July 2026) to fit Clio's document-corpus architecture:
our chunks come from curated vault documents, not authored snippets,
so 'emotional_weight' metadata is replaced by TYPE_WEIGHTS and
'protected' is derived from path/tag patterns.
"""

# ── Search flow ────────────────────────────────────────────────
SEARCH_CANDIDATES = 50   # oversample from Chroma before re-ranking
SEARCH_RESULTS = 6       # final results returned (matches v0.1 default)

# ── Hybrid ranking weights (must sum ~1.0) ─────────────────────
W_SEMANTIC = 0.45        # cosine similarity from vector search
W_KEYWORD = 0.13         # literal term overlap (names, exact phrases)
W_RECENCY = 0.14         # exponential decay on document date
W_ACCESS = 0.12          # Bjork dual-strength (consolidation/fading)
W_TYPE = 0.10            # document-type importance (see TYPE_WEIGHTS)
W_TAG = 0.06             # tag overlap between query terms and chunk tags

# ── Recency ────────────────────────────────────────────────────
RECENCY_LAMBDA = 0.015   # gentler than Claran's 0.03: our corpus is an
                         # archive, not a chat log — 30d≈0.64, 90d≈0.26,
                         # 365d≈0.004. Old memories fade, never vanish.
RECENCY_MISSING = 0.15   # neutral score for date-blind chunks
                         # (notes without frontmatter dates)

# ── Bjork access strength ──────────────────────────────────────
ACCESS_LEARNING_RATE = 0.3    # strength += rate * (1 - strength) on retrieval
ACCESS_DECAY_RATE = 0.05      # strength *= e^(-rate * days_idle)
ACCESS_DEFAULT = 0.10         # starting strength for never-retrieved chunks
PROTECTION_FLOOR = 0.30       # protected chunks never decay below this

# ── Protection patterns ────────────────────────────────────────
# A chunk is 'protected' if its source path contains any of these
# substrings OR its tags include any of the listed tags.
# EDIT THESE for your own vault: the notes that should never fade.
PROTECTED_PATH_PATTERNS = [
    "Core Documents",
    "Core_Documents",
    "Monthly",                   # monthly condensations
]
PROTECTED_TAGS = [
    "milestone", "pinned", "monthly-summary",
]

# ── Type weights (our substitute for emotional_weight) ─────────
# Keyed on the 'type' frontmatter field; unknown types get TYPE_DEFAULT.
# EDIT THESE to match the types your notes use.
TYPE_WEIGHTS = {
    "journal": 1.00,             # first-person writing — highest
    "letter": 0.90,
    "summary": 0.85,
    "monthly-summary": 0.80,
    "deployment-brief": 0.35,
    "project-research": 0.35,
    "guide": 0.30,
}
TYPE_DEFAULT = 0.50

# ── Superseded / correction handling ───────────────────────────
SUPERSEDED_MULTIPLIER = 0.1   # corrected chunks: heavily demoted, not hidden

# ── memory_start orientation ───────────────────────────────────
START_RECENT = 5         # latest chunks by document date
START_SERENDIPITY = 3    # weighted-random pulls toward the nearly-forgotten
START_ACTIVE = 4         # high-strength + recent = active threads
SERENDIPITY_POOL = 400   # sample pool size for serendipity draw
