"""
infrastructure/ml/hybrid.py — BM25 sparse retrieval + RRF merge for hybrid search.

Pure functions, no side effects at module level.
BM25 implementation from scratch — zero external dependencies beyond stdlib.
Includes lightweight Russian suffix stemmer for better sparse recall.
"""

import hashlib
import json
import logging
import math
import re
from pathlib import Path

log = logging.getLogger("default")


def content_hash(text: str) -> str:
    """Deterministic short hash for merge key between dense and sparse results."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Lightweight Russian stemmer (suffix stripping)
# ---------------------------------------------------------------------------

# Common Russian suffixes to strip for better sparse recall.
# Ordered by length (longest first) to avoid partial matches.
_RU_SUFFIXES = [
    "ости",
    "ость",
    "ений",
    "ение",
    "ания",
    "ями",
    "ого",
    "ать",
    "ить",
    "ыть",
    "ять",
    "ути",
    "яти",
    "ей",
    "ой",
    "ий",
    "ый",
    "ая",
    "яя",
    "ое",
    "ее",
    "ие",
    "ые",
    "ов",
    "ев",
    "ам",
    "ям",
    "ом",
    "ем",
    "ах",
    "ях",
    "ки",
    "ка",
    "ик",
    "ов",
    "ев",
    "ые",
    "ие",
    "ы",
    "и",
    "у",
    "ю",
    "я",
    "е",
    "а",
]

# Sort by length descending for greedy matching
_RU_SUFFIXES.sort(key=len, reverse=True)


def _stem_russian(word: str) -> str:
    """Strip common Russian suffixes to normalize word forms.

    This is a simple heuristic stemmer — not as accurate as pymorphy2,
    but zero dependencies and very fast.
    """
    if len(word) < 5:  # Too short to stem meaningfully
        return word

    for suffix in _RU_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _stem_token(token: str) -> str:
    """Apply stemming to a single token."""
    # Stem Latin tokens (basic English suffix stripping)
    if token.isascii():
        for suffix in ["tion", "sion", "ment", "ness", "able", "ible", "ful", "less", "ous", "ive"]:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token

    # Stem Russian tokens
    if re.match(r"[а-яё]", token):
        return _stem_russian(token)

    return token


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]{2,}", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, stem tokens. 2+ char tokens only."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [_stem_token(t) for t in tokens]


def tokenize_raw(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric without stemming. For indexing."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25 (Okapi BM25, k1=1.5, b=0.75)
# ---------------------------------------------------------------------------


class BM25Index:
    """Minimal BM25 index that can be serialized to/from dict.

    Stores content hashes alongside texts for hybrid search merge.
    """

    def __init__(self, texts: list[str], hashes: list[str] | None = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.texts = texts
        self.hashes = hashes or [content_hash(t) for t in texts]
        self.n_docs = len(texts)
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.token_freqs: list[dict[str, int]] = []
        self.doc_freq: dict[str, int] = {}
        self._build()

    def _build(self) -> None:
        self.doc_freq.clear()
        self.token_freqs.clear()
        self.doc_lens.clear()

        total_len = 0
        for text in self.texts:
            tokens = tokenize(text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)

            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
                self.doc_freq[t] = self.doc_freq.get(t, 0) + (1 if tf[t] == 1 else 0)
            self.token_freqs.append(tf)

        self.avgdl = total_len / self.n_docs if self.n_docs > 0 else 1.0

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        tf = self.token_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]
        for t in query_tokens:
            if t not in tf:
                continue
            term_freq = tf[t]
            idf = self._idf(t)
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, k: int = 25) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs sorted by descending score."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scored = [(i, self.score(q_tokens, i)) for i in range(self.n_docs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def search_with_hashes(self, query: str, k: int = 25) -> list[tuple[str, float]]:
        """Return (content_hash, score) pairs sorted by descending score."""
        results = self.search(query, k)
        return [(self.hashes[idx], score) for idx, score in results]

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "texts": self.texts,
            "hashes": self.hashes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        return cls(
            texts=data["texts"],
            hashes=data.get("hashes"),
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_bm25_index(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8")
    log.info("BM25 index saved: %d docs -> %s", index.n_docs, path)


def load_bm25_index(path: Path) -> BM25Index | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = BM25Index.from_dict(data)
        log.info("BM25 index loaded: %d docs from %s", idx.n_docs, path)
        return idx
    except Exception as e:
        log.warning("Failed to load BM25 index from %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------


def rrf_merge(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[str]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Takes (content_hash, score) pairs from each source.
    Returns merged list of content hashes sorted by descending RRF score.
    k=60 is the standard constant from the original RRF paper.
    """
    rrf_scores: dict[str, float] = {}

    for rank, (h, _score) in enumerate(dense_results):
        rrf_scores[h] = rrf_scores.get(h, 0.0) + dense_weight / (k + rank + 1)

    for rank, (h, _score) in enumerate(sparse_results):
        rrf_scores[h] = rrf_scores.get(h, 0.0) + sparse_weight / (k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [h for h, _score in merged]


# ---------------------------------------------------------------------------
# Text index for BM25 (stored alongside the index for LangChain doc mapping)
# ---------------------------------------------------------------------------


def save_text_index(texts: list[str], path: Path) -> None:
    """Save raw chunk texts so BM25 index can be rebuilt on load."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")


def load_text_index(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
