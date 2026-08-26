"""BM25 sparse retrieval and Reciprocal Rank Fusion (RRF) merge for hybrid search.

Pure functions with no side effects at module level.  The BM25 implementation
is written from scratch with zero external dependencies beyond stdlib.  Includes
a lightweight Russian suffix stemmer for better sparse recall on Cyrillic text.
"""

import json
import logging
import math
import re
from pathlib import Path

log = logging.getLogger("default")


# Re-export from domain for backward compatibility — canonical implementation lives in domain/utils.py
from domain.utils import content_hash  # noqa: E402, F401

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
    Uses an inverted index for fast candidate filtering during search.
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
        self.inverted_index: dict[str, set[int]] = {}
        self._build()

    def _build(self) -> None:
        self.doc_freq.clear()
        self.token_freqs.clear()
        self.doc_lens.clear()
        self.inverted_index.clear()

        total_len = 0
        for idx, text in enumerate(self.texts):
            tokens = tokenize(text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)

            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
                self.doc_freq[t] = self.doc_freq.get(t, 0) + (1 if tf[t] == 1 else 0)
                self.inverted_index.setdefault(t, set()).add(idx)
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
        """Return (doc_index, score) pairs sorted by descending score.

        Uses inverted index to only score documents containing at least one query token,
        avoiding a full scan of all N documents.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        # Collect candidate doc indices from inverted index (union of posting lists)
        candidate_indices: set[int] = set()
        for t in q_tokens:
            posting = self.inverted_index.get(t)
            if posting:
                candidate_indices.update(posting)

        if not candidate_indices:
            return []

        scored = [(i, self.score(q_tokens, i)) for i in candidate_indices]
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

    def add_text(self, text: str, text_hash: str | None = None) -> int:
        """Add a new text to the index. Returns the index of the new text.

        Note: doc_freq and avgdl are approximated during incremental updates.
        Consider periodic full rebuild for optimal ranking.
        """
        idx = self.n_docs
        self.texts.append(text)
        h = text_hash or content_hash(text)
        self.hashes.append(h)

        tokens = tokenize(text)
        doc_len = len(tokens)
        self.doc_lens.append(doc_len)

        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
            # Update doc_freq: increment only if this is the first occurrence in this doc
            if tf[t] == 1:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
            # Update inverted_index
            self.inverted_index.setdefault(t, set()).add(idx)
        self.token_freqs.append(tf)

        # Update avgdl incrementally
        total_len = self.avgdl * self.n_docs + doc_len
        self.n_docs += 1
        self.avgdl = total_len / self.n_docs if self.n_docs > 0 else 1.0

        return idx

    def replace_text(self, index: int, new_text: str, new_hash: str | None = None) -> None:
        """Replace text at given index. Updates all BM25 statistics.

        Note: doc_freq and avgdl are approximated during incremental updates.
        Consider periodic full rebuild for optimal ranking.
        """
        if index < 0 or index >= self.n_docs:
            raise IndexError(f"Index {index} out of range [0, {self.n_docs})")

        old_text = self.texts[index]
        old_tokens = tokenize(old_text)
        new_tokens = tokenize(new_text)

        # Remove old tokens from doc_freq and inverted_index
        old_tf = self.token_freqs[index]
        for t in old_tokens:
            if old_tf.get(t, 0) > 0:
                # Decrement doc_freq if this was the only occurrence
                if old_tf[t] == 1 and t in self.doc_freq:
                    self.doc_freq[t] -= 1
                    if self.doc_freq[t] <= 0:
                        del self.doc_freq[t]
                # Remove from inverted_index
                if t in self.inverted_index:
                    self.inverted_index[t].discard(index)
                    if not self.inverted_index[t]:
                        del self.inverted_index[t]

        # Update texts and hashes
        self.texts[index] = new_text
        self.hashes[index] = new_hash or content_hash(new_text)

        # Add new tokens to doc_freq and inverted_index
        new_tf: dict[str, int] = {}
        for t in new_tokens:
            new_tf[t] = new_tf.get(t, 0) + 1
            if new_tf[t] == 1:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
            self.inverted_index.setdefault(t, set()).add(index)
        self.token_freqs[index] = new_tf

        # Update doc_lens and avgdl
        old_len = len(old_tokens)
        new_len = len(new_tokens)
        self.doc_lens[index] = new_len
        total_len = self.avgdl * self.n_docs - old_len + new_len
        self.avgdl = total_len / self.n_docs if self.n_docs > 0 else 1.0

    @staticmethod
    def _remove_old_tokens(
        doc_freq: dict[str, int],
        inverted_index: dict[str, set[int]],
        old_tokens: list[str],
        old_tf: dict[str, int],
        index: int,
    ) -> None:
        """Remove old document's tokens from doc_freq and inverted_index."""
        for t in old_tokens:
            if old_tf.get(t, 0) > 0:
                if old_tf[t] == 1 and t in doc_freq:
                    doc_freq[t] -= 1
                    if doc_freq[t] <= 0:
                        del doc_freq[t]
                if t in inverted_index:
                    inverted_index[t].discard(index)
                    if not inverted_index[t]:
                        del inverted_index[t]

    @staticmethod
    def _remove_from_lists(
        texts: list[str],
        hashes: list[str],
        doc_lens: list[int],
        token_freqs: list[dict[str, int]],
        index: int,
    ) -> int:
        """Delete entries at index from texts, hashes, doc_lens, token_freqs. Returns removed_len."""
        removed_len = doc_lens[index]
        del texts[index]
        del hashes[index]
        del doc_lens[index]
        del token_freqs[index]
        return removed_len

    @staticmethod
    def _rebuild_inverted_indices(
        inverted_index: dict[str, set[int]], removed_index: int
    ) -> dict[str, set[int]]:
        """Shift down all inverted_index entries whose indices exceed removed_index."""
        new_inverted: dict[str, set[int]] = {}
        for t, posting in inverted_index.items():
            new_posting = set()
            for old_idx in posting:
                if old_idx < removed_index:
                    new_posting.add(old_idx)
                elif old_idx > removed_index:
                    new_posting.add(old_idx - 1)
            if new_posting:
                new_inverted[t] = new_posting
        return new_inverted

    def remove_text(self, index: int) -> None:
        """Remove text at given index. Updates all BM25 statistics.

        Note: doc_freq and avgdl are approximated during incremental updates.
        Consider periodic full rebuild for optimal ranking.
        """
        if index < 0 or index >= self.n_docs:
            raise IndexError(f"Index {index} out of range [0, {self.n_docs})")

        old_tokens = tokenize(self.texts[index])
        old_tf = self.token_freqs[index]

        self._remove_old_tokens(self.doc_freq, self.inverted_index, old_tokens, old_tf, index)
        removed_len = self._remove_from_lists(self.texts, self.hashes, self.doc_lens, self.token_freqs, index)

        self.n_docs -= 1
        total_len = self.avgdl * (self.n_docs + 1) - removed_len
        self.avgdl = total_len / self.n_docs if self.n_docs > 0 else 1.0

        self.inverted_index = self._rebuild_inverted_indices(self.inverted_index, index)


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
