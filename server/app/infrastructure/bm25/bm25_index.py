"""BM25 sparse retrieval index (Okapi BM25, k1=1.5, b=0.75).

Pure functions with no side effects at module level.  Uses an inverted index
for fast candidate filtering during search.
"""

from __future__ import annotations

import math

from domain.utils import content_hash
from infrastructure.bm25._tokenizer import tokenize


class BM25Index:
    """Minimal BM25 index that can be serialized to/from dict.

    Stores content hashes alongside texts for hybrid search merge.
    Uses an inverted index for fast candidate filtering during search.
    """

    def __init__(
        self,
        texts: list[str],
        hashes: list[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
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
        """Return (doc_index, score) pairs sorted by descending score."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

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
    def from_dict(cls, data: dict) -> BM25Index:
        return cls(
            texts=data["texts"],
            hashes=data.get("hashes"),
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
        )

    def add_text(self, text: str, text_hash: str | None = None) -> int:
        """Add a new text to the index. Returns the index of the new text."""
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
            if tf[t] == 1:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
            self.inverted_index.setdefault(t, set()).add(idx)
        self.token_freqs.append(tf)

        total_len = self.avgdl * self.n_docs + doc_len
        self.n_docs += 1
        self.avgdl = total_len / self.n_docs if self.n_docs > 0 else 1.0

        return idx

    def replace_text(self, index: int, new_text: str, new_hash: str | None = None) -> None:
        """Replace text at given index. Updates all BM25 statistics."""
        if index < 0 or index >= self.n_docs:
            raise IndexError(f"Index {index} out of range [0, {self.n_docs})")

        old_text = self.texts[index]
        old_tokens = tokenize(old_text)
        new_tokens = tokenize(new_text)

        old_tf = self.token_freqs[index]
        for t in old_tokens:
            if old_tf.get(t, 0) > 0:
                if old_tf[t] == 1 and t in self.doc_freq:
                    self.doc_freq[t] -= 1
                    if self.doc_freq[t] <= 0:
                        del self.doc_freq[t]
                if t in self.inverted_index:
                    self.inverted_index[t].discard(index)
                    if not self.inverted_index[t]:
                        del self.inverted_index[t]

        self.texts[index] = new_text
        self.hashes[index] = new_hash or content_hash(new_text)

        new_tf: dict[str, int] = {}
        for t in new_tokens:
            new_tf[t] = new_tf.get(t, 0) + 1
            if new_tf[t] == 1:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
            self.inverted_index.setdefault(t, set()).add(index)
        self.token_freqs[index] = new_tf

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
        """Remove text at given index. Updates all BM25 statistics."""
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
