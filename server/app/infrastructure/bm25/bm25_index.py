"""BM25 sparse retrieval index (Okapi BM25, k1=1.5, b=0.75).

Pure functions with no side effects at module level.  Uses an inverted index
for fast candidate filtering during search.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

from domain.shared.hashing import content_hash
from infrastructure.bm25.tokenizer import tokenize

log = logging.getLogger("default")


class BM25Index:
    """Minimal BM25 index that can be serialized to/from dict.

    Stores content hashes alongside texts for hybrid search merge.
    Uses an inverted index for fast candidate filtering during search.
    Optionally stores per-document ACL metadata for pre-filtering.
    """

    def __init__(
        self,
        texts: list[str],
        hashes: list[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        doc_visibility: list[str | None] | None = None,
        doc_owner_id: list[int | None] | None = None,
        doc_group_id: list[int | None] | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.texts = texts
        self.hashes = hashes or [content_hash(t) for t in texts]
        self._hash_to_idx: dict[str, int] = {h: i for i, h in enumerate(self.hashes)}
        self.n_docs = len(texts)
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.token_freqs: list[dict[str, int]] = []
        self.doc_freq: dict[str, int] = {}
        self.inverted_index: dict[str, set[int]] = {}
        self.doc_visibility: list[str | None] = doc_visibility or [None] * len(texts)
        self.doc_owner_id: list[int | None] = doc_owner_id or [None] * len(texts)
        self.doc_group_id: list[int | None] = doc_group_id or [None] * len(texts)
        self.last_survival_ratio: float | None = None
        self.build()

    def build(self) -> None:
        self.doc_freq.clear()
        self.token_freqs.clear()
        self.doc_lens.clear()
        self.inverted_index.clear()
        self._hash_to_idx = {h: i for i, h in enumerate(self.hashes)}

        total_len = 0

        for idx, text in enumerate(self.texts):
            tokens = tokenize(text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)

            self.token_freqs.append(self.index_tokens(tokens, idx))

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

    def _doc_matches_acl(
        self,
        doc_idx: int,
        visibility_conditions: list,
        user_id: int,
        user_group_ids: list[int],
    ) -> bool:
        """Check if doc at doc_idx matches any VisibilityCondition."""
        vis = self.doc_visibility[doc_idx]

        if vis is None:
            return False

        owner = self.doc_owner_id[doc_idx]
        group = self.doc_group_id[doc_idx]

        for cond in visibility_conditions:
            if cond.visibility.value != vis:
                continue

            if cond.owner_match == "self" and owner != user_id:
                continue

            if cond.owner_match == "assigned":
                if owner is None or cond.owner_ids is None or owner not in cond.owner_ids:
                    continue

            if cond.group_match:
                effective_groups = cond.group_ids if cond.group_ids is not None else user_group_ids

                if group is None or group not in effective_groups:
                    continue

            return True

        return False

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

    def search_with_hashes(
        self,
        query: str,
        k: int = 25,
        *,
        visibility_conditions: list,
        user_id: int,
        user_group_ids: list[int],
    ) -> list[tuple[str, float]]:
        """Return ACL-filtered (content_hash, score) pairs sorted by descending score."""
        if visibility_conditions is None or user_id is None or user_group_ids is None:
            raise RuntimeError(
                "search_with_hashes() requires visibility_conditions, user_id and user_group_ids."
            )

        self.last_survival_ratio = None
        q_tokens = tokenize(query)

        if not q_tokens:
            return []

        candidate_indices: set[int] = set()

        for t in q_tokens:
            posting = self.inverted_index.get(t)

            if posting:
                candidate_indices.update(posting)

        before_count = len(candidate_indices)
        candidate_indices = {
            i
            for i in candidate_indices
            if self._doc_matches_acl(i, visibility_conditions, user_id, user_group_ids)
        }

        if before_count > 0:
            self.last_survival_ratio = len(candidate_indices) / before_count

        if not candidate_indices:
            return []

        scored = [(i, self.score(q_tokens, i)) for i in candidate_indices]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self.hashes[idx], score) for idx, score in scored[:k]]

    def to_dict(self) -> dict:
        d: dict = {
            "k1": self.k1,
            "b": self.b,
            "texts": self.texts,
            "hashes": self.hashes,
        }
        has_acl = any(v is not None for v in self.doc_visibility)

        if has_acl:
            d["doc_visibility"] = self.doc_visibility
            d["doc_owner_id"] = self.doc_owner_id
            d["doc_group_id"] = self.doc_group_id

        return d

    @classmethod
    def from_dict(cls, data: dict) -> BM25Index:
        return cls(
            texts=data["texts"],
            hashes=data.get("hashes"),
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
            doc_visibility=data.get("doc_visibility"),
            doc_owner_id=data.get("doc_owner_id"),
            doc_group_id=data.get("doc_group_id"),
        )

    def find_by_hash(self, target_hash: str) -> int | None:
        """O(1) lookup by content hash."""
        return self._hash_to_idx.get(target_hash)

    def add_text(
        self,
        text: str,
        text_hash: str | None = None,
        visibility: str | None = None,
        owner_id: int | None = None,
        group_id: int | None = None,
    ) -> int:
        """Add a new text to the index. Returns the index of the new text."""
        idx = self.n_docs
        self.texts.append(text)
        h = text_hash or content_hash(text)
        self.hashes.append(h)
        self._hash_to_idx[h] = idx

        tokens = tokenize(text)
        doc_len = len(tokens)
        self.doc_lens.append(doc_len)

        self.token_freqs.append(self.index_tokens(tokens, idx))

        self.doc_visibility.append(visibility)
        self.doc_owner_id.append(owner_id)
        self.doc_group_id.append(group_id)

        total_len = self.avgdl * self.n_docs + doc_len
        self.n_docs += 1
        self.avgdl = total_len / self.n_docs if self.n_docs > 0 else 1.0

        return idx

    def replace_text(
        self,
        index: int,
        new_text: str,
        new_hash: str | None = None,
        visibility: str | None = None,
        owner_id: int | None = None,
        group_id: int | None = None,
    ) -> None:
        """Replace text at given index. Updates all BM25 statistics."""
        if index < 0 or index >= self.n_docs:
            raise IndexError(f"Index {index} out of range [0, {self.n_docs})")

        old_tokens = tokenize(self.texts[index])
        new_tokens = tokenize(new_text)

        self.remove_tokens(index)

        self.texts[index] = new_text
        old_hash = self.hashes[index]
        new_hash_val = new_hash or content_hash(new_text)
        self.hashes[index] = new_hash_val
        self._hash_to_idx.pop(old_hash, None)
        self._hash_to_idx[new_hash_val] = index

        self.token_freqs[index] = self.index_tokens(new_tokens, index)

        old_len = len(old_tokens)
        new_len = len(new_tokens)
        self.doc_lens[index] = new_len
        total_len = self.avgdl * self.n_docs - old_len + new_len
        self.avgdl = total_len / self.n_docs if self.n_docs > 0 else 1.0

        if visibility is not None:
            self.doc_visibility[index] = visibility

        if owner_id is not None:
            self.doc_owner_id[index] = owner_id

        if group_id is not None:
            self.doc_group_id[index] = group_id

    def index_tokens(self, tokens: list[str], index: int) -> dict[str, int]:
        frequencies = dict(Counter(tokens))

        for term in frequencies:
            self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
            self.inverted_index.setdefault(term, set()).add(index)

        return frequencies

    def remove_tokens(self, index: int) -> None:
        for term in self.token_freqs[index]:
            self.doc_freq[term] -= 1

            if self.doc_freq[term] == 0:
                del self.doc_freq[term]

            posting = self.inverted_index[term]
            posting.discard(index)

            if not posting:
                del self.inverted_index[term]

    @staticmethod
    def remove_from_lists(
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

    def shift_inverted_indices(self, removed_index: int) -> None:
        # Compact document positions require shifting only postings after the deletion.
        if removed_index == self.n_docs:
            return

        for term, posting in self.inverted_index.items():
            if any(position > removed_index for position in posting):
                self.inverted_index[term] = {
                    position - 1 if position > removed_index else position for position in posting
                }

    def remove_text(self, index: int) -> None:
        """Remove text at given index. Updates all BM25 statistics."""
        if index < 0 or index >= self.n_docs:
            raise IndexError(f"Index {index} out of range [0, {self.n_docs})")

        self.remove_tokens(index)
        removed_len = self.remove_from_lists(self.texts, self.hashes, self.doc_lens, self.token_freqs, index)

        del self.doc_visibility[index]
        del self.doc_owner_id[index]
        del self.doc_group_id[index]

        self.n_docs -= 1
        total_len = self.avgdl * (self.n_docs + 1) - removed_len
        self.avgdl = total_len / self.n_docs if self.n_docs > 0 else 1.0

        self.shift_inverted_indices(index)
        self._hash_to_idx = {h: i for i, h in enumerate(self.hashes)}
