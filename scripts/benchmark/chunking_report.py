"""Offline chunking report: sizes, fragment reachability and retrieval recall without an LLM.

Splits the benchmark corpus with the current code, then measures:

- chunk sizes in bge-m3 tokens (avg/p50/p95, share < 50 tokens, prefix share);
- fragment reachability: labelled fragments that lie whole inside one chunk;
- BM25 recall@5/@10 and MRR with the production sparse encoder and Qdrant IDF (free, local);
- with ``--dense``: bge-m3 recall via DeepInfra (embeddings cached by text hash,
  so only changed chunks are paid for) and the production weighted RRF hybrid of BM25 + dense;
- with ``--rerank`` (implies ``--dense``): the production reranker over the hybrid
  candidates, with its score filters (scores cached by query + text hash);
- with ``--contextualize``: plan 4.3 LLM notes (OpenRouter, the production auxiliary
  model) are added to dense inputs; notes are cached in ``data/chunking_cache/notes.sqlite``.

No generation and no judge: this is the cheap per-change check of the chunking
plan. The full online benchmark is run only at the end of a phase.

Dense text includes parent scope; BM25 indexes source bodies without the context prefix. Fragments are
matched against the chunk as ``matches_fragment`` does online. Search depths, RRF constant
and weights (including the exact-reference sparse boost), the rerank budget and its
thresholds are read from the production config sections. Provision diversity
reordering, ACL and temporal filters are not reproduced, so absolute numbers are a proxy;
compare runs of this script with each other, not with the online table.

    task chunking:report
    task chunking:report -- --rerank --out data/chunking_reports/prod-proxy.json
    task chunking:report -- --dense --baseline data/chunking_reports/v1.json --out data/chunking_reports/phase1.json
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import date
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.documents import Document
import numpy as np
from tokenizers import Tokenizer

from application.indexing.contextualization import ChunkContextualization
from application.search import HybridRetriever
from composition.domain_packs import PACK_REGISTRATIONS, selected_pack
from config.sections.ml import MLProviderSection
from config.sections.rag import HybridSearchSection, RankingSection, RetrievalSection
from domain.benchmark.evaluation import matches_fragment, source_name
from domain.benchmark.question import evaluation_annotations
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.ingestion.raw_document import RawDocument
from domain.rag.classification import has_exact_reference
from domain.rag.evidence.context import content_with_parent_scope
from infrastructure.domain_profile import register_pack_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.clients.inference_clients import RerankerClient
from infrastructure.ml.ingestion.llm_contextualizer import LLMChunkContextualizer
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter
from infrastructure.ml.rag.rag_reranking import reranker_text
from infrastructure.ml.tokenization import build_token_counter
from infrastructure.repositories.vector.sparse_vectors import encode_sparse


if TYPE_CHECKING:
    from domain.ingestion.ports import TokenCounter


log = logging.getLogger("chunking_report")

ROOT = Path(__file__).resolve().parents[2]
# Production retrieval defaults (standard breadth); runtime overrides are not applied.
HYBRID = HybridSearchSection()
RETRIEVAL = RetrievalSection()
RANKING = RankingSection()
ML = MLProviderSection()
DENSE_DEPTH = RETRIEVAL.retriever_fetch_k
SPARSE_DEPTH = HYBRID.bm25_fetch_k
FUSED_DEPTH = RETRIEVAL.retriever_fetch_k
RERANK_TOP_N = RETRIEVAL.retriever_top_k_broad
RRF_K = HYBRID.rrf_k
SMALL_CHUNK_TOKENS = 50
EMBED_MODEL = "BAAI/bge-m3"
EMBED_URL = "https://api.deepinfra.com/v1/openai/embeddings"
EMBED_BATCH = 64
EMBED_RETRIES = 5
TOKENIZER_CANDIDATES = (
    ROOT / "data/tei_cache/bge-m3/tokenizer.json",
    *sorted(Path.home().glob(".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*/tokenizer.json")),
)
# Keys compared with the baseline; True means a higher value is better.
COMPARED = {
    "chunks": None,
    "avg_tokens": None,
    "p95_tokens": None,
    "share_under_50_tokens": False,
    "avg_prefix_share": False,
    "fragment_reachability": True,
}
RETRIEVAL_KEYS = ("recall_at_5", "recall_at_10", "mrr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, default=None, help="default: selected pack's bundled corpus")
    parser.add_argument(
        "--pack",
        choices=("generic", *PACK_REGISTRATIONS),
        default="belarus_law",
        help="document rules of the corpus's knowledge space (default: belarus_law)",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        action="append",
        default=None,
        help="directory of <domain>.json sets; repeatable (default: selected pack's sets)",
    )
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument(
        "--gate",
        type=float,
        default=None,
        metavar="TOLERANCE",
        help="fail when reachability or recall/MRR drops below the baseline by more than TOLERANCE",
    )
    parser.add_argument(
        "--thresholds", type=Path, default=None, help="pack acceptance JSON; failures exit nonzero"
    )
    parser.add_argument("--tokenizer", type=Path, default=None, help="bge-m3 tokenizer.json")
    parser.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    parser.add_argument("--label", default="working tree")
    parser.add_argument("--dense", action="store_true", help="add bge-m3 and hybrid recall (DeepInfra, paid)")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/chunking_cache/embeddings.sqlite")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="rerank hybrid candidates with the production reranker (DeepInfra, paid); implies --dense",
    )
    parser.add_argument("--rerank-cache", type=Path, default=ROOT / "data/chunking_cache/rerank.sqlite")
    parser.add_argument("--chunks-out", type=Path, default=None, help="dump chunks as JSON for inspection")
    parser.add_argument(
        "--contextualize",
        action="store_true",
        help="add plan 4.3 LLM notes to dense inputs (OpenRouter, paid)",
    )
    parser.add_argument("--notes-cache", type=Path, default=ROOT / "data/chunking_cache/notes.sqlite")
    args = parser.parse_args()

    # `task chunking:report` runs from server/; relative paths always mean the repository root.
    path_options = (
        "corpus", "baseline", "thresholds", "tokenizer", "out", "cache", "rerank_cache", "chunks_out", "notes_cache"
    )

    for name in path_options:
        value = getattr(args, name)

        if value is not None and not value.is_absolute():
            setattr(args, name, ROOT / value)

    args.dense = args.dense or args.rerank
    pack = selected_pack(args.pack)
    defaults = [path for path in pack.benchmark_sets if path.is_dir()]

    if args.corpus is None:
        args.corpus = next(
            (path.parent / "corpus" for path in pack.benchmark_sets if (path.parent / "corpus").is_dir()),
            ROOT / "data/chunking_corpus",
        )

    if args.pack == "belarus_law":
        defaults.insert(0, ROOT / "benchmarks/chunking")

        default_baseline = ROOT / "docs/quality/artifacts/chunking-baseline-v1.json"

        # The baseline artifact is local (gitignored); a fresh checkout runs without comparison.
        if args.baseline is None and default_baseline.exists():
            args.baseline = default_baseline

    args.datasets = [path if path.is_absolute() else ROOT / path for path in args.datasets or defaults]

    if not args.datasets:
        parser.error(f"No benchmark sets for {args.pack!r}; pass --datasets with labelled documents")

    if args.gate is not None and args.baseline is None:
        parser.error("--gate needs --baseline")

    if args.baseline is not None and not args.baseline.exists():
        message = f"Baseline not found: {args.baseline}"
        raise SystemExit(message)

    return args


# --- splitting -------------------------------------------------------------------------------


def build_pipeline(
    pack_key: str = "belarus_law",
    token_counter: TokenCounter | None = None,
) -> tuple[DomainProfileRegistry, DomainSettingsAdapter, Any, Any]:
    pack = selected_pack(pack_key)
    settings = DomainSettingsAdapter()
    values = {}

    for profile in pack.build_profiles(settings):
        values.update({(item.key, profile.key): item.value for item in profile.config_defaults()})

    settings.replace_snapshot(values)
    registry = DomainProfileRegistry()
    register_pack_profiles(registry, pack, settings)
    return (
        registry,
        settings,
        LangchainDocumentParser(registry, settings),
        LangchainDocumentSplitter(registry, settings, token_counter),
    )


class SqliteNoteCache:
    """Local note cache, so repeated report runs do not pay for the same notes."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS notes (key TEXT PRIMARY KEY, note TEXT NOT NULL)")

    async def get(self, key: str) -> str | None:
        row = self.db.execute("SELECT note FROM notes WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    async def set(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO notes VALUES (?, ?)", (key, value))
        self.db.commit()


# The report's paid mode always calls OpenRouter with a fixed fast model.
CONTEXTUALIZATION_MODEL = "meta-llama/llama-3.1-8b-instruct"
KEY_NAMES = ("LLM_API_KEY",)


def openrouter_key() -> str:
    """Read the key from the environment or server/.env.secrets; it is never printed."""
    for key_name in KEY_NAMES:
        if key := os.environ.get(key_name):
            return key

    secrets = ROOT / "server/.env.secrets"

    if secrets.exists():
        values = {}

        for line in secrets.read_text().splitlines():
            name, _, value = line.partition("=")
            values[name.strip()] = value.strip().strip("'\"")

        for key_name in KEY_NAMES:
            if values.get(key_name):
                return values[key_name]

    message = "LLM_API_KEY is not set (environment or server/.env.secrets)"
    raise SystemExit(message)


def build_contextualization(notes_cache: Path, tokenizer_path: Path) -> ChunkContextualization:
    from langchain_openai import ChatOpenAI  # noqa: PLC0415 - Optional paid mode of the report only.

    llm = ChatOpenAI(
        model=CONTEXTUALIZATION_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key(),
        temperature=0,
        max_tokens=200,
        max_retries=2,
        timeout=60,
    )
    return ChunkContextualization(
        LLMChunkContextualizer(lambda: llm, SqliteNoteCache(notes_cache), concurrency=8, timeout_seconds=60),
        build_token_counter(str(tokenizer_path)),
    )


def split_corpus(
    corpus: Path,
    contextualization: ChunkContextualization | None = None,
    *,
    pack_key: str = "belarus_law",
    token_counter: TokenCounter | None = None,
) -> list[dict[str, Any]]:
    """Split every file the way the loader does: one registry-classified domain per document."""
    registry, settings, parser, splitter = build_pipeline(pack_key, token_counter)
    files = sorted(path for path in corpus.iterdir() if path.is_file() and not path.name.startswith("."))

    if not files:
        message = f"No documents in {corpus}"
        raise SystemExit(message)

    documents = []

    for path in files:
        raw = parser.parse(path)
        domain = registry.classify("\n".join(item.page_content for item in raw), settings=settings).domain_key
        chunks = splitter.split(
            [RawDocument(item.page_content, dict(item.metadata)) for item in raw], domain=domain
        )

        if contextualization is not None:
            for chunk in chunks:
                chunk.metadata["source"] = path.name

            asyncio.run(contextualization.apply(chunks))

        def rerank_input(chunk: Any, source: str = path.name) -> str:
            metadata = {**chunk.metadata, "source": source}
            return reranker_text(Document(page_content=chunk.page_content, metadata=metadata))

        documents.append(
            {
                "file": path.name,
                "domain": domain,
                "chunks": [
                    {
                        "content": chunk.page_content,
                        "scoped_content": content_with_parent_scope(
                            Document(page_content=chunk.page_content, metadata=chunk.metadata)
                        ),
                        "prefix": (chunk.embed_text or chunk.page_content).removesuffix(chunk.page_content),
                        "index_text": chunk.embed_text or chunk.page_content,
                        "rerank_text": rerank_input(chunk),
                        "metadata": {
                            "source": path.name,
                            **{
                                key: chunk.metadata[key]
                                for key in (
                                    "section",
                                    "unit_kind",
                                    "content_type",
                                    "page",
                                    "pages",
                                    "page_start",
                                    "page_end",
                                )
                                if chunk.metadata.get(key) is not None
                            },
                        },
                    }
                    for chunk in chunks
                ],
            }
        )
        print(f"  {path.name[:70]}: {domain}, {len(chunks)} chunks", file=sys.stderr)

    return documents


# --- size and reachability -------------------------------------------------------------------


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def size_stats(chunks: list[dict[str, Any]], count: Any) -> dict[str, Any]:
    full = [count(chunk["index_text"]) for chunk in chunks]
    bodies = [count(chunk["content"]) for chunk in chunks]
    return {
        "chunks": len(chunks),
        "avg_tokens": round(statistics.mean(full), 1),
        "p50_tokens": percentile(full, 0.5),
        "p95_tokens": percentile(full, 0.95),
        "max_tokens": max(full),
        "share_under_50_tokens": round(sum(t < SMALL_CHUNK_TOKENS for t in full) / len(full), 4),
        "avg_body_tokens": round(statistics.mean(bodies), 1),
        "avg_prefix_share": round(1 - sum(bodies) / sum(full), 4),
    }


def labelled_questions(datasets: list[Path], domain: str) -> list[dict[str, Any]]:
    path = next((directory / f"{domain}.json" for directory in datasets if (directory / f"{domain}.json").exists()), None)

    if path is None:
        return []

    questions = []

    for question in json.loads(path.read_text()):
        fragments = (evaluation_annotations(question.get("annotations")) or {}).get(
            "expected_fragments"
        ) or []

        if fragments:
            questions.append(
                {"case_id": question["case_id"], "question": question["question"], "fragments": fragments}
            )

    return questions


def reachability(questions: list[dict[str, Any]], by_source: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = reachable = 0
    unreachable = set()

    for question in questions:
        for fragment in question["fragments"]:
            first = fragment["alternatives"][0] if "alternatives" in fragment else fragment
            doc = by_source.get(source_name(first["source"]))
            hit = doc is not None and any(matches_fragment(fragment, chunk) for chunk in doc["chunks"])
            total += 1
            reachable += hit

            if not hit:
                unreachable.add(question["case_id"])

    return {
        "fragments": total,
        "fragment_reachability": round(reachable / total, 4) if total else None,
        "unreachable_cases": sorted(unreachable),
    }


# --- retrieval -------------------------------------------------------------------------------


def retrieval_metrics(
    questions: list[dict[str, Any]], rankings: dict[str, list[int]], chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fragment-level recall and MRR, computed exactly as scripts for online runs do."""
    r5, r10, rr, misses = [], [], [], []

    for question in questions:
        retrieved = [chunks[index] for index in rankings[question["case_id"]]]
        ranks = [
            next((rank for rank, chunk in enumerate(retrieved, 1) if matches_fragment(fragment, chunk)), None)
            for fragment in question["fragments"]
        ]
        r5.append(sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks))
        r10.append(sum(rank is not None and rank <= 10 for rank in ranks) / len(ranks))
        found = [rank for rank in ranks if rank is not None]
        rr.append(1 / min(found) if found else 0.0)

        if any(rank is None or rank > 10 for rank in ranks):
            misses.append(question["case_id"])

    return {
        "questions": len(questions),
        "recall_at_5": round(statistics.mean(r5), 4),
        "recall_at_10": round(statistics.mean(r10), 4),
        "mrr": round(statistics.mean(rr), 4),
        "misses_at_10": misses,
    }


def bm25_rankings(questions: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Score with the production sparse encoder and Qdrant's IDF modifier."""
    postings: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for position, chunk in enumerate(chunks):
        vector = encode_sparse(chunk["content"])

        for term, weight in zip(vector.indices, vector.values, strict=True):
            postings[term].append((position, weight))

    total = len(chunks)
    rankings: dict[str, list[int]] = {}

    for question in questions:
        query = encode_sparse(question["question"], query=True)
        scores: dict[int, float] = defaultdict(float)

        for term, query_weight in zip(query.indices, query.values, strict=True):
            matches = postings.get(term, [])
            idf = math.log(1 + (total - len(matches) + 0.5) / (len(matches) + 0.5))

            for position, weight in matches:
                scores[position] += query_weight * idf * weight

        ranked = sorted(scores, key=lambda position: (-scores[position], position))
        rankings[question["case_id"]] = ranked[:SPARSE_DEPTH]

    return rankings


def rrf(question: str, dense: list[int], sparse: list[int]) -> list[int]:
    """Weighted RRF as ``rrf_merge``: the sparse weight is boosted for exact structural references."""
    sparse_weight = HYBRID.sparse_weight

    if has_exact_reference(question):
        sparse_weight *= RANKING.exact_ref_sparse_boost

    scores: dict[int, float] = {}

    for ranking, weight in ((dense, HYBRID.dense_weight), (sparse, sparse_weight)):
        for rank, index in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0.0) + weight / (RRF_K + rank)

    return sorted(scores, key=lambda index: -scores[index])[:FUSED_DEPTH]


class EmbeddingCache:
    """bge-m3 vectors keyed by sha256(model + text); unchanged chunks are never re-embedded."""

    def __init__(self, path: Path, api_key: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB)")
        self.api_key = api_key
        self.paid_tokens = 0

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(f"{EMBED_MODEL}\n{text}".encode()).hexdigest()

    def vectors(self, texts: list[str]) -> np.ndarray:
        keys = [self.key(text) for text in texts]
        known: dict[str, bytes] = {}

        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            known.update(
                self.db.execute(f"SELECT key, vector FROM vectors WHERE key IN ({placeholders})", batch)
            )

        missing = list(
            dict.fromkeys((key, text) for key, text in zip(keys, texts, strict=True) if key not in known)
        )
        cached = sum(key in known for key in keys)
        print(
            f"  embeddings: {cached}/{len(texts)} cached, {len(missing)} unique texts to embed",
            file=sys.stderr,
        )

        with httpx.Client(timeout=120) as client:
            for start in range(0, len(missing), EMBED_BATCH):
                batch = missing[start : start + EMBED_BATCH]
                for (key, _text), vector in zip(
                    batch, self.embed(client, [text for _key, text in batch]), strict=True
                ):
                    blob = np.asarray(vector, dtype=np.float32).tobytes()
                    known[key] = blob
                    self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)", (key, blob))
                self.db.commit()

        matrix = np.stack([np.frombuffer(known[key], dtype=np.float32) for key in keys])
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def embed(self, client: httpx.Client, texts: list[str]) -> list[list[float]]:
        for attempt in range(EMBED_RETRIES):
            response = client.post(
                EMBED_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": texts, "model": EMBED_MODEL, "encoding_format": "float"},
            )

            if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < EMBED_RETRIES:
                time.sleep(2**attempt)
                continue

            response.raise_for_status()
            payload = response.json()
            self.paid_tokens += int((payload.get("usage") or {}).get("prompt_tokens") or 0)
            return [item["embedding"] for item in sorted(payload["data"], key=lambda item: item["index"])]

        message = "unreachable"
        raise RuntimeError(message)


def secret(key_name: str) -> str:
    """Read the key from the environment or server/.env.secrets; it is never printed."""
    if key := os.environ.get(key_name):
        return key

    secrets = ROOT / "server/.env.secrets"

    if secrets.exists():
        for line in secrets.read_text().splitlines():
            name, _, value = line.partition("=")

            if name.strip() == key_name and value.strip():
                return value.strip().strip("'\"")

    message = f"{key_name} is not set (environment or server/.env.secrets)"
    raise SystemExit(message)


def dense_rankings(
    questions: list[dict[str, Any]], chunks: list[dict[str, Any]], cache: EmbeddingCache
) -> dict[str, list[int]]:
    chunk_matrix = cache.vectors([chunk["index_text"] for chunk in chunks])
    query_matrix = cache.vectors([question["question"] for question in questions])
    scores = query_matrix @ chunk_matrix.T
    top = np.argsort(-scores, axis=1)[:, :DENSE_DEPTH]
    return {question["case_id"]: [int(i) for i in row] for question, row in zip(questions, top, strict=True)}


class RerankCache:
    """Reranker scores keyed by sha256(model + query + text); repeated pairs are never re-scored."""

    def __init__(self, path: Path, api_key: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)")
        # RerankerClient builds requests and parses scores in the configured API format;
        # this synchronous script sends them over its own client.
        self.client = RerankerClient(ML.rerank_base_url, ML.rerank_model, ML.rerank_api_format, api_key)
        self.http = httpx.Client(timeout=60.0, follow_redirects=False)
        self.scored_pairs = 0

    @staticmethod
    def key(query: str, text: str) -> str:
        return hashlib.sha256(f"{ML.rerank_model}\n{query}\n{text}".encode()).hexdigest()

    def scores(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []

        keys = [self.key(query, text) for text in texts]
        placeholders = ",".join("?" * len(keys))
        known: dict[str, float] = dict(
            self.db.execute(f"SELECT key, score FROM scores WHERE key IN ({placeholders})", keys)
        )
        missing = list(dict.fromkeys((key, text) for key, text in zip(keys, texts, strict=True) if key not in known))

        if missing:
            for key, score in zip(
                [key for key, _ in missing], self.predict(query, [text for _, text in missing]), strict=True
            ):
                known[key] = score
                self.db.execute("INSERT OR REPLACE INTO scores VALUES (?, ?)", (key, score))

            self.db.commit()
            self.scored_pairs += len(missing)

        return [known[key] for key in keys]

    def predict(self, query: str, texts: list[str]) -> list[float]:
        for attempt in range(EMBED_RETRIES):
            try:
                pairs = [(query, text) for text in texts]
                return self.client.scores(self.http.post(**self.client.request(pairs)), len(pairs))
            except httpx.HTTPStatusError as error:
                retryable = error.response.status_code in {429, 500, 502, 503, 504}

                if not retryable or attempt + 1 == EMBED_RETRIES:
                    raise

            time.sleep(2**attempt)

        message = "unreachable"
        raise RuntimeError(message)


def rerank_rankings(
    questions: list[dict[str, Any]],
    candidates: dict[str, list[int]],
    chunks: list[dict[str, Any]],
    cache: RerankCache,
) -> dict[str, list[int]]:
    """Rerank hybrid candidates, apply the production score filters and keep the rerank budget."""
    retriever = HybridRetriever()
    rankings = {}

    for question in questions:
        positions = candidates[question["case_id"]]
        scores = cache.scores(question["question"], [chunks[index]["rerank_text"] for index in positions])
        scored = sorted(zip(positions, scores, strict=True), key=lambda item: -item[1])
        eligible = retriever.apply_rerank_filters(
            scored, min_score=RANKING.rerank_min_score, score_gap_ratio=RANKING.rerank_score_gap_ratio
        )
        rankings[question["case_id"]] = [index for index, _ in eligible[:RERANK_TOP_N]]

    return rankings


# --- report ----------------------------------------------------------------------------------


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer_path = args.tokenizer or next((path for path in TOKENIZER_CANDIDATES if path.exists()), None)

    if tokenizer_path is None:
        message = "bge-m3 tokenizer.json not found; pass --tokenizer"
        raise SystemExit(message)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    contextualization = (
        build_contextualization(args.notes_cache, tokenizer_path) if args.contextualize else None
    )
    # Chunk with the same tokenizer as the metrics: without it the splitter falls back to
    # embed_tokenizer_path, which is absent on CI and silently switches to a character estimate.
    documents = split_corpus(
        args.corpus,
        contextualization,
        pack_key=args.pack,
        token_counter=build_token_counter(str(tokenizer_path)),
    )

    if args.chunks_out:
        args.chunks_out.parent.mkdir(parents=True, exist_ok=True)
        args.chunks_out.write_text(json.dumps(documents, ensure_ascii=False))

    report: dict[str, Any] = {
        "label": args.label,
        "pack": args.pack,
        "date": date.today().isoformat(),
        "tokenizer": "BAAI/bge-m3",
        "retrieval": (
            f"proxy: BM25 (prod tokenizer, depth {SPARSE_DEPTH}) / bge-m3 cosine (depth {DENSE_DEPTH}) / "
            f"RRF k={RRF_K}, weights {HYBRID.dense_weight}/{HYBRID.sparse_weight} "
            f"(exact-ref sparse x{RANKING.exact_ref_sparse_boost}), fused depth {FUSED_DEPTH}"
            + (
                f" / {ML.rerank_model} top {RERANK_TOP_N}, min_score {RANKING.rerank_min_score}, "
                f"gap {RANKING.rerank_score_gap_ratio}"
                if args.rerank else ", no reranker"
            )
        ),
        "domains": {},
        "documents": [
            {"file": d["file"], "domain": d["domain"], **size_stats(d["chunks"], count)} for d in documents
        ],
    }
    by_source = {source_name(d["file"]): d for d in documents}
    all_chunks = [chunk for d in documents for chunk in d["chunks"]]
    domains = sorted({d["domain"] for d in documents})
    questions = {domain: labelled_questions(args.datasets, domain) for domain in domains}
    flat_questions = [q for items in questions.values() for q in items]

    # One index over the whole corpus, like production: other documents compete for the top-k.
    rankings = {"bm25": bm25_rankings(flat_questions, all_chunks)}
    cache = None

    if args.dense:
        cache = EmbeddingCache(args.cache, secret("EMBEDDING_API_KEY"))
        rankings["dense"] = dense_rankings(flat_questions, all_chunks, cache)
        rankings["hybrid"] = {
            q["case_id"]: rrf(q["question"], rankings["dense"][q["case_id"]], rankings["bm25"][q["case_id"]])
            for q in flat_questions
        }

    rerank_cache = None

    if args.rerank:
        rerank_cache = RerankCache(args.rerank_cache, secret("RERANK_API_KEY"))
        rankings["rerank"] = rerank_rankings(flat_questions, rankings["hybrid"], all_chunks, rerank_cache)

    report["overall"] = {
        "dataset_questions": len(flat_questions),
        **reachability(flat_questions, by_source),
        "retrieval": {
            mode: retrieval_metrics(flat_questions, ranking, all_chunks)
            for mode, ranking in rankings.items()
        } if flat_questions else {},
    }

    for domain in domains:
        items = [d for d in documents if d["domain"] == domain]
        stats = size_stats([chunk for d in items for chunk in d["chunks"]], count)
        stats["documents"] = len(items)
        stats["chunks_per_document_avg"] = round(stats["chunks"] / len(items), 1)

        if questions[domain]:
            stats["dataset_questions"] = len(questions[domain])
            stats.update(reachability(questions[domain], by_source))
            body_sources = {
                name: {
                    **document,
                    "chunks": [
                        {key: value for key, value in chunk.items() if key != "scoped_content"}
                        for chunk in document["chunks"]
                    ],
                }
                for name, document in by_source.items()
            }
            stats["body_fragment_reachability"] = reachability(questions[domain], body_sources)[
                "fragment_reachability"
            ]
            body_chunks = [
                {key: value for key, value in chunk.items() if key != "scoped_content"}
                for chunk in all_chunks
            ]
            stats["body_retrieval"] = {
                mode: retrieval_metrics(questions[domain], ranking, body_chunks)
                for mode, ranking in rankings.items()
            }
            stats["retrieval"] = {
                mode: retrieval_metrics(questions[domain], ranking, all_chunks)
                for mode, ranking in rankings.items()
            }

        report["domains"][domain] = stats

    if cache is not None:
        report["embedding_tokens_paid"] = cache.paid_tokens

    if rerank_cache is not None:
        report["rerank_pairs_paid"] = rerank_cache.scored_pairs

    return report


def delta(value: Any, base: Any, higher_is_better: bool | None) -> str:
    if base is None or value is None:
        return f"{value}"

    diff = value - base
    mark = ""

    if higher_is_better is not None and not math.isclose(diff, 0, abs_tol=1e-9):
        mark = " ✓" if (diff > 0) == higher_is_better else " ✗"

    return f"{value} ({diff:+.4g}{mark})"


def threshold_failures(name: str, stats: dict[str, Any], minimums: dict[str, Any]) -> list[str]:
    failures = []

    if not minimums:
        failures.append(f"{name}: no minimums")

    for metric, minimum in minimums.items():
        value = (
            stats.get("retrieval", {}).get("bm25", {}).get(metric.removeprefix("bm25."))
            if metric.startswith("bm25.") else stats.get(metric)
        )

        if (
            type(value) not in (int, float) or not math.isfinite(value)
            or type(minimum) not in (int, float) or not math.isfinite(minimum)
            or value < minimum
        ):
            failures.append(f"{name}.{metric}: {value!r} < required {minimum!r} (or missing)")

    return failures


def acceptance_failures(report: dict[str, Any], specification: dict[str, Any]) -> list[str]:
    """Check corpus classification and retrieval floors; absent measurements fail closed."""
    failures = []

    if report.get("pack") != specification.get("pack"):
        failures.append("Acceptance specification belongs to a different pack")

    actual_profiles = {doc["file"]: doc["domain"] for doc in report.get("documents", [])}

    if not specification.get("expected_profiles") or actual_profiles != specification["expected_profiles"]:
        failures.append("Corpus files or document classifications differ from expected_profiles")

    minimums = specification.get("minimums", {})

    if not minimums:
        failures.append("Acceptance specification has no minimums")

    measurements = [
        (domain, report.get("domains", {}).get(domain, {}), thresholds)
        for domain, thresholds in minimums.items()
    ]

    if "overall_minimums" in specification:
        measurements.append(("overall", report.get("overall", {}), specification["overall_minimums"]))

    for domain, stats, thresholds in measurements:
        failures.extend(threshold_failures(domain, stats, thresholds))

    return failures


def gate_failures(report: dict[str, Any], baseline: dict[str, Any], tolerance: float) -> list[str]:
    """Drops against the reference beyond the tolerance; a changed question set must refresh the reference."""
    failures = []
    sections = {**report.get("domains", {}), "overall": report.get("overall", {})}
    reference = {**baseline.get("domains", {}), "overall": baseline.get("overall", {})}

    for name, base in reference.items():
        stats = sections.get(name)

        if stats is None:
            failures.append(f"{name}: missing from the report")
            continue

        if stats.get("dataset_questions") != base.get("dataset_questions"):
            failures.append(
                f"{name}: {stats.get('dataset_questions')} questions, reference has {base.get('dataset_questions')}; "
                "refresh the reference in the same PR"
            )
            continue

        measured = [("fragment_reachability", stats.get("fragment_reachability"), base.get("fragment_reachability"))]

        for mode, metrics in base.get("retrieval", {}).items():
            current = stats.get("retrieval", {}).get(mode)

            if current is None:
                failures.append(f"{name}.{mode}: missing from the report")
                continue

            measured.extend((f"{mode}.{key}", current.get(key), metrics.get(key)) for key in RETRIEVAL_KEYS)

        failures.extend(
            f"{name}.{key}: {value!r} < reference {expected!r} - {tolerance}"
            for key, value, expected in measured
            if expected is not None and (value is None or value < expected - tolerance)
        )

    return failures


def print_comparison(report: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    base_domains = (baseline or {}).get("domains", {})
    print(f"\n{report['label']}" + (f"  vs  {baseline['label']}" if baseline else ""))

    for domain, stats in report["domains"].items():
        base = base_domains.get(domain, {})
        print(f"\n[{domain}]")

        for key, better in COMPARED.items():
            if key in stats:
                print(f"  {key:24} {delta(stats[key], base.get(key), better)}")

        for mode, metrics in stats.get("retrieval", {}).items():
            base_metrics = base.get("retrieval", {}).get(mode, {})
            values = "  ".join(
                f"{key}={delta(metrics[key], base_metrics.get(key), True)}" for key in RETRIEVAL_KEYS
            )
            print(f"  {mode:8} {values}")
            gained = sorted(set(base_metrics.get("misses_at_10", [])) - set(metrics["misses_at_10"]))
            lost = sorted(
                set(metrics["misses_at_10"]) - set(base_metrics.get("misses_at_10", metrics["misses_at_10"]))
            )

            if gained or lost:
                print(f"           found now: {gained or '-'}  lost: {lost or '-'}")

        if stats.get("unreachable_cases"):
            print(f"  unreachable: {stats['unreachable_cases']}")

    if "embedding_tokens_paid" in report:
        print(f"\nDeepInfra embedding tokens paid in this run: {report['embedding_tokens_paid']}")

    if "rerank_pairs_paid" in report:
        print(f"Reranker pairs paid in this run: {report['rerank_pairs_paid']}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    baseline = json.loads(args.baseline.read_text()) if args.baseline is not None else None

    if baseline is not None and baseline.get("pack", "belarus_law") != args.pack:
        message = "Baseline belongs to a different domain pack"
        raise SystemExit(message)

    report = build_report(args)
    failures = []

    if args.thresholds is not None:
        specification = json.loads(args.thresholds.read_text())
        failures = acceptance_failures(report, specification)
        report["acceptance"] = {
            "provenance": specification.get("provenance"),
            "status": specification.get("status"),
            "passed": not failures,
            "failures": failures,
        }

    print_comparison(report, baseline)

    if args.gate is not None and baseline is not None:
        gate = gate_failures(report, baseline, args.gate)
        failures.extend(gate)
        print(f"\nOffline gate vs {args.baseline.name}: {'FAIL' if gate else 'PASS'} (tolerance {args.gate})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"\nreport: {args.out}")

    if failures:
        raise SystemExit("Offline checks failed:\n" + "\n".join(failures))

    if args.thresholds is not None:
        print(f"\nAcceptance: PASS ({report['acceptance']['provenance']}, {report['acceptance']['status']})")


if __name__ == "__main__":
    main()
