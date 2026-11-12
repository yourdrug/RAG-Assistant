"""Offline chunking report: sizes, fragment reachability and retrieval recall without an LLM.

Splits the benchmark corpus with the current code, then measures:

- chunk sizes in bge-m3 tokens (avg/p50/p95, share < 50 tokens, prefix share);
- fragment reachability: labelled fragments that lie whole inside one chunk;
- BM25 recall@5/@10 and MRR with the production tokenizer (free, local);
- with ``--dense``: bge-m3 recall via DeepInfra (embeddings cached by text hash,
  so only changed chunks are paid for) and an RRF hybrid of BM25 + dense.

No generation and no judge: this is the cheap per-change check of the chunking
plan. The full online benchmark is run only at the end of a phase.

Index text equals production (``content_with_parent_scope``); fragments are
matched against the chunk as ``matches_fragment`` does online. Reranking, ACL
and temporal filters are not reproduced, so absolute numbers are a proxy;
compare runs of this script with each other, not with the online table.

    task chunking:report
    task chunking:report -- --dense --baseline data/chunking_reports/v1.json --out data/chunking_reports/phase1.json
"""

from __future__ import annotations

import argparse
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
from typing import Any

import httpx
from langchain_core.documents import Document
import numpy as np
from tokenizers import Tokenizer

from domain.benchmark.evaluation import matches_fragment, source_name
from domain.benchmark.question import evaluation_annotations
from domain.ingestion.profiles.catalog import SETTINGS_PROFILE_TYPES
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.ingestion.raw_document import RawDocument
from domain.rag.evidence.context import content_with_parent_scope
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.domain_profile import register_all_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter


log = logging.getLogger("chunking_report")

ROOT = Path(__file__).resolve().parents[2]
SEARCH_DEPTH = 50
RRF_K = 60
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
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/chunking_corpus")
    parser.add_argument("--datasets", type=Path, default=ROOT / "benchmarks/chunking")
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "docs/quality/artifacts/chunking-baseline-v1.json"
    )
    parser.add_argument("--tokenizer", type=Path, default=None, help="bge-m3 tokenizer.json")
    parser.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    parser.add_argument("--label", default="working tree")
    parser.add_argument("--dense", action="store_true", help="add bge-m3 and hybrid recall (DeepInfra, paid)")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/chunking_cache/embeddings.sqlite")
    parser.add_argument("--chunks-out", type=Path, default=None, help="dump chunks as JSON for inspection")
    args = parser.parse_args()

    # `task chunking:report` runs from server/; relative paths always mean the repository root.
    for name in ("corpus", "datasets", "baseline", "tokenizer", "out", "cache", "chunks_out"):
        value = getattr(args, name)

        if value is not None and not value.is_absolute():
            setattr(args, name, ROOT / value)

    if not args.baseline.exists():
        message = f"Baseline not found: {args.baseline}"
        raise SystemExit(message)

    return args


# --- splitting -------------------------------------------------------------------------------


def build_pipeline() -> tuple[DomainProfileRegistry, DomainSettingsAdapter, Any, Any]:
    settings = DomainSettingsAdapter()
    values = {}

    for profile_type in SETTINGS_PROFILE_TYPES:
        profile = profile_type(settings=settings)
        values.update({(item.key, profile.key): item.value for item in profile.config_defaults()})

    settings.replace_snapshot(values)
    registry = DomainProfileRegistry()
    register_all_profiles(registry, settings)
    return (
        registry,
        settings,
        LangchainDocumentParser(registry, settings),
        LangchainDocumentSplitter(registry, settings),
    )


def split_corpus(corpus: Path) -> list[dict[str, Any]]:
    """Split every file the way the loader does: one registry-classified domain per document."""
    registry, settings, parser, splitter = build_pipeline()
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
        documents.append(
            {
                "file": path.name,
                "domain": domain,
                "chunks": [
                    {
                        "content": chunk.page_content,
                        "prefix": chunk.metadata.get("context_prefix") or "",
                        "index_text": content_with_parent_scope(
                            Document(page_content=chunk.page_content, metadata=chunk.metadata)
                        ),
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
    full = [count(chunk["content"]) for chunk in chunks]
    bodies = [count(chunk["content"][len(chunk["prefix"]) :]) for chunk in chunks]
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


def labelled_questions(datasets: Path, domain: str) -> list[dict[str, Any]]:
    path = datasets / f"{domain}.json"

    if not path.exists():
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
    index = BM25Index([chunk["index_text"] for chunk in chunks])
    return {q["case_id"]: [i for i, _score in index.search(q["question"], k=SEARCH_DEPTH)] for q in questions}


def rrf(*rankings: list[int]) -> list[int]:
    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0.0) + 1 / (RRF_K + rank)

    return sorted(scores, key=lambda index: -scores[index])[:SEARCH_DEPTH]


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
            )  # noqa: S608

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


def deepinfra_key() -> str:
    """Read the key from the environment or server/.env.secrets; it is never printed."""
    if key := os.environ.get("DEEPINFRA_API_KEY"):
        return key

    secrets = ROOT / "server/.env.secrets"

    if secrets.exists():
        for line in secrets.read_text().splitlines():
            name, _, value = line.partition("=")

            if name.strip() == "DEEPINFRA_API_KEY" and value.strip():
                return value.strip().strip("'\"")

    message = "DEEPINFRA_API_KEY is not set (environment or server/.env.secrets)"
    raise SystemExit(message)


def dense_rankings(
    questions: list[dict[str, Any]], chunks: list[dict[str, Any]], cache: EmbeddingCache
) -> dict[str, list[int]]:
    chunk_matrix = cache.vectors([chunk["index_text"] for chunk in chunks])
    query_matrix = cache.vectors([question["question"] for question in questions])
    scores = query_matrix @ chunk_matrix.T
    top = np.argsort(-scores, axis=1)[:, :SEARCH_DEPTH]
    return {question["case_id"]: [int(i) for i in row] for question, row in zip(questions, top, strict=True)}


# --- report ----------------------------------------------------------------------------------


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer_path = args.tokenizer or next((path for path in TOKENIZER_CANDIDATES if path.exists()), None)

    if tokenizer_path is None:
        message = "bge-m3 tokenizer.json not found; pass --tokenizer"
        raise SystemExit(message)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    documents = split_corpus(args.corpus)

    if args.chunks_out:
        args.chunks_out.parent.mkdir(parents=True, exist_ok=True)
        args.chunks_out.write_text(json.dumps(documents, ensure_ascii=False))

    report: dict[str, Any] = {
        "label": args.label,
        "date": date.today().isoformat(),
        "tokenizer": "BAAI/bge-m3",
        "retrieval": f"proxy: BM25 (prod tokenizer) / bge-m3 cosine / RRF k={RRF_K}, no reranker, depth {SEARCH_DEPTH}",
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
        cache = EmbeddingCache(args.cache, deepinfra_key())
        rankings["dense"] = dense_rankings(flat_questions, all_chunks, cache)
        rankings["hybrid"] = {
            case: rrf(rankings["bm25"][case], rankings["dense"][case]) for case in rankings["bm25"]
        }

    for domain in domains:
        items = [d for d in documents if d["domain"] == domain]
        stats = size_stats([chunk for d in items for chunk in d["chunks"]], count)
        stats["documents"] = len(items)
        stats["chunks_per_document_avg"] = round(stats["chunks"] / len(items), 1)

        if questions[domain]:
            stats["dataset_questions"] = len(questions[domain])
            stats.update(reachability(questions[domain], by_source))
            stats["retrieval"] = {
                mode: retrieval_metrics(questions[domain], ranking, all_chunks)
                for mode, ranking in rankings.items()
            }

        report["domains"][domain] = stats

    if cache is not None:
        report["embedding_tokens_paid"] = cache.paid_tokens

    return report


def delta(value: Any, base: Any, higher_is_better: bool | None) -> str:
    if base is None or value is None:
        return f"{value}"

    diff = value - base
    mark = ""

    if higher_is_better is not None and not math.isclose(diff, 0, abs_tol=1e-9):
        mark = " ✓" if (diff > 0) == higher_is_better else " ✗"

    return f"{value} ({diff:+.4g}{mark})"


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


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    report = build_report(args)
    baseline = json.loads(args.baseline.read_text())
    print_comparison(report, baseline)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"\nreport: {args.out}")


if __name__ == "__main__":
    main()
