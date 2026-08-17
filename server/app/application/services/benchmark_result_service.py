"""Application service for benchmark result listing and detail retrieval."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("default")

_RESULTS_GLOB = re.compile(r"^benchmark_\d{8}_\d{6}.*\.json$")


@dataclass(frozen=True)
class BenchmarkResultSummary:
    filename: str
    model: str | None = None
    total_questions: int = 0
    total_time_sec: float = 0.0
    hit_rate: float | None = None
    avg_mrr: float | None = None
    avg_faithfulness: float | None = None
    avg_relevancy: float | None = None
    avg_correctness: float | None = None
    avg_similarity: float | None = None


@dataclass(frozen=True)
class BenchmarkResultDetail:
    filename: str
    summary: BenchmarkResultSummary
    results: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkResultsList:
    results: list[BenchmarkResultSummary] = field(default_factory=list)
    total: int = 0


class BenchmarkResultService:
    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir

    def list_results(self) -> BenchmarkResultsList:
        if not self._results_dir.exists():
            return BenchmarkResultsList(results=[], total=0)

        files = sorted(
            [f for f in self._results_dir.iterdir() if f.suffix == ".json" and _RESULTS_GLOB.match(f.name)],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        summaries = []
        for f in files:
            s = self._load_summary(f)
            if s is not None:
                summaries.append(s)

        return BenchmarkResultsList(results=summaries, total=len(summaries))

    def get_result(self, filename: str) -> BenchmarkResultDetail | None:
        filepath = self._results_dir / filename

        if not filepath.exists() or not filepath.suffix == ".json":
            return None

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except Exception:
            return None

        summary = self._load_summary(filepath)
        if summary is None:
            return None

        return BenchmarkResultDetail(filename=filename, summary=summary, results=data)

    def _load_summary(self, filepath: Path) -> BenchmarkResultSummary | None:
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                return None

            ts, model = self._parse_result_filename(filepath.name)

            faiths = [r["generator_metrics"]["faithfulness"] for r in data]
            rels = [r["generator_metrics"]["relevancy"] for r in data]
            corrs = [
                r["generator_metrics"]["correctness"]
                for r in data
                if r["generator_metrics"].get("correctness") is not None
            ]
            hit_rates = [
                r["retriever_metrics"]["hit_rate"]
                for r in data
                if r["retriever_metrics"].get("hit_rate") is not None
            ]
            mrrs = [
                r["retriever_metrics"]["mrr"] for r in data if r["retriever_metrics"].get("mrr") is not None
            ]
            sims = [r["retriever_metrics"]["avg_similarity"] for r in data]

            return BenchmarkResultSummary(
                filename=filepath.name,
                model=model or None,
                total_questions=len(data),
                total_time_sec=round(sum(r.get("latency_sec", 0) for r in data), 1),
                hit_rate=round(sum(hit_rates) / len(hit_rates), 3) if hit_rates else None,
                avg_mrr=round(sum(mrrs) / len(mrrs), 3) if mrrs else None,
                avg_faithfulness=round(sum(faiths) / len(faiths), 1) if faiths else None,
                avg_relevancy=round(sum(rels) / len(rels), 1) if rels else None,
                avg_correctness=round(sum(corrs) / len(corrs), 1) if corrs else None,
                avg_similarity=round(sum(sims) / len(sims), 3) if sims else None,
            )
        except Exception as exc:
            logger.warning("Failed to parse benchmark result %s: %s", filepath.name, exc)
            return None

    @staticmethod
    def _parse_result_filename(name: str) -> tuple[str, str]:
        stem = Path(name).stem
        m = re.match(r"benchmark_(\d{8}_\d{6})_(.+)", stem)
        if m:
            return m.group(1), m.group(2)
        return "", stem
