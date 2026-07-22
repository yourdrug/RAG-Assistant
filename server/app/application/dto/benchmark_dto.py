"""Benchmark-related DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunBenchmarkCommand:
    questions_path: str | None = None
    out_dir: str | None = None
    top_k: int | None = None
    judge_model: str | None = None


@dataclass(frozen=True)
class BenchmarkResultDTO:
    id: str
    question: str
    answer: str
    expected_answer: str | None
    faithfulness: float | None
    relevancy: float | None
    correctness: float | None
    hit_rate: int | None
    mrr: float | None
    avg_similarity: float
    latency_sec: float


@dataclass(frozen=True)
class BenchmarkSummaryDTO:
    total_questions: int
    total_time_sec: float
    avg_faithfulness: float | None
    avg_relevancy: float | None
    avg_correctness: float | None
    hit_rate: float | None
    avg_mrr: float | None
    avg_similarity: float
    results: list[BenchmarkResultDTO] = field(default_factory=list)
    json_path: str | None = None
    csv_path: str | None = None
