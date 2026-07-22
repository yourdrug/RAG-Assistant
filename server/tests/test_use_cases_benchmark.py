"""
Tests for benchmark use case: RunBenchmark.
All dependencies are mocked.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from application.use_cases.benchmark.run_benchmark import RunBenchmark

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings(data_dir="/data", retriever_top_k=25, llm_model="qwen2.5:7b"):
    return SimpleNamespace(data_dir=data_dir, retriever_top_k=retriever_top_k, llm_model=llm_model)


# ---------------------------------------------------------------------------
# RunBenchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def setup_method(self):
        self.benchmark_service = MagicMock()
        self.settings = _mock_settings()
        self.use_case = RunBenchmark(self.benchmark_service, self.settings)

    def test_uses_default_paths(self):
        self.benchmark_service.run.return_value = {"status": "done"}

        result = self.use_case.execute()

        call_kwargs = self.benchmark_service.run.call_args.kwargs
        assert call_kwargs["questions_path"] == "/data/test_questions.json"
        assert call_kwargs["out_dir"] == "/data/benchmark_results"
        assert call_kwargs["top_k"] == 25
        assert call_kwargs["judge_model"] == "qwen2.5:7b"

    def test_custom_questions_path(self):
        self.benchmark_service.run.return_value = {}

        self.use_case.execute(questions_path="/custom/questions.json")

        call_kwargs = self.benchmark_service.run.call_args.kwargs
        assert call_kwargs["questions_path"] == "/custom/questions.json"

    def test_custom_out_dir(self):
        self.benchmark_service.run.return_value = {}

        self.use_case.execute(out_dir="/custom/output")

        call_kwargs = self.benchmark_service.run.call_args.kwargs
        assert call_kwargs["out_dir"] == "/custom/output"

    def test_custom_top_k(self):
        self.benchmark_service.run.return_value = {}

        self.use_case.execute(top_k=10)

        call_kwargs = self.benchmark_service.run.call_args.kwargs
        assert call_kwargs["top_k"] == 10

    def test_custom_judge_model(self):
        self.benchmark_service.run.return_value = {}

        self.use_case.execute(judge_model="gpt-4")

        call_kwargs = self.benchmark_service.run.call_args.kwargs
        assert call_kwargs["judge_model"] == "gpt-4"

    def test_all_custom_params(self):
        self.benchmark_service.run.return_value = {"results": []}

        result = self.use_case.execute(
            questions_path="/q.json",
            out_dir="/out",
            top_k=5,
            judge_model="claude-3",
        )

        call_kwargs = self.benchmark_service.run.call_args.kwargs
        assert call_kwargs == {
            "questions_path": "/q.json",
            "out_dir": "/out",
            "top_k": 5,
            "judge_model": "claude-3",
        }

    def test_returns_benchmark_service_result(self):
        expected = {"accuracy": 0.85, "total": 10}
        self.benchmark_service.run.return_value = expected

        result = self.use_case.execute()

        assert result == expected
