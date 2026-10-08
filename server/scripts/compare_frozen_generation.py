"""Compare generators on frozen messages; no retrieval or application config writes.

Run from server with PYTHONPATH=app. Prepare accepts exported benchmark_runs JSON
and dataset checkpoint JSON. Credentials are read from settings only at execution.
Successful requests are checkpointed individually; resume never repeats them.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

BASELINE_MODEL = "qwen/qwen-2.5-7b-instruct"
CANDIDATE_MODEL = "qwen/qwen-2.5-72b-instruct"
DEFAULT_SEEDS = [4101, 4102, 4103]
logger = logging.getLogger("default")


def digest(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare_snapshot(runs, dataset, run_id, ids, diagnostic_ids, git_commit=None):
    from domain.value_objects.llm_provider import Breadth
    from infrastructure.ml.rag.evidence_focus import format_generation_messages
    from infrastructure.ml.rag.rag_prompts import build_prompt

    run = next((run for run in runs if run["id"] == run_id), None)
    if run is None:
        raise ValueError(f"Run {run_id} not found")
    source = {case["id"]: case for case in run["per_question_results"]}
    annotations = {case["id"]: case.get("annotations") for case in dataset}
    if len(set(ids)) != len(ids) or not set(diagnostic_ids).issubset(ids):
        raise ValueError("IDs must be unique and include all diagnostic IDs")
    cases = []
    for question_id in ids:
        case = source[question_id]
        context = case["evidence"]["context"]
        if not context:
            raise ValueError(f"Question {question_id} has no saved context")
        budget = case["evidence"].get("prompt_budget") or {}
        breadth = budget.get("effective_breadth", case["breadth"])
        enumerate_cases = budget.get("enumerate_cases", breadth == Breadth.BROAD.value)
        prompt = build_prompt(breadth, enumerate_cases=enumerate_cases, question=case["question"])
        messages = format_generation_messages(prompt, context=context, history=[], question=case["question"])
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        serialized = [{"role": roles[message.type], "content": message.content} for message in messages]
        cases.append(
            {
                "id": question_id,
                "cohort": "diagnostic" if question_id in diagnostic_ids else "sufficient_context",
                "question": case["question"],
                "context": context,
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "messages": serialized,
                "messages_sha256": digest(serialized),
                "expected_answer": case["expected_answer"],
                "annotations": annotations.get(question_id),
                "historical_answer": case["answer"],
                "historical_correctness": case["correctness"],
                "historical_context_fact_coverage": case["evidence_metrics"]["context_fact_coverage"],
                "effective_breadth": breadth,
                "enumerate_cases": enumerate_cases,
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_run_id": run_id,
        "source_sweep_id": run["sweep_id"],
        "dataset": run["dataset"],
        "git_commit": git_commit,
        "prompt_scope": "Current common production prompt; no domain addendum, history or summary. "
        "Historical context is unchanged; this is not a replay of historical full messages.",
        "runtime": {
            "models": [BASELINE_MODEL, CANDIDATE_MODEL],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 2048,
            "seeds": DEFAULT_SEEDS,
        },
        "cases": cases,
    }


def validate_snapshot(snapshot):
    runtime = snapshot["runtime"]
    if len(set(runtime["models"])) != len(runtime["models"]) or not runtime["models"]:
        raise ValueError("Models must be nonempty and unique")
    if not runtime["seeds"] or len(set(runtime["seeds"])) != len(runtime["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    ids = [case["id"] for case in snapshot["cases"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Cases must be nonempty and unique")
    for case in snapshot["cases"]:
        if digest(case["messages"]) != case["messages_sha256"]:
            raise ValueError(f"Frozen messages changed for {case['id']}")
        if hashlib.sha256(case["context"].encode()).hexdigest() != case["context_sha256"]:
            raise ValueError(f"Frozen context changed for {case['id']}")
        if not any(case["context"] in message["content"] for message in case["messages"]):
            raise ValueError(f"Context missing from messages for {case['id']}")


def initial_results(snapshot, previous=None):
    validate_snapshot(snapshot)
    snapshot_hash = digest(snapshot)
    if previous is not None:
        if previous.get("snapshot_sha256") != snapshot_hash:
            raise ValueError("Resume snapshot mismatch; use a different output file")
        return previous
    return {
        "snapshot_sha256": snapshot_hash,
        "started_at": datetime.now(UTC).isoformat(),
        "runtime": snapshot["runtime"],
        "catalog": {},
        "results": [],
        "errors": [],
    }


def make_jobs(snapshot, completed):
    runtime = snapshot["runtime"]
    jobs = [
        (case, model, repeat, seed)
        for repeat, seed in enumerate(runtime["seeds"], 1)
        for case in snapshot["cases"]
        for model in runtime["models"]
        if (case["id"], model, repeat) not in completed
    ]
    jobs.sort(key=lambda job: digest([job[0]["id"], job[1], job[2], DEFAULT_SEEDS]))
    return jobs


async def preflight(client, base_url, snapshot, provider):
    response = await client.get(base_url + "/models")
    response.raise_for_status()
    catalog = {model["id"]: model for model in response.json()["data"]}
    selected = {}
    for model in snapshot["runtime"]["models"]:
        if model not in catalog:
            raise ValueError(f"Model is unavailable: {model}")
        metadata = catalog[model]
        selected[model] = {key: metadata.get(key) for key in ("id", "name", "context_length", "pricing")}
        response = await client.get(base_url + "/models/" + model + "/endpoints")
        response.raise_for_status()
        selected[model]["endpoints"] = response.json()["data"].get("endpoints", [])
        if provider and not any(
            endpoint.get("provider_name", "").casefold() == provider.casefold()
            for endpoint in selected[model]["endpoints"]
        ):
            raise ValueError(f"Provider {provider} is unavailable for {model}")
    return selected


def summary(snapshot, rows):
    output = {}
    expected_per_model = len(snapshot["cases"]) * len(snapshot["runtime"]["seeds"])
    for model in snapshot["runtime"]["models"]:
        values = [row for row in rows if row["requested_model"] == model]
        costs = [row["usage"].get("cost") for row in values]
        elapsed = [row["latency_sec"] for row in values]
        output[model] = {
            "completed": len(values),
            "expected": expected_per_model,
            "total_input_tokens": sum(row["usage"].get("prompt_tokens", 0) for row in values),
            "total_output_tokens": sum(row["usage"].get("completion_tokens", 0) for row in values),
            "reported_cost_usd": sum(costs) if costs and all(cost is not None for cost in costs) else None,
            "mean_latency_sec": sum(elapsed) / len(elapsed) if elapsed else None,
            "finish_reasons": dict(Counter(row["finish_reason"] for row in values)),
            "providers": dict(Counter(row.get("provider") for row in values)),
        }
    return output


async def run_comparison(snapshot, out: Path, *, provider=None, concurrency=2):  # noqa: C901
    from config import settings

    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    if not settings.openrouter_api_key:
        raise ValueError("OpenRouter credentials are not configured")
    results = initial_results(snapshot, read_json(out) if out.exists() else None)
    completed = {(row["id"], row["requested_model"], row["repeat"]) for row in results["results"]}
    jobs = make_jobs(snapshot, completed)
    if results.get("provider_constraint") != provider and completed:
        raise ValueError("Resume provider mismatch")
    results["provider_constraint"] = provider
    base_url = settings.openrouter_base_url.rstrip("/")
    limiter = asyncio.Semaphore(concurrency)
    headers = {"Authorization": "Bearer " + settings.openrouter_api_key}
    async with httpx.AsyncClient(headers=headers, timeout=180) as client:
        if jobs:
            results["catalog"] = await preflight(client, base_url, snapshot, provider)
        save_json(out, results)

        async def generate(case, model, repeat, seed):
            async with limiter:
                payload = {
                    "model": model,
                    "messages": case["messages"],
                    "temperature": snapshot["runtime"]["temperature"],
                    "top_p": snapshot["runtime"]["top_p"],
                    "max_tokens": snapshot["runtime"]["max_tokens"],
                    "seed": seed,
                    "stream": False,
                }
                if provider:
                    payload["provider"] = {"only": [provider], "allow_fallbacks": False}
                started = time.monotonic()
                try:
                    response = await client.post(base_url + "/chat/completions", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    if data.get("error"):
                        raise ValueError("Provider returned an error in HTTP 200 response")
                    choice = data["choices"][0]
                    answer = choice["message"].get("content")
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("Provider returned no answer text")
                    row = {
                        "id": case["id"],
                        "cohort": case["cohort"],
                        "requested_model": model,
                        "actual_model": data.get("model"),
                        "provider": data.get("provider"),
                        "repeat": repeat,
                        "seed": seed,
                        "context_sha256": case["context_sha256"],
                        "messages_sha256": case["messages_sha256"],
                        "generation_id": data.get("id"),
                        "answer": answer,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": data.get("usage") or {},
                        "latency_sec": round(time.monotonic() - started, 3),
                        "finished_at": datetime.now(UTC).isoformat(),
                    }
                    results["results"].append(row)
                    results["summary"] = summary(snapshot, results["results"])
                    save_json(out, results)
                    logger.info(
                        "Completed %s %s repeat=%s (%s total)",
                        case["id"],
                        model,
                        repeat,
                        len(results["results"]),
                    )
                except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                    # Never log response bodies or credentials. Failed attempts stay
                    # explicit and can be retried by invoking this command again.
                    results["errors"].append(
                        {
                            "id": case["id"],
                            "model": model,
                            "repeat": repeat,
                            "error_type": type(exc).__name__,
                            "http_status": exc.response.status_code
                            if isinstance(exc, httpx.HTTPStatusError)
                            else None,
                            "at": datetime.now(UTC).isoformat(),
                        }
                    )
                    save_json(out, results)
                    logger.error("Failed %s %s repeat=%s: %s", case["id"], model, repeat, type(exc).__name__)

        await asyncio.gather(*(generate(*job) for job in jobs))
    results["summary"] = summary(snapshot, results["results"])
    results["finished_at"] = datetime.now(UTC).isoformat()
    save_json(out, results)
    expected = len(snapshot["cases"]) * len(snapshot["runtime"]["models"]) * len(snapshot["runtime"]["seeds"])
    if len(results["results"]) != expected:
        raise RuntimeError(f"Only {len(results['results'])}/{expected} requests completed; rerun to resume")
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--runs", type=Path, required=True)
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--run-id", type=int, required=True)
    prepare.add_argument("--ids", type=int, nargs="+", required=True)
    prepare.add_argument("--diagnostic-ids", type=int, nargs="*", default=[])
    prepare.add_argument("--git-commit")
    prepare.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("input", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--provider")
    run.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        dataset = read_json(args.dataset)
        if isinstance(dataset, dict):
            dataset = dataset["annotations"]
        snapshot = prepare_snapshot(
            read_json(args.runs), dataset, args.run_id, args.ids, args.diagnostic_ids, args.git_commit
        )
        validate_snapshot(snapshot)
        if args.out.exists():
            raise FileExistsError("Snapshot already exists; choose a new output path")
        save_json(args.out, snapshot)
        logger.info("Frozen %s cases; snapshot=%s", len(snapshot["cases"]), digest(snapshot))
    else:
        asyncio.run(
            run_comparison(
                read_json(args.input), args.out, provider=args.provider, concurrency=args.concurrency
            )
        )


if __name__ == "__main__":
    main()
