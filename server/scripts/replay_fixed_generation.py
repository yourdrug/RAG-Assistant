"""Replay generation on frozen context, without retrieval or context packing.

Run from server with PYTHONPATH=app and DATA_DIR pointing at a writable folder.
Input: {runtime: {model, temperature, max_tokens}, cases: [{id, question,
context, expected_answer, baseline_answer, breadth?, enumerate_cases?}]}.
Credentials come from application settings, never from snapshots.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import settings
from domain.value_objects.llm_provider import Breadth
from infrastructure.ml.rag.rag_prompts import build_prompt
from infrastructure.ml.rag.evidence_focus import format_generation_messages


async def replay(snapshot: dict) -> dict:
    runtime = snapshot["runtime"]
    llm = ChatOpenAI(
        model=runtime["model"],
        temperature=runtime["temperature"],
        max_tokens=runtime["max_tokens"],
        api_key=SecretStr(settings.openrouter_api_key),
        base_url=settings.openrouter_base_url,
        timeout=120,
        max_retries=1,
    )
    results = []
    for case in snapshot["cases"]:
        prompt = build_prompt(
            case.get("breadth", Breadth.NARROW.value),
            enumerate_cases=case.get("enumerate_cases", True),
            domain_addendum=case.get("domain_addendum"),
            question=case["question"],
        )
        messages = format_generation_messages(
            prompt,
            context=case["context"],
            question=case["question"],
            history=[],
        )
        response = await llm.ainvoke(messages)
        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "context_sha256": hashlib.sha256(case["context"].encode()).hexdigest(),
                "context_chars": len(case["context"]),
                "expected_answer": case["expected_answer"],
                "baseline_answer": case.get("baseline_answer"),
                "answer": response.content,
                "usage": response.usage_metadata,
            }
        )
    return {"runtime": runtime, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(replay(json.loads(args.input.read_text())))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
