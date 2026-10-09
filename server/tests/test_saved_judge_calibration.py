"""Frozen judging criteria and paid-call resume; semantic calibration is a real LLM experiment."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def experiment():
    path = Path(__file__).parents[1] / 'scripts' / 'calibrate_saved_judge.py'
    spec = importlib.util.spec_from_file_location('saved_judge_calibration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_exact_criteria_and_resume_preserve_frozen_answers(experiment, monkeypatch, tmp_path):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, json):
            prompt = json['messages'][0]['content']
            calls.append(prompt)
            payload = experiment.json.loads(prompt.split('Данные JSON:\n')[1])
            if 'answer' in payload:
                assert payload['answer'] == 'saved answer'
            metrics = [m for m in experiment.INSTRUCTIONS if f'{m}:' in prompt]
            for m in metrics:
                assert experiment.INSTRUCTIONS[m] in prompt
            if 'faithfulness' in metrics:
                assert 'expected_answer' not in payload
                assert payload['context'] == 'saved context'
            data = {
                'choices': [
                    {
                        'finish_reason': 'stop',
                        'message': {
                            'content': experiment.json.dumps(
                                {m: {'score': 10, 'reason': 'fixture'} for m in metrics}
                            )
                        },
                    }
                ],
                'model': 'fixed',
            }
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    monkeypatch.setattr(experiment.httpx, 'AsyncClient', Client)
    snapshot = {
        'judge_model': 'fixed',
        'cases': [
            {
                'key': 'case',
                'cohort': 'calibration',
                'repeats': 3,
                'question': {'question': '?', 'expected_answer': 'reference', 'annotations': {}},
                'answer': 'saved answer',
                'context': 'saved context',
            }
        ],
    }
    out = tmp_path / 'results.json'
    await experiment.run(snapshot, out, 'calibration')
    assert len(calls) == 9
    await experiment.run(snapshot, out, 'calibration')
    assert len(calls) == 9
    assert len(json.loads(out.read_text())['requests']) == 9
    snapshot['cases'][0]['context'] = 'changed context'
    with pytest.raises(ValueError, match='Resume input/protocol mismatch'):
        await experiment.run(snapshot, out, 'calibration')
    assert len(calls) == 9

    snapshot['cases'][0]['context'] = 'saved context'
    monkeypatch.setattr(experiment, 'JUDGE_RUBRIC_VERSION', 'changed-rubric')
    with pytest.raises(ValueError, match='Resume input/protocol mismatch'):
        await experiment.run(snapshot, out, 'calibration')
    assert len(calls) == 9
