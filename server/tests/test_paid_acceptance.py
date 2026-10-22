"""The live harness must never infer during verification or without explicit consent."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from domain.runtime.cost import BudgetError


@pytest.fixture
def acceptance():
    path = Path(__file__).parents[2] / 'scripts' / 'billing/check_paid_accounting.py'
    spec = importlib.util.spec_from_file_location('paid_acceptance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verification_requires_every_provider_confirmation(acceptance):
    rows = [{'operation_id': 'op', 'actual_units': 1000}]
    with pytest.raises(BudgetError, match='Missing'):
        acceptance.verify_confirmations(rows, {})
    assert (
        acceptance.verify_confirmations(rows, {'op': {'usd': '0.000001', 'reference': 'provider-export'}})
        == 1
    )
    with pytest.raises(BudgetError, match='differs'):
        acceptance.verify_confirmations(rows, {'op': {'usd': '0.000002', 'reference': 'provider-export'}})
    with pytest.raises(BudgetError, match='Pending'):
        acceptance.verify_confirmations(
            [{'operation_id': 'op', 'actual_units': None}],
            {'op': {'usd': '0', 'reference': 'provider-export'}},
        )


def test_dispatch_requires_cap_models_and_paid_opt_in(acceptance, monkeypatch):
    monkeypatch.setattr('sys.argv', ['check_paid_accounting', '--account', 'test', '--principal', 'user:1'])
    with pytest.raises(SystemExit) as error:
        acceptance.arguments()
    assert error.value.code == 2


def test_confirmation_coverage_requires_every_path_even_when_models_match(acceptance):
    args = SimpleNamespace(embedding_model='embed', reranker_model='rerank', llm_model='m', judge_model='m')
    rows = (
        [{'provider': 'deepinfra', 'model': 'embed', 'operation': 'embedding'}] * 3
        + [{'provider': 'deepinfra', 'model': 'rerank', 'operation': 'reranking'}] * 2
        + [{'provider': 'openrouter', 'model': 'm', 'operation': 'llm'}] * 5
    )
    acceptance.verify_coverage(rows, args)
    with pytest.raises(BudgetError, match='missing required'):
        acceptance.verify_coverage(rows[:-1], args)
