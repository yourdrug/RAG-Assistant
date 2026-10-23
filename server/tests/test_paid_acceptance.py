"""The live harness must never infer during verification or without explicit consent."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "connect", "query"])
async def test_verification_uses_manager_and_closes_resources(  # noqa: PLR0915
    acceptance, monkeypatch, tmp_path, failure
):
    confirmations = tmp_path / "confirmations.json"
    confirmations.write_text("{}")
    args = SimpleNamespace(
        principal="user:1", account="test", since="2026-10-22T00:00:00+00:00", confirmations=confirmations
    )
    database = MagicMock(connect=AsyncMock(), disconnect=AsyncMock())
    session = AsyncMock()
    database.get_session.return_value.__aenter__.return_value = session
    session.execute.return_value = MagicMock()
    rows = [{"operation_id": "op"}]
    session.execute.return_value.mappings.return_value.all.return_value = rows

    if failure == "connect":
        database.connect.side_effect = ConnectionError

    if failure == "query":
        session.execute.side_effect = ConnectionError

    redis = MagicMock(aclose=AsyncMock())
    monkeypatch.setattr(acceptance, "DatabaseManager", lambda: database)
    monkeypatch.setattr(acceptance.Redis, "from_url", lambda *_args, **_kwargs: redis)
    monkeypatch.setattr(acceptance, "BudgetRuntime", MagicMock())
    monkeypatch.setattr(acceptance, "verify_coverage", MagicMock())
    monkeypatch.setattr(acceptance, "verify_confirmations", MagicMock(return_value=1))
    dispatch = AsyncMock()
    monkeypatch.setattr(acceptance, "dispatch", dispatch)

    if failure == "none":
        await acceptance.main(args)
        database.get_session.assert_called_once_with(master=True)
        acceptance.verify_confirmations.assert_called_once_with(rows, {})
    else:
        with pytest.raises(ConnectionError):
            await acceptance.main(args)

    database.connect.assert_awaited_once_with()
    database.disconnect.assert_awaited_once_with()
    redis.aclose.assert_awaited_once_with()
    dispatch.assert_not_awaited()
