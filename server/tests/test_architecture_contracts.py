"""Exercise dependency boundaries with allowed and forbidden imports."""

from pathlib import Path
import tomllib

from grimp import ImportGraph
from importlinter.configuration import configure
from importlinter.contracts.forbidden import ForbiddenContract
import pytest


SERVER = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((SERVER / "pyproject.toml").read_text())["tool"]["importlinter"]
CONTRACTS = {
    options["source_modules"][0]: options
    for options in CONFIG["contracts"]
    if options["name"].endswith(": declared application dependencies")
}


def test_every_application_package_has_a_dependency_contract() -> None:
    packages = {
        f"application.{path.name}"
        for path in (SERVER / "app/application").iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }
    assert set(CONTRACTS) == packages


@pytest.mark.parametrize(
    "importer, imported, allowed",
    [
        ("health.service", "admin", True),
        ("health.service", "uow.factory", True),
        ("health.service", "jobs.service", False),
        ("health.service", "chat.service", False),
        ("health.service", "ports.event_bus", False),
        ("jobs.service", "uow.factory", True),
        ("jobs.service", "health.service", False),
        ("jobs.service", "admin", False),
        ("jobs.service", "ports.job_enqueuer", False),
        ("indexing.pipeline", "act_versioning.service", False),
        ("documents.commands", "ingestion", True),
        ("documents.commands", "chat.service", False),
        ("documents.commands", "ports.file_storage", False),
        ("documents.upload", "ports.file_storage", True),
        ("documents.commands", "ports.event_bus", False),
        ("chunks.service", "documents", True),
        ("chunks.service", "ports.cache_invalidator", False),
        ("chunks.mutations", "ports.cache_invalidator", True),
        ("ingestion.service", "indexing", True),
        ("ingestion.service", "documents", False),
        ("benchmark.orchestrator", "chat", True),
        ("benchmark.orchestrator", "documents", False),
        ("ports.file_storage", "documents", False),
        ("ports.file_storage", "health", False),
        ("ports.file_storage", "jobs", False),
        ("uow.factory", "jobs", False),
        ("uow.factory", "ports.file_storage", False),
    ],
)
def test_application_dependency_boundary(importer: str, imported: str, allowed: bool) -> None:
    configure()  # type: ignore[no-untyped-call]  # import-linter exposes an untyped setup function.
    options = CONTRACTS[f"application.{importer.split('.', 1)[0]}"]
    graph = ImportGraph()

    def add_import(source: str, target: str) -> None:
        for module in (source, target):
            parts = module.split(".")

            for length in range(1, len(parts) + 1):
                graph.add_module(".".join(parts[:length]))

        graph.add_import(importer=source, imported=target, line_number=1, line_contents="")

    for module in CONTRACTS:
        graph.add_module(module)

    for edge in options.get("ignore_imports", []):
        add_import(*edge.split(" -> "))

    add_import(f"application.{importer}", f"application.{imported}")

    contract_options = {
        key: str(value) if isinstance(value, bool) else value for key, value in options.items()
    }
    contract = ForbiddenContract(options["name"], CONFIG, contract_options)
    assert contract.check(graph, verbose=False).kept is allowed


@pytest.mark.parametrize(
    "context, imported, allowed",
    [
        ("shared", "domain.shared.exceptions", True),
        ("identity", "domain.shared.exceptions", True),
        ("identity", "domain.identity.roles", True),
        *[
            (context, module, False)
            for context, name in (
                ("shared", "Domain shared kernel is independent of contexts"),
                ("identity", "Domain identity depends only on shared"),
            )
            for options in CONFIG["contracts"]
            if options["name"] == name
            for module in options["forbidden_modules"]
        ],
    ],
)
def test_domain_context_dependency_boundary(context: str, imported: str, allowed: bool) -> None:
    configure()  # type: ignore[no-untyped-call]  # import-linter exposes an untyped setup function.
    options = next(
        options for options in CONFIG["contracts"] if options.get("source_modules") == [f"domain.{context}"]
    )
    graph = ImportGraph()
    importer = f"domain.{context}.example"
    modules = [importer, imported, *options["forbidden_modules"]]

    for module in modules:
        parts = module.split(".")

        for length in range(1, len(parts) + 1):
            graph.add_module(".".join(parts[:length]))

    graph.add_import(importer=importer, imported=imported, line_number=1, line_contents="")
    contract = ForbiddenContract(options["name"], CONFIG, options)
    assert contract.check(graph, verbose=False).kept is allowed


@pytest.mark.parametrize("context", ["shared", "identity"])
def test_context_contract_covers_every_forbidden_domain_package(context: str) -> None:
    options = next(
        options for options in CONFIG["contracts"] if options.get("source_modules") == [f"domain.{context}"]
    )
    siblings = {
        f"domain.{path.stem}"
        for path in (SERVER / "app/domain").iterdir()
        if path.name not in {"__init__.py", "shared", context}
        and (path.suffix == ".py" or (path.is_dir() and (path / "__init__.py").exists()))
    }
    assert set(options["forbidden_modules"]) == siblings
