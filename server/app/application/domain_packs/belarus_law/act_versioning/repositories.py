"""Law pack repositories of a unit of work."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.domain_packs.belarus_law.acts.repositories import ActRepositories


if TYPE_CHECKING:
    from application.uow import UnitOfWork


class ActRepositoriesNotRegisteredError(RuntimeError):
    """The unit of work was built without the law pack's repositories."""

    def __init__(self) -> None:
        super().__init__("Act repositories are not registered: enable the belarus_law domain pack")


def act_repositories(uow: UnitOfWork) -> ActRepositories:
    for repositories in uow.pack_repositories:
        if isinstance(repositories, ActRepositories):
            return repositories

    raise ActRepositoriesNotRegisteredError
