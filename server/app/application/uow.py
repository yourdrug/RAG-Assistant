"""Unit of Work — manages transaction across all repositories."""

from __future__ import annotations

from types import TracebackType

from domain.repositories.client_assignment_repository import ClientAssignmentRepository
from domain.repositories.conversation_repository import ConversationRepository
from domain.repositories.document_repository import DocumentRepository
from domain.repositories.group_repository import GroupRepository
from domain.repositories.message_repository import MessageRepository
from domain.repositories.user_repository import UserRepository
from shared.session import SessionProtocol
from shared.unit_of_work import BaseUnitOfWork


class UnitOfWork(BaseUnitOfWork):
    """Concrete UoW holding all application repositories.

    Usage:
        with UnitOfWork(session) as uow:
            user = uow.users.get_by_id(1)
            uow.conversations.create(user_id=1)
            # Transaction commits automatically on clean exit
    """

    users: UserRepository
    conversations: ConversationRepository
    messages: MessageRepository
    documents: DocumentRepository
    groups: GroupRepository
    client_assignments: ClientAssignmentRepository

    def __init__(
        self,
        session: SessionProtocol,
        users: UserRepository,
        conversations: ConversationRepository,
        messages: MessageRepository,
        documents: DocumentRepository,
        groups: GroupRepository,
        client_assignments: ClientAssignmentRepository,
    ) -> None:
        super().__init__(session)
        self.users = users
        self.conversations = conversations
        self.messages = messages
        self.documents = documents
        self.groups = groups
        self.client_assignments = client_assignments

    def __enter__(self) -> UnitOfWork:
        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
