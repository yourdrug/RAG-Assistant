"""Unit of Work — async transaction management across all repositories."""

from __future__ import annotations

from domain.repositories import ApiKeyRepository
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
        async with UnitOfWork(session) as uow:
            user = await uow.users.get_by_id(1)
            await uow.conversations.create(user_id=1)
            # Transaction commits automatically on clean exit
    """

    users: UserRepository
    conversations: ConversationRepository
    messages: MessageRepository
    documents: DocumentRepository
    groups: GroupRepository
    client_assignments: ClientAssignmentRepository
    api_keys: ApiKeyRepository

    def __init__(
        self,
        session: SessionProtocol,
        users: UserRepository,
        conversations: ConversationRepository,
        messages: MessageRepository,
        documents: DocumentRepository,
        groups: GroupRepository,
        client_assignments: ClientAssignmentRepository,
        api_keys: ApiKeyRepository,
    ) -> None:
        super().__init__(session)
        self.users = users
        self.conversations = conversations
        self.messages = messages
        self.documents = documents
        self.groups = groups
        self.client_assignments = client_assignments
        self.api_keys = api_keys
