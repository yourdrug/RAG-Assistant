"""Tests for application use cases — business logic orchestration.

All repository and service dependencies are mocked. Tests verify:
  - Correct delegation to domain entities
  - Correct interaction with repositories
  - Error handling and edge cases
  - DTO mapping correctness

AAA (Arrange-Act-Assert) pattern throughout.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app"))

from application.dto.auth_dto import CreateUserCommand, LoginCommand, LoginResult, UserDTO
from application.dto.chat_dto import ChatCommand, ChatResult
from application.dto.document_dto import DocumentDTO
from application.dto.ingest_dto import IngestRegistryResult, IngestStatusResult
from application.use_cases.auth.authenticate_user import AuthenticateUser
from application.use_cases.auth.create_user import CreateUser
from application.use_cases.auth.list_users import ListUsers
from application.use_cases.auth.toggle_user_active import ToggleUserActive
from application.use_cases.chat.sync_chat import SyncChat
from application.use_cases.chat.stream_chat import StreamChat
from application.use_cases.document.delete_document import DeleteDocument
from application.use_cases.document.get_document import GetDocument
from application.use_cases.document.list_documents import ListDocuments
from application.use_cases.document.upload_document import UploadDocument
from application.use_cases.ingest.get_registry import GetIngestRegistry
from application.use_cases.ingest.ingest_single_file import IngestSingleFile
from application.use_cases.ingest.run_ingestion import RunIngestion
from application.use_cases.benchmark.run_benchmark import RunBenchmark
from domain.entities.user import User
from domain.entities.document import Document
from domain.entities.message import Message
from domain.entities.conversation import Conversation
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.message_role import MessageRole
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _mock_user_repo(**overrides) -> MagicMock:
    repo = MagicMock()
    repo.get_by_email.return_value = overrides.get("get_by_email")
    repo.get_by_id.return_value = overrides.get("get_by_id")
    repo.save.return_value = overrides.get("save")
    repo.list_all.return_value = overrides.get("list_all", [])
    repo.exists_admin.return_value = overrides.get("exists_admin", False)
    repo.set_active.return_value = overrides.get("set_active", True)
    return repo


def _mock_document_repo(**overrides) -> MagicMock:
    repo = MagicMock()
    repo.get_by_id.return_value = overrides.get("get_by_id")
    repo.save.return_value = overrides.get("save")
    repo.delete.return_value = None
    repo.find_active_slot.return_value = overrides.get("find_active_slot")
    repo.list_visible.return_value = overrides.get("list_visible", [])
    return repo


def _mock_password_hasher() -> MagicMock:
    hasher = MagicMock()
    hasher.hash.return_value = "hashed_pw"
    return hasher


def _mock_password_verifier() -> MagicMock:
    verifier = MagicMock()
    verifier.verify.return_value = True
    return verifier


def _mock_token_provider() -> MagicMock:
    provider = MagicMock()
    provider.create_token.return_value = "jwt_token_abc"
    return provider


def _make_user_entity(**overrides) -> User:
    defaults = dict(id=1, email="test@example.com", hashed_password="hashed", role=UserRole.USER, kind=UserKind.INTERNAL, is_active=True)
    defaults.update(overrides)
    return User(**defaults)


def _make_document_entity(**overrides) -> Document:
    defaults = dict(id=1, filename="doc.pdf", source_path="/uploads/doc.pdf", visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, status=DocumentStatus.DONE, chunks=10, chars=5000)
    defaults.update(overrides)
    return Document(**defaults)


# ===========================================================================
# CreateUser Use Case Tests
# ===========================================================================


class TestCreateUser:
    def test_create_user_success(self):
        # Arrange
        repo = _mock_user_repo(
            get_by_email=None,
            save=_make_user_entity(id=10, email="new@example.com"),
        )
        hasher = _mock_password_hasher()
        uc = CreateUser(user_repo=repo, password_hasher=hasher)
        cmd = CreateUserCommand(email="new@example.com", password="secret", role="user", kind="internal")

        # Act
        result = uc.execute(cmd, creator_role="admin")

        # Assert
        assert isinstance(result, UserDTO)
        assert result.email == "new@example.com"
        assert result.id == 10
        hasher.hash.assert_called_once_with("secret")
        repo.save.assert_called_once()

    def test_create_user_duplicate_email_raises(self):
        # Arrange
        existing = _make_user_entity(email="dup@example.com")
        repo = _mock_user_repo(get_by_email=existing)
        hasher = _mock_password_hasher()
        uc = CreateUser(user_repo=repo, password_hasher=hasher)
        cmd = CreateUserCommand(email="dup@example.com", password="secret")

        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="already exists"):
            uc.execute(cmd, creator_role="admin")

    def test_create_user_non_admin_creator_raises(self):
        # Arrange
        repo = _mock_user_repo()
        hasher = _mock_password_hasher()
        uc = CreateUser(user_repo=repo, password_hasher=hasher)
        cmd = CreateUserCommand(email="new@example.com", password="secret")

        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="Only admin"):
            uc.execute(cmd, creator_role="user")

    def test_create_client_user_with_admin_role_raises(self):
        # Arrange
        repo = _mock_user_repo()
        hasher = _mock_password_hasher()
        uc = CreateUser(user_repo=repo, password_hasher=hasher)
        cmd = CreateUserCommand(email="new@example.com", password="secret", role="admin", kind="client")

        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="Client cannot be admin"):
            uc.execute(cmd, creator_role="admin")

    def test_create_user_invalid_role_raises(self):
        # Arrange
        repo = _mock_user_repo()
        hasher = _mock_password_hasher()
        uc = CreateUser(user_repo=repo, password_hasher=hasher)
        cmd = CreateUserCommand(email="new@example.com", password="secret", role="superadmin")

        # Act & Assert
        with pytest.raises(ValidationError):
            uc.execute(cmd, creator_role="admin")


# ===========================================================================
# AuthenticateUser Use Case Tests
# ===========================================================================


class TestAuthenticateUser:
    def test_authenticate_success(self):
        # Arrange
        user = _make_user_entity()
        repo = _mock_user_repo(get_by_email=user)
        verifier = _mock_password_verifier()
        provider = _mock_token_provider()
        uc = AuthenticateUser(user_repo=repo, password_verifier=verifier, token_provider=provider)
        cmd = LoginCommand(email="test@example.com", password="secret")

        # Act
        result = uc.execute(cmd)

        # Assert
        assert isinstance(result, LoginResult)
        assert result.access_token == "jwt_token_abc"
        assert result.role == "user"
        assert result.kind == "internal"
        provider.create_token.assert_called_once_with(user_id=1, role=UserRole.USER)

    def test_authenticate_user_not_found(self):
        # Arrange
        repo = _mock_user_repo(get_by_email=None)
        verifier = _mock_password_verifier()
        provider = _mock_token_provider()
        uc = AuthenticateUser(user_repo=repo, password_verifier=verifier, token_provider=provider)
        cmd = LoginCommand(email="nobody@example.com", password="secret")

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid email or password"):
            uc.execute(cmd)

    def test_authenticate_inactive_user(self):
        # Arrange
        user = _make_user_entity(is_active=False)
        repo = _mock_user_repo(get_by_email=user)
        verifier = _mock_password_verifier()
        provider = _mock_token_provider()
        uc = AuthenticateUser(user_repo=repo, password_verifier=verifier, token_provider=provider)
        cmd = LoginCommand(email="test@example.com", password="secret")

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid email or password"):
            uc.execute(cmd)

    def test_authenticate_wrong_password(self):
        # Arrange
        user = _make_user_entity()
        repo = _mock_user_repo(get_by_email=user)
        verifier = _mock_password_verifier()
        verifier.verify.return_value = False
        provider = _mock_token_provider()
        uc = AuthenticateUser(user_repo=repo, password_verifier=verifier, token_provider=provider)
        cmd = LoginCommand(email="test@example.com", password="wrong")

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid email or password"):
            uc.execute(cmd)

    def test_authenticate_same_error_for_not_found_and_wrong_password(self):
        """Security: don't reveal whether email exists."""
        repo_not_found = _mock_user_repo(get_by_email=None)
        verifier = _mock_password_verifier()
        provider = _mock_token_provider()
        uc_not_found = AuthenticateUser(user_repo=repo_not_found, password_verifier=verifier, token_provider=provider)

        user = _make_user_entity()
        repo_wrong_pw = _mock_user_repo(get_by_email=user)
        verifier2 = _mock_password_verifier()
        verifier2.verify.return_value = False
        uc_wrong_pw = AuthenticateUser(user_repo=repo_wrong_pw, password_verifier=verifier2, token_provider=provider)

        with pytest.raises(ValidationError, match="Invalid email or password"):
            uc_not_found.execute(LoginCommand(email="x@x.com", password="pw"))
        with pytest.raises(ValidationError, match="Invalid email or password"):
            uc_wrong_pw.execute(LoginCommand(email="x@x.com", password="pw"))


# ===========================================================================
# ListUsers Use Case Tests
# ===========================================================================


class TestListUsers:
    def test_list_users_empty(self):
        # Arrange
        repo = _mock_user_repo(list_all=[])
        uc = ListUsers(user_repo=repo)

        # Act
        result = uc.execute()

        # Assert
        assert result == []

    def test_list_users_multiple(self):
        # Arrange
        users = [
            _make_user_entity(id=1, email="a@test.com"),
            _make_user_entity(id=2, email="b@test.com"),
        ]
        repo = _mock_user_repo(list_all=users)
        uc = ListUsers(user_repo=repo)

        # Act
        result = uc.execute()

        # Assert
        assert len(result) == 2
        assert result[0].email == "a@test.com"
        assert result[1].email == "b@test.com"
        assert all(isinstance(r, UserDTO) for r in result)

    def test_list_users_dto_mapping(self):
        # Arrange
        user = _make_user_entity(id=5, email="u@test.com", role=UserRole.ADMIN, kind=UserKind.CLIENT, is_active=False)
        repo = _mock_user_repo(list_all=[user])
        uc = ListUsers(user_repo=repo)

        # Act
        result = uc.execute()

        # Assert
        assert result[0].id == 5
        assert result[0].role == "admin"
        assert result[0].kind == "client"
        assert result[0].is_active is False


# ===========================================================================
# ToggleUserActive Use Case Tests
# ===========================================================================


class TestToggleUserActive:
    def test_deactivate_user(self):
        # Arrange
        user = _make_user_entity(id=10)
        repo = _mock_user_repo(get_by_id=user)
        uc = ToggleUserActive(user_repo=repo)

        # Act
        result = uc.execute(10, False, admin_id=1)

        # Assert
        assert result == {"id": 10, "is_active": False}
        repo.set_active.assert_called_once_with(10, False)

    def test_activate_user(self):
        # Arrange
        user = _make_user_entity(id=10, is_active=False)
        repo = _mock_user_repo(get_by_id=user)
        uc = ToggleUserActive(user_repo=repo)

        # Act
        result = uc.execute(10, True, admin_id=1)

        # Assert
        assert result == {"id": 10, "is_active": True}

    def test_user_not_found_raises(self):
        # Arrange
        repo = _mock_user_repo(get_by_id=None)
        uc = ToggleUserActive(user_repo=repo)

        # Act & Assert
        with pytest.raises(EntityNotFound):
            uc.execute(999, False, admin_id=1)

    def test_admin_cannot_deactivate_self(self):
        # Arrange
        user = _make_user_entity(id=42)
        repo = _mock_user_repo(get_by_id=user)
        uc = ToggleUserActive(user_repo=repo)

        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="Cannot deactivate yourself"):
            uc.execute(42, False, admin_id=42)


# ===========================================================================
# DeleteDocument Use Case Tests
# ===========================================================================


class TestDeleteDocument:
    def test_delete_own_document(self):
        # Arrange
        doc = _make_document_entity(owner_id=10)
        doc_repo = _mock_document_repo(get_by_id=doc)
        vector_repo = MagicMock()
        storage = MagicMock()
        uc = DeleteDocument(document_repo=doc_repo, vector_store_repo=vector_repo, file_storage=storage)

        # Act
        uc.execute(1, user_id=10, user_role="user")

        # Assert
        vector_repo.delete_by_document_id.assert_called_once_with(1)
        storage.delete_file.assert_called_once_with("/uploads/doc.pdf")
        doc_repo.delete.assert_called_once_with(1)

    def test_admin_can_delete_any_document(self):
        # Arrange
        doc = _make_document_entity(owner_id=99)
        doc_repo = _mock_document_repo(get_by_id=doc)
        vector_repo = MagicMock()
        storage = MagicMock()
        uc = DeleteDocument(document_repo=doc_repo, vector_store_repo=vector_repo, file_storage=storage)

        # Act
        uc.execute(1, user_id=1, user_role="admin")

        # Assert
        doc_repo.delete.assert_called_once_with(1)

    def test_non_owner_non_admin_cannot_delete(self):
        # Arrange
        doc = _make_document_entity(owner_id=10)
        doc_repo = _mock_document_repo(get_by_id=doc)
        vector_repo = MagicMock()
        storage = MagicMock()
        uc = DeleteDocument(document_repo=doc_repo, vector_store_repo=vector_repo, file_storage=storage)

        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="Can only delete your own"):
            uc.execute(1, user_id=20, user_role="user")

    def test_document_not_found_raises(self):
        # Arrange
        doc_repo = _mock_document_repo(get_by_id=None)
        vector_repo = MagicMock()
        storage = MagicMock()
        uc = DeleteDocument(document_repo=doc_repo, vector_store_repo=vector_repo, file_storage=storage)

        # Act & Assert
        with pytest.raises(EntityNotFound):
            uc.execute(999, user_id=1, user_role="admin")

    def test_delete_document_without_source_path_skips_file_deletion(self):
        # Arrange
        doc = _make_document_entity(owner_id=10, source_path="")
        doc_repo = _mock_document_repo(get_by_id=doc)
        vector_repo = MagicMock()
        storage = MagicMock()
        uc = DeleteDocument(document_repo=doc_repo, vector_store_repo=vector_repo, file_storage=storage)

        # Act
        uc.execute(1, user_id=10, user_role="user")

        # Assert
        storage.delete_file.assert_not_called()
        doc_repo.delete.assert_called_once()


# ===========================================================================
# GetDocument Use Case Tests
# ===========================================================================


class TestGetDocument:
    def test_get_document_success(self):
        # Arrange
        doc = _make_document_entity()
        doc_repo = _mock_document_repo(get_by_id=doc)
        group_repo = MagicMock()
        group_repo.get_user_group_ids.return_value = []
        client_repo = MagicMock()
        client_repo.get_assigned_client_ids.return_value = []
        uc = GetDocument(document_repo=doc_repo, group_repo=group_repo, client_assignment_repo=client_repo)

        # Act
        result = uc.execute(1, user_id=1, user_kind="internal", user_role="user")

        # Assert
        assert isinstance(result, DocumentDTO)
        assert result.id == 1
        assert result.filename == "doc.pdf"

    def test_get_document_not_found_raises(self):
        # Arrange
        doc_repo = _mock_document_repo(get_by_id=None)
        group_repo = MagicMock()
        client_repo = MagicMock()
        uc = GetDocument(document_repo=doc_repo, group_repo=group_repo, client_assignment_repo=client_repo)

        # Act & Assert
        with pytest.raises(EntityNotFound):
            uc.execute(999, user_id=1, user_kind="internal", user_role="user")

    def test_get_document_no_access_raises(self):
        # Arrange — internal_private doc owned by user 10, requested by user 20
        doc = _make_document_entity(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=10)
        doc_repo = _mock_document_repo(get_by_id=doc)
        group_repo = MagicMock()
        group_repo.get_user_group_ids.return_value = []
        client_repo = MagicMock()
        client_repo.get_assigned_client_ids.return_value = []
        uc = GetDocument(document_repo=doc_repo, group_repo=group_repo, client_assignment_repo=client_repo)

        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="No access"):
            uc.execute(1, user_id=20, user_kind="internal", user_role="user")

    def test_get_document_client_sees_own(self):
        # Arrange
        doc = _make_document_entity(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=10)
        doc_repo = _mock_document_repo(get_by_id=doc)
        group_repo = MagicMock()
        client_repo = MagicMock()
        uc = GetDocument(document_repo=doc_repo, group_repo=group_repo, client_assignment_repo=client_repo)

        # Act
        result = uc.execute(1, user_id=10, user_kind="client", user_role="user")

        # Assert
        assert result.id == 1

    def test_get_document_skips_acl_for_client(self):
        """Client kind skips group/assignment lookups."""
        doc = _make_document_entity(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=10)
        doc_repo = _mock_document_repo(get_by_id=doc)
        group_repo = MagicMock()
        client_repo = MagicMock()
        uc = GetDocument(document_repo=doc_repo, group_repo=group_repo, client_assignment_repo=client_repo)

        # Act
        uc.execute(1, user_id=10, user_kind="client", user_role="user")

        # Assert
        group_repo.get_user_group_ids.assert_not_called()
        client_repo.get_assigned_client_ids.assert_not_called()


# ===========================================================================
# ListDocuments Use Case Tests
# ===========================================================================


class TestListDocuments:
    def test_list_documents_internal_user(self):
        # Arrange
        docs = [_make_document_entity(id=1), _make_document_entity(id=2)]
        doc_repo = _mock_document_repo(list_visible=docs)
        group_repo = MagicMock()
        group_repo.get_user_group_ids.return_value = [5]
        client_repo = MagicMock()
        client_repo.get_assigned_client_ids.return_value = [10]
        uc = ListDocuments(document_repo=doc_repo, group_repo=group_repo, client_assignment_repo=client_repo)

        # Act
        result = uc.execute(user_id=1, user_kind="internal")

        # Assert
        assert len(result) == 2
        doc_repo.list_visible.assert_called_once_with(
            user_kind="internal", user_id=1, group_ids=[5], assigned_client_ids=[10]
        )

    def test_list_documents_client_user(self):
        # Arrange
        doc_repo = _mock_document_repo(list_visible=[])
        group_repo = MagicMock()
        client_repo = MagicMock()
        uc = ListDocuments(document_repo=doc_repo, group_repo=group_repo, client_assignment_repo=client_repo)

        # Act
        result = uc.execute(user_id=1, user_kind="client")

        # Assert
        assert result == []
        doc_repo.list_visible.assert_called_once_with(
            user_kind="client", user_id=1, group_ids=[], assigned_client_ids=[]
        )
        group_repo.get_user_group_ids.assert_not_called()
        client_repo.get_assigned_client_ids.assert_not_called()

    def test_list_documents_empty_result(self):
        # Arrange
        doc_repo = _mock_document_repo(list_visible=[])
        group_repo = MagicMock()
        group_repo.get_user_group_ids.return_value = []
        client_repo = MagicMock()
        client_repo.get_assigned_client_ids.return_value = []
        uc = ListDocuments(document_repo=doc_repo, group_repo=group_repo, client_assignment_repo=client_repo)

        # Act
        result = uc.execute(user_id=1, user_kind="internal")

        # Assert
        assert result == []


# ===========================================================================
# StreamChat Use Case Tests
# ===========================================================================


class TestStreamChat:
    @pytest.mark.asyncio
    async def test_stream_chat_new_conversation(self):
        # Arrange
        conv = Conversation(id=100, user_id=1)
        conv_repo = MagicMock()
        conv_repo.get_or_create.return_value = conv
        msg_repo = MagicMock()

        async def _stream(**kwargs):
            yield "Hello "
            yield "world"

        rag_service = MagicMock()
        rag_service.stream = _stream
        settings = MagicMock()
        settings.history_window = 8
        uc = StreamChat(
            conversation_repo=conv_repo, message_repo=msg_repo,
            rag_service=rag_service, settings=settings,
        )

        # Act
        chunks = []
        async for chunk in uc.execute(
            question="hi", conversation_id=None, user_id=1,
            user_kind="internal", user_role="user",
            user_group_ids=[], assigned_client_ids=[],
        ):
            chunks.append(chunk)

        # Assert
        assert "Hello " in chunks
        assert "world" in chunks
        assert msg_repo.save.call_count == 2  # user msg + assistant msg

    @pytest.mark.asyncio
    async def test_stream_chat_with_sources(self):
        # Arrange
        conv = Conversation(id=100, user_id=1)
        conv_repo = MagicMock()
        conv_repo.get_or_create.return_value = conv
        msg_repo = MagicMock()

        async def _stream(**kwargs):
            yield "answer"
            yield '\n__sources__:{"sources": [{"source": "doc.pdf"}]}'

        rag_service = MagicMock()
        rag_service.stream = _stream
        settings = MagicMock()
        settings.history_window = 8
        uc = StreamChat(
            conversation_repo=conv_repo, message_repo=msg_repo,
            rag_service=rag_service, settings=settings,
        )

        # Act
        chunks = []
        async for chunk in uc.execute(
            question="q", conversation_id=1, user_id=1,
            user_kind="internal", user_role="user",
            user_group_ids=[], assigned_client_ids=[],
        ):
            chunks.append(chunk)

        # Assert
        assert "answer" in chunks
        assert any("__meta__" in c for c in chunks)



# ===========================================================================
# SyncChat Use Case Tests
# ===========================================================================


class TestSyncChat:
    @pytest.mark.asyncio
    async def test_sync_chat_success(self):
        # Arrange
        conv = Conversation(id=100, user_id=1)
        conv_repo = MagicMock()
        conv_repo.get_or_create.return_value = conv
        msg_repo = MagicMock()
        rag_service = AsyncMock()
        rag_service.invoke.return_value = ("The answer", [{"source": "doc.pdf"}])
        settings = MagicMock()
        settings.history_window = 8
        uc = SyncChat(
            conversation_repo=conv_repo, message_repo=msg_repo,
            rag_service=rag_service, settings=settings,
        )
        cmd = ChatCommand(question="What is X?", conversation_id=1)

        # Act
        result = await uc.execute(
            command=cmd, user_id=1, user_kind="internal", user_role="user",
            user_group_ids=[], assigned_client_ids=[],
        )

        # Assert
        assert isinstance(result, ChatResult)
        assert result.answer == "The answer"
        assert result.conversation_id == 100
        assert result.sources == [{"source": "doc.pdf"}]
        assert msg_repo.save.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_chat_strips_trailing_user_msg_from_history(self):
        """If history ends with USER message, strip it (current question is the user msg)."""
        conv = Conversation(id=100, user_id=1)
        conv_repo = MagicMock()
        conv_repo.get_or_create.return_value = conv
        msg_repo = MagicMock()
        # History ends with USER — should be stripped
        msg_repo.get_history.return_value = [
            Message(role=MessageRole.ASSISTANT, content="prev"),
            Message(role=MessageRole.USER, content="prev question"),
        ]
        rag_service = AsyncMock()
        rag_service.invoke.return_value = ("answer", [])
        settings = MagicMock()
        settings.history_window = 8
        uc = SyncChat(
            conversation_repo=conv_repo, message_repo=msg_repo,
            rag_service=rag_service, settings=settings,
        )

        # Act
        await uc.execute(
            command=ChatCommand(question="new q", conversation_id=1),
            user_id=1, user_kind="internal", user_role="user",
            user_group_ids=[], assigned_client_ids=[],
        )

        # Assert — only the assistant message should be in history passed to RAG
        call_args = rag_service.invoke.call_args
        history = call_args.kwargs.get("history") or call_args[1].get("history")
        assert len(history) == 1
        assert history[0].role == MessageRole.ASSISTANT


# ===========================================================================
# RunIngestion Use Case Tests
# ===========================================================================


class TestRunIngestion:
    def test_run_full_ingestion(self):
        # Arrange
        registry_repo = MagicMock()
        ingestion_service = MagicMock()
        uc = RunIngestion(registry_repo=registry_repo, ingestion_service=ingestion_service)

        # Act
        uc.execute(docs_dir="/docs", reset=False)

        # Assert
        ingestion_service.run_full_ingestion.assert_called_once_with("/docs", reset=False, prefix=None)

    def test_run_full_ingestion_with_reset(self):
        # Arrange
        registry_repo = MagicMock()
        ingestion_service = MagicMock()
        uc = RunIngestion(registry_repo=registry_repo, ingestion_service=ingestion_service)

        # Act
        uc.execute(docs_dir="/docs", reset=True)

        # Assert
        ingestion_service.run_full_ingestion.assert_called_once_with("/docs", reset=True, prefix=None)


# ===========================================================================
# IngestSingleFile Use Case Tests
# ===========================================================================


class TestIngestSingleFile:
    def test_ingest_single_file(self):
        # Arrange
        registry_repo = MagicMock()
        ingestion_service = MagicMock()
        uc = IngestSingleFile(registry_repo=registry_repo, ingestion_service=ingestion_service)

        # Act
        uc.execute(file_path="/docs/file.pdf", force=False)

        # Assert
        ingestion_service.run_single_file.assert_called_once_with("/docs/file.pdf")
        ingestion_service.force_reindex.assert_not_called()

    def test_ingest_single_file_with_force(self):
        # Arrange
        registry_repo = MagicMock()
        ingestion_service = MagicMock()
        uc = IngestSingleFile(registry_repo=registry_repo, ingestion_service=ingestion_service)

        # Act
        uc.execute(file_path="/docs/file.pdf", force=True)

        # Assert
        ingestion_service.force_reindex.assert_called_once_with("file.pdf")
        ingestion_service.run_single_file.assert_called_once_with("/docs/file.pdf")


# ===========================================================================
# GetIngestRegistry Use Case Tests
# ===========================================================================


class TestGetIngestRegistry:
    def test_empty_registry(self):
        # Arrange
        repo = MagicMock()
        repo.load.return_value = {}
        uc = GetIngestRegistry(registry_repo=repo)

        # Act
        result = uc.execute()

        # Assert
        assert isinstance(result, IngestRegistryResult)
        assert result.total_files == 0
        assert result.total_chunks == 0
        assert result.files == []

    def test_registry_with_files(self):
        # Arrange
        repo = MagicMock()
        repo.load.return_value = {
            "doc1.pdf": {"chunks": 10, "chars": 5000, "indexed_at": "2024-01-01", "source": "local"},
            "doc2.md": {"chunks": 5, "chars": 2000, "indexed_at": "2024-01-02", "source": "s3"},
        }
        uc = GetIngestRegistry(registry_repo=repo)

        # Act
        result = uc.execute()

        # Assert
        assert result.total_files == 2
        assert result.total_chunks == 15
        assert len(result.files) == 2

    def test_registry_sorted_alphabetically(self):
        # Arrange
        repo = MagicMock()
        repo.load.return_value = {
            "z_doc.pdf": {"chunks": 1},
            "a_doc.md": {"chunks": 2},
        }
        uc = GetIngestRegistry(registry_repo=repo)

        # Act
        result = uc.execute()

        # Assert
        assert result.files[0].filename == "a_doc.md"
        assert result.files[1].filename == "z_doc.pdf"

    def test_registry_missing_fields_default(self):
        """Registry items with missing optional fields use defaults."""
        # Arrange
        repo = MagicMock()
        repo.load.return_value = {"doc.pdf": {}}
        uc = GetIngestRegistry(registry_repo=repo)

        # Act
        result = uc.execute()

        # Assert
        assert result.files[0].chunks == 0
        assert result.files[0].chars == 0
        assert result.files[0].indexed_at == ""
        assert result.files[0].source == ""


# ===========================================================================
# RunBenchmark Use Case Tests
# ===========================================================================


class TestRunBenchmark:
    def test_run_benchmark_with_defaults(self):
        # Arrange
        benchmark_service = MagicMock()
        benchmark_service.run.return_value = {"accuracy": 0.85}
        settings = MagicMock()
        settings.data_dir = "/data"
        settings.retriever_top_k = 6
        settings.llm_model = "qwen2.5"
        uc = RunBenchmark(benchmark_service=benchmark_service, settings=settings)

        # Act
        result = uc.execute()

        # Assert
        assert result == {"accuracy": 0.85}
        benchmark_service.run.assert_called_once_with(
            questions_path="/data/test_questions.json",
            out_dir="/data/benchmark_results",
            top_k=6,
            judge_model="qwen2.5",
        )

    def test_run_benchmark_with_custom_params(self):
        # Arrange
        benchmark_service = MagicMock()
        benchmark_service.run.return_value = {}
        settings = MagicMock()
        settings.data_dir = "/data"
        settings.retriever_top_k = 6
        settings.llm_model = "qwen2.5"
        uc = RunBenchmark(benchmark_service=benchmark_service, settings=settings)

        # Act
        result = uc.execute(
            questions_path="/custom/questions.json",
            out_dir="/custom/results",
            top_k=10,
            judge_model="gpt-4",
        )

        # Assert
        benchmark_service.run.assert_called_once_with(
            questions_path="/custom/questions.json",
            out_dir="/custom/results",
            top_k=10,
            judge_model="gpt-4",
        )


# ===========================================================================
# UploadDocument Use Case Tests
# ===========================================================================


class TestUploadDocument:
    @pytest.mark.asyncio
    async def test_upload_new_document(self):
        # Arrange
        doc = _make_document_entity(id=10, status=DocumentStatus.PENDING)
        doc_repo = _mock_document_repo(save=doc, get_by_id=doc, find_active_slot=None)
        group_repo = MagicMock()
        group_repo.get_user_group_ids.return_value = [5]
        processor = MagicMock()
        storage = MagicMock()
        storage.supported_extensions = [".pdf", ".md", ".txt"]
        uc = UploadDocument(
            document_repo=doc_repo, group_repo=group_repo,
            document_processor=processor, file_storage=storage,
        )

        # Act
        result = await uc.execute(
            filename="test.pdf", file_data=b"data",
            visibility="internal_group", group_id=5,
            user_id=1, user_kind="internal", user_role="user",
        )

        # Assert
        assert isinstance(result, DocumentDTO)
        doc_repo.save.assert_called_once()
        storage.upload_file.assert_called_once()
        processor.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_unsupported_extension_raises(self):
        # Arrange
        doc = _make_document_entity(id=10, status=DocumentStatus.PENDING)
        doc_repo = _mock_document_repo(save=doc, get_by_id=doc, find_active_slot=None)
        group_repo = MagicMock()
        group_repo.get_user_group_ids.return_value = []
        processor = MagicMock()
        storage = MagicMock()
        storage.supported_extensions = [".pdf", ".md"]
        uc = UploadDocument(
            document_repo=doc_repo, group_repo=group_repo,
            document_processor=processor, file_storage=storage,
        )

        # Act & Assert
        with pytest.raises(ValidationError, match="Unsupported file format"):
            await uc.execute(
                filename="test.exe", file_data=b"data",
                visibility="internal_private", group_id=None,
                user_id=1, user_kind="internal", user_role="user",
            )

    @pytest.mark.asyncio
    async def test_upload_replaces_existing_done_document(self):
        # Arrange
        existing = _make_document_entity(id=5, status=DocumentStatus.DONE)
        new_doc = _make_document_entity(id=10, status=DocumentStatus.PENDING)
        doc_repo = _mock_document_repo(save=new_doc, get_by_id=new_doc, find_active_slot=existing)
        group_repo = MagicMock()
        group_repo.get_user_group_ids.return_value = []
        processor = MagicMock()
        storage = MagicMock()
        storage.supported_extensions = [".pdf"]
        uc = UploadDocument(
            document_repo=doc_repo, group_repo=group_repo,
            document_processor=processor, file_storage=storage,
        )

        # Act
        await uc.execute(
            filename="doc.pdf", file_data=b"data",
            visibility="internal_private", group_id=None,
            user_id=1, user_kind="internal", user_role="user",
        )

        # Assert
        call_kwargs = processor.process.call_args.kwargs
        assert call_kwargs["replace_id"] == 5

    @pytest.mark.asyncio
    async def test_upload_processing_document_raises(self):
        # Arrange
        existing = _make_document_entity(id=5, status=DocumentStatus.PROCESSING)
        doc_repo = _mock_document_repo(find_active_slot=existing)
        group_repo = MagicMock()
        group_repo.get_user_group_ids.return_value = []
        processor = MagicMock()
        storage = MagicMock()
        uc = UploadDocument(
            document_repo=doc_repo, group_repo=group_repo,
            document_processor=processor, file_storage=storage,
        )

        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="already being processed"):
            await uc.execute(
                filename="doc.pdf", file_data=b"data",
                visibility="internal_private", group_id=None,
                user_id=1, user_kind="internal", user_role="user",
            )

    @pytest.mark.asyncio
    async def test_upload_invalid_visibility_raises(self):
        # Arrange
        doc_repo = _mock_document_repo()
        group_repo = MagicMock()
        processor = MagicMock()
        storage = MagicMock()
        uc = UploadDocument(
            document_repo=doc_repo, group_repo=group_repo,
            document_processor=processor, file_storage=storage,
        )

        # Act & Assert
        with pytest.raises(ValidationError):
            await uc.execute(
                filename="doc.pdf", file_data=b"data",
                visibility="invalid_vis", group_id=None,
                user_id=1, user_kind="internal", user_role="user",
            )


class TestUploadDocumentStorageKey:
    def test_storage_key_with_owner(self):
        key = UploadDocument._storage_key(owner_id=10, group_id=None, document_id=5, filename="doc.pdf")
        assert key == "uploads/users/10/5_doc.pdf"

    def test_storage_key_with_group(self):
        key = UploadDocument._storage_key(owner_id=None, group_id=3, document_id=5, filename="doc.pdf")
        assert key == "uploads/groups/3/5_doc.pdf"

    def test_storage_key_public(self):
        key = UploadDocument._storage_key(owner_id=None, group_id=None, document_id=5, filename="doc.pdf")
        assert key == "uploads/public/5_doc.pdf"

    def test_storage_key_strips_path(self):
        key = UploadDocument._storage_key(owner_id=10, group_id=None, document_id=5, filename="/path/to/doc.pdf")
        assert key == "uploads/users/10/5_doc.pdf"
