"""Async PostgreSQL connection manager using SQLAlchemy 2.0.

Provides ``DatabaseManager`` with separate read/write async engines,
session factories, and convenience helpers (``fetch_all``, ``fetch_one``,
``execute``).  The module-level ``database`` singleton is the primary
entry point used throughout the application.

Lifecycle:
  - Created at import time (module-level assignment)
  - ``connect()`` called in FastAPI ``lifespan()``
  - ``disconnect()`` called in FastAPI ``lifespan()`` shutdown
  - Application-scoped: one instance per process
  - Sessions are per-request: ``get_session()`` creates a fresh ``AsyncSession``
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from logging import (
    Logger,
    getLogger,
)
from typing import TYPE_CHECKING

from sqlalchemy import Connection, create_engine
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from config import DatabaseSection, process_section
from domain.shared.exceptions import DatabaseError
from infrastructure.database.dataclasses import DatabaseURL
from infrastructure.database.enums import DatabaseNodeRole


if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
    )


logger: Logger = getLogger("default")


def connection_arguments(database_settings: DatabaseSection) -> dict[str, object]:

    return {
        "connect_timeout": 5,
        "options": (
            "-c timezone=Europe/Minsk "
            f"-c statement_timeout={database_settings.db_statement_timeout_ms} "
            f"-c idle_in_transaction_session_timeout={database_settings.db_idle_in_transaction_timeout_ms}"
        ),
    }


def configured_database_url(database_settings: DatabaseSection) -> DatabaseURL:

    return DatabaseURL(
        database_settings.db_user,
        database_settings.db_password,
        database_settings.db_host,
        database_settings.db_port,
        database_settings.db_name,
    )


class DatabaseNode:
    """DatabaseNode: Class for async connection to PostgreSQL database using SQLAlchemy."""

    def __init__(
        self,
        node_role: DatabaseNodeRole,
        node_url: DatabaseURL,
        node_number: int,
        database_settings: DatabaseSection,
    ) -> None:
        """__init__: Initialize database node.

        Args:
            node_role (DatabaseNodeRole): Role of the database node.
            node_url (DatabaseURL): Database URL.
            node_number (int): Number of the database node.
            database_settings (DatabaseSection): Pool and timeout settings.
        """
        self.database_settings: DatabaseSection = database_settings
        self.node_role: DatabaseNodeRole = node_role
        self.node_url: DatabaseURL = node_url
        self.node_number: int = node_number
        self.async_engine: AsyncEngine | None = None
        self.async_session_factory: async_sessionmaker[AsyncSession] | None = None

    async def _create_engine(
        self,
    ) -> None:
        """_create_engine: Create the async engine using the constructed database URL.

        First param is the database URL.
        Second param disables SQL query logging.
        Third param forces to use SQLAlchemy 2.0+ style APIs.
        Fourth param sets the number of connections to keep open in the pool.
        Fifth param allows extra connections beyond pool_size if needed.
        Sixth param sets the timezone for connections.
        """
        database_settings = self.database_settings

        self.async_engine = create_async_engine(
            self.node_url.url,
            echo=False,
            future=True,
            pool_size=database_settings.db_pool_size,
            max_overflow=database_settings.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=database_settings.db_pool_recycle_sec,
            connect_args=connection_arguments(database_settings),
        )

    async def _create_async_session_factory(
        self,
    ) -> None:
        """_create_async_session_factory: Create the async session factory using the async engine.

        First param is the async engine.
        Second param class is the session class.
        Third param prevents attribute expiration after commit (better for async).
        """
        self.async_session_factory = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def _dispose_connections(
        self,
    ) -> None:
        """_dispose_connections: Dispose of the async engine."""
        if not self.async_engine:
            msg = "Async engine is not created"
            raise DatabaseError(msg)

        await self.async_engine.dispose()

    async def _delete_async_session_factory(
        self,
    ) -> None:
        """_delete_async_session_factory: Delete the async session factory."""
        self.async_session_factory = None

    async def _delete_engine(
        self,
    ) -> None:
        """_delete_engine: Delete the async engine."""
        self.async_engine = None

    async def connect(
        self,
    ) -> None:
        """connect: Connect to the database using the constructed database URL."""
        await self._create_engine()
        await self._create_async_session_factory()

    async def disconnect(
        self,
    ) -> None:
        """disconnect: Disconnect from the database."""
        await self._dispose_connections()
        await self._delete_async_session_factory()
        await self._delete_engine()

    def new_session(
        self,
    ) -> AsyncSession:
        """new_session: Create a new async session using the async session factory.

        Returns:
            AsyncSession: New async session.
        """
        if not self.async_session_factory:
            msg = "Async session factory is not created"
            raise DatabaseError(msg)

        return self.async_session_factory()

    def get_role(
        self,
    ) -> str:
        """get_role: Get the role of the database node.

        Returns:
            str: Role of the database node.
        """
        return self.node_role.value

    def get_number(
        self,
    ) -> int:
        """get_number: Get the number of the database node.

        Returns:
            int: Number of the database node.
        """
        return self.node_number


class DatabaseManager:
    """DatabaseManager: Class, that handles database in cluster mode or single mode."""

    def __init__(
        self,
        database_settings: DatabaseSection,
    ) -> None:
        """__init__: Initialize database manager."""
        self.database_settings: DatabaseSection = database_settings
        self.is_cluster: bool = False
        self.master_node: DatabaseNode | None = None
        self.slave_nodes: list[DatabaseNode] = []
        self.current_slave_index: int = 0
        self.master_connection_engine: Engine | None = None
        self.async_master_connection_engine: AsyncEngine | None = None

    def _setup_cluster_environment(
        self,
    ) -> None:
        """_setup_cluster_environment: Setup for cluster environment."""
        database_settings = self.database_settings

        master_url: DatabaseURL = configured_database_url(self.database_settings)

        self.master_node = DatabaseNode(
            node_role=DatabaseNodeRole.MASTER,
            node_url=master_url,
            node_number=0,
            database_settings=self.database_settings,
        )

        for node_number, (host, port) in enumerate(
            zip(
                database_settings.db_slave_hosts_list,
                database_settings.db_slave_ports_list,
                strict=True,
            )
        ):
            if host and port:
                slave_url: DatabaseURL = DatabaseURL(
                    database_settings.db_user,
                    database_settings.db_password,
                    host,
                    port,
                    database_settings.db_name,
                )

                slave_node: DatabaseNode = DatabaseNode(
                    node_role=DatabaseNodeRole.SLAVE,
                    node_url=slave_url,
                    node_number=node_number + 1,
                    database_settings=self.database_settings,
                )

                self.slave_nodes.append(slave_node)

        if not self.slave_nodes:
            self.slave_nodes.append(self.master_node)

    def _setup_single_node_environment(
        self,
    ) -> None:
        """_setup_single_node_environment: Setup for single node environment."""
        master_url: DatabaseURL = configured_database_url(self.database_settings)

        self.master_node = DatabaseNode(
            node_role=DatabaseNodeRole.MASTER,
            node_url=master_url,
            node_number=0,
            database_settings=self.database_settings,
        )

        self.slave_nodes.append(self.master_node)

    def _detect_environment(
        self,
    ) -> None:
        """_detect_environment: Detect if we're in a cluster environment or single node."""
        database_settings = self.database_settings

        self.is_cluster = bool(database_settings.db_slave_hosts and database_settings.db_slave_ports)

        if self.is_cluster:
            self._setup_cluster_environment()
        else:
            self._setup_single_node_environment()

    def _get_next_slave(
        self,
    ) -> DatabaseNode:
        """_get_next_slave: Get next slave node using round-robin."""
        if not self.slave_nodes:
            msg = "No slave nodes available"
            raise DatabaseError(msg)

        slave: DatabaseNode = self.slave_nodes[self.current_slave_index]

        self.current_slave_index = (self.current_slave_index + 1) % len(self.slave_nodes)

        return slave

    async def connect(
        self,
    ) -> None:
        """connect: Connect to all database nodes."""
        self._detect_environment()

        if self.master_node:
            await self.master_node.connect()
            logger.info(
                "Connected to master database. Role: %s; Number: %s;",
                self.master_node.get_role(),
                self.master_node.get_number(),
            )

        if self.master_node is not None and self.master_node.async_engine is not None:
            self.create_master_connection_engines(
                self.master_node.async_engine.url, connection_arguments(self.database_settings)
            )

        slaves_to_connect: list[DatabaseNode] = [
            slave for slave in self.slave_nodes if slave != self.master_node
        ]

        for slave_node in slaves_to_connect:
            await slave_node.connect()
            logger.info(
                "Connected to slave database. Role: %s; Number: %s;",
                slave_node.get_role(),
                slave_node.get_number(),
            )

        logger.info("Database environment: %s", "CLUSTER" if self.is_cluster else "SINGLE NODE")

    async def disconnect(
        self,
    ) -> None:
        """disconnect: Disconnect from all database nodes."""
        if self.async_master_connection_engine is not None:
            await self.async_master_connection_engine.dispose()
            self.async_master_connection_engine = None

        if self.master_connection_engine is not None:
            self.master_connection_engine.dispose()
            self.master_connection_engine = None

        slaves_to_disconnect: list[DatabaseNode] = [
            slave for slave in self.slave_nodes if slave != self.master_node
        ]

        for slave_node in slaves_to_disconnect:
            if slave_node.async_engine is not None:
                await slave_node.disconnect()

            logger.info(
                "Disconnected from slave database. Role: %s; Number: %s;",
                slave_node.get_role(),
                slave_node.get_number(),
            )

        if self.master_node is not None and self.master_node.async_engine is not None:
            await self.master_node.disconnect()
            logger.info(
                "Disconnected from master database. Role: %s; Number: %s;",
                self.master_node.get_role(),
                self.master_node.get_number(),
            )

    def create_master_connection_engines(self, url: URL, connect_args: dict[str, object]) -> None:
        """Dedicated nonpooled engines for session locks, independent of ordinary sessions."""
        self.master_connection_engine = create_engine(
            url,
            poolclass=NullPool,
            isolation_level="AUTOCOMMIT",
            connect_args=connect_args,
        )
        self.async_master_connection_engine = create_async_engine(
            url,
            poolclass=NullPool,
            isolation_level="AUTOCOMMIT",
            connect_args=connect_args,
        )

    @property
    def master_url(self) -> URL:
        engine = self.async_master_connection_engine or self.master_connection_engine

        if engine is not None:
            return engine.url

        if self.master_node is not None:
            if self.master_node.async_engine is not None:
                return self.master_node.async_engine.url

            return self.master_node.node_url.url

        return configured_database_url(self.database_settings).url

    @contextmanager
    def master_connection(self, *, connect_args: dict[str, object] | None = None) -> Iterator[Connection]:
        """Use the dedicated engine, or a scoped engine for standalone tools."""
        engine = self.master_connection_engine
        owned = engine is None or connect_args is not None

        if owned:
            engine = create_engine(
                self.master_url,
                poolclass=NullPool,
                isolation_level="AUTOCOMMIT",
                connect_args=connection_arguments(self.database_settings) | (connect_args or {}),
            )

        if engine is None:
            msg = "Master connection engine is not created"
            raise DatabaseError(msg)

        try:
            with engine.connect() as connection:
                yield connection
        finally:
            if owned:
                engine.dispose()

    @asynccontextmanager
    async def async_master_connection(
        self, *, connect_args: dict[str, object] | None = None
    ) -> AsyncIterator[AsyncConnection]:
        """Use the dedicated engine, or a scoped engine before application startup."""
        engine = self.async_master_connection_engine
        owned = engine is None or connect_args is not None

        if owned:
            engine = create_async_engine(
                self.master_url,
                poolclass=NullPool,
                isolation_level="AUTOCOMMIT",
                connect_args=connection_arguments(self.database_settings) | (connect_args or {}),
            )

        if engine is None:
            msg = "Async master connection engine is not created"
            raise DatabaseError(msg)

        try:
            async with engine.connect() as connection:
                yield connection
        finally:
            if owned:
                await engine.dispose()

    def get_write_session(
        self,
    ) -> AsyncSession:
        """get_write_session: Get session for write operations (master).

        Returns:
            AsyncSession: Session for write operations.
        """
        if not self.master_node:
            msg = "Master database node is not available"
            raise DatabaseError(msg)

        return self.master_node.new_session()

    def get_read_session(
        self,
    ) -> AsyncSession:
        """get_read_session: Get session for read operations (slave with round-robin).

        Returns:
            AsyncSession: Session for read operations.
        """
        slave_node: DatabaseNode = self._get_next_slave()

        return slave_node.new_session()

    def get_session(
        self,
        *,
        master: bool = False,
    ) -> AsyncSession:
        """get_session: Get appropriate session based on operation type.

        Args:
            master (bool): True if write operation, False if read operation.

        Returns:
            AsyncSession: Session for read or write operations.
        """
        return self.get_write_session() if master else self.get_read_session()


# Process-wide singleton: the section is the live settings object, so hot-reload still applies.
database: DatabaseManager = DatabaseManager(process_section(DatabaseSection))
