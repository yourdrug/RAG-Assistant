"""
database.py: File containing async connection to PostgreSQL database using SQLAlchemy.
"""

from logging import (
    Logger,
    getLogger,
)

from shared.domain.exceptions import DatabaseError
from shared.infrastructure.db.dataclasses import DatabaseURL
from shared.infrastructure.db.enums import DatabaseNodeRole
from shared.infrastructure.db.settings import settings
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger: Logger = getLogger("default")


class DatabaseNode:
    """
    DatabaseNode: Class for async connection to PostgreSQL database using SQLAlchemy.
    """

    def __init__(
        self,
        node_role: DatabaseNodeRole,
        node_url: DatabaseURL,
        node_number: int,
    ) -> None:
        """
        __init__: Initialize database node.

        Args:
            node_role (DatabaseNodeRole): Role of the database node.
            node_url (DatabaseURL): Database URL.
            node_number (int): Number of the database node.
        """

        self.node_role: DatabaseNodeRole = node_role
        self.node_url: DatabaseURL = node_url
        self.node_number: int = node_number
        self.async_engine: AsyncEngine | None = None
        self.async_session_factory: async_sessionmaker | None = None

    async def _create_engine(
        self,
    ) -> None:
        """
        _create_engine: Create the async engine using the constructed database URL.
        First param is the database URL.
        Second param disables SQL query logging.
        Third param forces to use SQLAlchemy 2.0+ style APIs.
        Fourth param sets the number of connections to keep open in the pool.
        Fifth param allows extra connections beyond pool_size if needed.
        Sixth param sets the timezone for connections.
        """

        self.async_engine = create_async_engine(
            self.node_url.url,
            echo=False,
            future=True,
            pool_size=100,
            max_overflow=20,
            connect_args={"options": "-c timezone=Europe/Minsk"},
        )

    async def _create_async_session_factory(
        self,
    ) -> None:
        """
        _create_async_session_factory: Create the async session factory using the async engine.
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
        """
        _dispose_connections: Dispose of the async engine.
        """

        if not self.async_engine:
            raise DatabaseError("Async engine is not created")

        await self.async_engine.dispose()

    async def _delete_async_session_factory(
        self,
    ) -> None:
        """
        _delete_async_session_factory: Delete the async session factory.
        """

        self.async_session_factory = None

    async def _delete_engine(
        self,
    ) -> None:
        """
        _delete_engine: Delete the async engine.
        """

        self.async_engine = None

    async def connect(
        self,
    ) -> None:
        """
        connect: Connect to the database using the constructed database URL.
        """

        await self._create_engine()
        await self._create_async_session_factory()

    async def disconnect(
        self,
    ) -> None:
        """
        disconnect: Disconnect from the database.
        """

        await self._dispose_connections()
        await self._delete_async_session_factory()
        await self._delete_engine()

    def new_session(
        self,
    ) -> AsyncSession:
        """
        new_session: Create a new async session using the async session factory.

        Returns:
            AsyncSession: New async session.
        """

        if not self.async_session_factory:
            raise DatabaseError("Async session factory is not created")

        return self.async_session_factory()

    def get_role(
        self,
    ) -> str:
        """
        get_role: Get the role of the database node.

        Returns:
            str: Role of the database node.
        """

        return self.node_role.value

    def get_number(
        self,
    ) -> int:
        """
        get_number: Get the number of the database node.

        Returns:
            int: Number of the database node.
        """

        return self.node_number


class DatabaseManager:
    """
    DatabaseManager: Class, that handles database in cluster mode or single mode.
    """

    def __init__(
        self,
    ) -> None:
        """
        __init__: Initialize database manager.
        """

        self.is_cluster: bool = False
        self.master_node: DatabaseNode | None = None
        self.slave_nodes: list[DatabaseNode] = []
        self.current_slave_index: int = 0

    def _setup_cluster_environment(
        self,
    ) -> None:
        """
        _setup_cluster_environment: Setup for cluster environment.
        """

        master_url: DatabaseURL = DatabaseURL(
            settings.DB_USER,
            settings.DB_PASSWORD,
            settings.DB_HOST,
            settings.DB_PORT,
            settings.DB_NAME,
        )

        self.master_node = DatabaseNode(
            node_role=DatabaseNodeRole.MASTER,
            node_url=master_url,
            node_number=0,
        )

        for node_number, (host, port) in enumerate(
            zip(
                getattr(settings, "DB_SLAVE_HOSTS", []),
                getattr(settings, "DB_SLAVE_PORTS", []),
                strict=True,
            )
        ):
            if host and port:
                slave_url: DatabaseURL = DatabaseURL(
                    settings.DB_USER,
                    settings.DB_PASSWORD,
                    host.strip(),
                    port.strip(),
                    settings.DB_NAME,
                )

                slave_node: DatabaseNode = DatabaseNode(
                    node_role=DatabaseNodeRole.SLAVE,
                    node_url=slave_url,
                    node_number=node_number + 1,
                )

                self.slave_nodes.append(slave_node)

        if not self.slave_nodes:
            self.slave_nodes.append(self.master_node)

    def _setup_single_node_environment(
        self,
    ) -> None:
        """
        _setup_single_node_environment: Setup for single node environment.
        """

        master_url: DatabaseURL = DatabaseURL(
            settings.DB_USER,
            settings.DB_PASSWORD,
            settings.DB_HOST,
            settings.DB_PORT,
            settings.DB_NAME,
        )

        self.master_node = DatabaseNode(
            node_role=DatabaseNodeRole.MASTER,
            node_url=master_url,
            node_number=0,
        )

        self.slave_nodes.append(self.master_node)

    def _detect_environment(
        self,
    ) -> None:
        """
        _detect_environment: Detect if we're in a cluster environment or single node.
        """

        self.is_cluster = bool(settings.DB_SLAVE_HOSTS and settings.DB_SLAVE_PORTS)

        if self.is_cluster:
            self._setup_cluster_environment()
        else:
            self._setup_single_node_environment()

    def _get_next_slave(
        self,
    ) -> DatabaseNode:
        """
        _get_next_slave: Get next slave node using round-robin.
        """

        if not self.slave_nodes:
            raise DatabaseError("No slave nodes available")

        slave: DatabaseNode = self.slave_nodes[self.current_slave_index]

        self.current_slave_index = (self.current_slave_index + 1) % len(self.slave_nodes)

        return slave

    async def connect(
        self,
    ) -> None:
        """
        connect: Connect to all database nodes.
        """

        self._detect_environment()

        if self.master_node:
            await self.master_node.connect()
            logger.info(
                "Connected to master database. "
                f"Role: {self.master_node.get_role()}; "
                f"Number: {self.master_node.get_number()};"
            )

        slaves_to_connect: list[DatabaseNode] = [
            slave for slave in self.slave_nodes if slave != self.master_node
        ]

        for slave_node in slaves_to_connect:
            await slave_node.connect()
            logger.info(
                f"Connected to slave database. Role: {slave_node.get_role()}; Number: {slave_node.get_number()};"
            )

        logger.info(f"Database environment: {'CLUSTER' if self.is_cluster else 'SINGLE NODE'}")

    async def disconnect(
        self,
    ) -> None:
        """
        disconnect: Disconnect from all database nodes.
        """

        slaves_to_disconnect: list[DatabaseNode] = [
            slave for slave in self.slave_nodes if slave != self.master_node
        ]

        for slave_node in slaves_to_disconnect:
            await slave_node.disconnect()
            logger.info(
                f"Disconnected from slave database. Role: {slave_node.get_role()}; Number: {slave_node.get_number()};"
            )

        if self.master_node:
            await self.master_node.disconnect()
            logger.info(
                "Disconnected from master database. "
                f"Role: {self.master_node.get_role()}; "
                f"Number: {self.master_node.get_number()};"
            )

    def get_write_session(
        self,
    ) -> AsyncSession:
        """
        get_write_session: Get session for write operations (master).

        Returns:
            AsyncSession: Session for write operations.
        """

        if not self.master_node:
            raise DatabaseError("Master database node is not available")

        return self.master_node.new_session()

    def get_read_session(
        self,
    ) -> AsyncSession:
        """
        get_read_session: Get session for read operations (slave with round-robin).

        Returns:
            AsyncSession: Session for read operations.
        """

        slave_node: DatabaseNode = self._get_next_slave()

        return slave_node.new_session()

    def get_session(
        self,
        master: bool = False,
    ) -> AsyncSession:
        """
        get_session: Get appropriate session based on operation type.

        Args:
            master (bool): True if write operation, False if read operation.

        Returns:
            AsyncSession: Session for read or write operations.
        """

        return self.get_write_session() if master else self.get_read_session()


database: DatabaseManager = DatabaseManager()
