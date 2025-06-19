"""Contains implementation of qdrant manager holder."""

from typing import ClassVar

from ..config.qdrant import QdrantCollectionConfig, QdrantConfig
from .qdrant_manager import QdrantManager


class QdrantManagerHodler:
    """
    Singleton-style holder for a globally accessible QdrantManager instance.

    This class ensures that a single shared instance of QdrantManager is initialized
    and can be accessed from anywhere in the application via the `get()` method.
    Useful for dependency injection or central service coordination.
    """

    manager: ClassVar[QdrantManager | None] = None

    @classmethod
    async def init(
        cls,
        qdrant_config: QdrantConfig,
        collection_configs: dict[str, QdrantCollectionConfig],
    ) -> None:
        """
        Initialize the QdrantManager instance with given configuration.

        Parameters
        ----------
        qdrant_config : QdrantConfig
            Configuration for Qdrant server connection.

        collection_configs : dict[str, QdrantCollectionConfig]
            Dictionary of collection configurations keyed by collection name.
        """
        cls.manager = QdrantManager(
            config=qdrant_config, collection_configs=collection_configs
        )
        await cls.manager.init()

    @classmethod
    def get(cls) -> QdrantManager:
        """Get initialized qdrant manager.

        Returns
        -------
        QdrantManager
            initializer qdrant manager instance.
        """
        if cls.manager is None:
            raise RuntimeError("Qdrant manager must be initialized")
        return cls.manager
