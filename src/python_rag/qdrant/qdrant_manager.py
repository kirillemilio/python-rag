"""Contains implementation of qdrant manager class."""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from prometheus_client import Counter, Gauge, Histogram
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    CompressionRatio,
    Filter,
    HnswConfigDiff,
    PointStruct,
    ProductQuantization,
    ProductQuantizationConfig,
    VectorParams,
)

from ..config.qdrant import QdrantCollectionConfig, QdrantConfig
from ..dto.document import Chunk, ChunkWithEmbedding
from ..monitoring import MetricsHolder
from .scored_chunk import ScoredChunk

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manages Qdrant vector collections for storage and retrieval of document embeddings.

    This class wraps around the Qdrant client and provides high-level methods
    to initialize collections, add document chunks in batches, and perform deletion operations
    by chunk ID, document ID, or entire collection.

    Attributes
    ----------
    config : QdrantConfig
        Configuration for connecting to the Qdrant server.

    collection_configs : dict[str, QdrantCollectionConfig]
        Mapping of collection names to their configuration parameters.
    """

    config: QdrantConfig
    collections_config: dict[str, QdrantCollectionConfig]

    _pr_hist: MetricsHolder[Histogram]
    _pr_counter: MetricsHolder[Counter]
    _pr_gauge: MetricsHolder[Gauge]

    def __init__(
        self,
        config: QdrantConfig,
        collection_configs: dict[str, QdrantCollectionConfig],
    ) -> None:
        """
        Initialize the Qdrant manager with connection and collection configurations.

        Parameters
        ----------
        config : QdrantConfig
            Configuration for Qdrant client (host, ports, credentials, etc.).

        collection_configs : dict[str, QdrantCollectionConfig]
            Definitions for each Qdrant collection to manage.
        """
        self.client = AsyncQdrantClient(
            host=config.host,
            port=config.http_port,
            https=config.use_secure,
            timeout=config.timeout,
            grpc_port=config.grpc_port,
            prefer_grpc=config.use_grpc,
            api_key=config.api_key,
        )
        self.collection_configs = collection_configs

        # Prometheus metrics
        self._pr_hist = MetricsHolder.get_default_hist()
        self._pr_counter = MetricsHolder.get_default_counter()
        self._pr_gauge = MetricsHolder.get_default_gauge()

    async def init(self) -> None:
        """
        Initialize all Qdrant collections defined in the configuration.

        This method checks for the existence of each collection defined in `collections_config`.
        If a collection does not exist, it will be created with the corresponding parameters,
        including HNSW indexing, optional Product Quantization (PQ), and vector storage settings.

        Collections are initialized using:
        - HNSW configuration (`hnsw_m`, `ef_construct`, `full_scan_threshold`, `on_disk`)
        - Vector configuration (`vector_size`, `distance`, `on_disk`)
        - Optional compression (`num_subvectors`, PQ compression ratio)

        This must be called before using other Qdrant operations.

        Raises
        ------
        qdrant_client.QdrantException
            If there is an error connecting to the Qdrant server or creating a collection.
        """
        for collection_name, collection_config in self.collection_configs.items():
            if not await self.client.collection_exists(collection_name):
                logger.info(f"[Qdrant] Creating collection '{collection_name}'...")
                quant_config: ProductQuantization | None = None
                if collection_config.compression:
                    quant_config = ProductQuantization(
                        product=ProductQuantizationConfig(
                            compression=CompressionRatio(collection_config.compression),
                            always_ram=True,
                        )
                    )
                await self.client.create_collection(
                    collection_name=collection_name,
                    hnsw_config=HnswConfigDiff(
                        m=collection_config.hnsw_m,
                        ef_construct=collection_config.ef_construct,
                        full_scan_threshold=collection_config.full_scan_threshold,
                        on_disk=False,
                    ),
                    quantization_config=quant_config,
                    vectors_config=VectorParams(
                        size=collection_config.vector_size,
                        distance=collection_config.get_distance_enum(),
                        on_disk=False,
                    ),
                )
                logger.info(
                    f"[Qdrant] Collection '{collection_name}' created successfully."
                )
            else:
                logger.info(f"[Qdrant] Collection '{collection_name}' already exists.")
                continue

    async def add_chunks_batch(
        self, collection_name: str, chunks: Sequence[ChunkWithEmbedding]
    ) -> None:
        """
        Add a batch of document chunks and their vectors to a Qdrant collection.

        Parameters
        ----------
        collection_name : str
            Name of the target Qdrant collection.
        chunks : Sequence[ChunkWithEmbedding]
            Metadata for each vector, must match `vectors` in length.
        """
        logger.debug(
            f"[Qdrant] Inserting {len(chunks)} vectors into '{collection_name}'"
        )
        points = [
            PointStruct(
                id=chunk.get_point_id(),
                vector=chunk.get_embedding().tolist(),  # type: ignore
                payload=chunk.get_without_embedding().to_dict(),
            )
            for chunk in chunks
        ]

        await self.client.upsert(collection_name=collection_name, points=points)
        logger.info(
            f"[Qdrant] Successfully added {len(points)} chunks to '{collection_name}'"
        )

    async def add_chunk(self, collection_name: str, chunk: ChunkWithEmbedding) -> None:
        """Add a single chunk with embedding to collection.

        Parameters
        ----------
        collection_name : str
            Nmae of the target Qdrant collection.
        chunk : ChunkWithEmbedding
            chunk with embedding to add to collection.
        """
        await self.add_chunks_batch(collection_name=collection_name, chunks=[chunk])

    async def search_without_embeddings(
        self,
        collection_name: str,
        query_vector: NDArray[np.float32],
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_: Filter | None = None,
        with_payload: bool = True,
    ) -> list[ScoredChunk[Chunk]]:
        """
        Search for nearest document chunks given a query embedding.

        Parameters
        ----------
        collection_name : str
            Name of the Qdrant collection to search.
        query_vector : NDArray[np.float32]
            Embedding vector of the query.
        top_k : int, optional
            Number of top results to return, by default 5.
        score_threshold : float, optional
            Minimum similarity score threshold.
        filter_ : Filter, optional
            Optional Qdrant filter to restrict search scope.
        with_payload : bool, optional
            Whether to include payload in the result.

        Returns
        -------
        list[ScoredPoint]
            List of scored points (chunks) sorted by relevance.
        """
        logger.debug(
            f"[Qdrant] Searching in collection '{collection_name}' with top_k={top_k}"
        )
        distance_mode = self.collection_configs[collection_name].distance
        results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),  # Qdrant expects list[float]
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vectors=False,
            query_filter=filter_,
        )
        logger.info(
            f"[Qdrant] Retrieved {len(results)} results from collection '{collection_name}'"
        )
        outputs = []
        for res in results:
            outputs.append(
                ScoredChunk(
                    chunk=Chunk.model_validate(res.payload),
                    score=res.score,
                    distance=distance_mode,
                )
            )
        return outputs

    async def search_with_embeddings(
        self,
        collection_name: str,
        query_vector: NDArray[np.float32],
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_: Filter | None = None,
        with_payload: bool = True,
    ) -> list[ScoredChunk[ChunkWithEmbedding]]:
        """
        Search for nearest document chunks given a query embedding.

        Parameters
        ----------
        collection_name : str
            Name of the Qdrant collection to search.
        query_vector : NDArray[np.float32]
            Embedding vector of the query.
        top_k : int, optional
            Number of top results to return, by default 5.
        score_threshold : float, optional
            Minimum similarity score threshold.
        filter_ : Filter, optional
            Optional Qdrant filter to restrict search scope.
        with_payload : bool, optional
            Whether to include payload in the result.

        Returns
        -------
        list[ScoredPoint]
            List of scored points (chunks) sorted by relevance.
        """
        logger.debug(
            f"[Qdrant] Searching in collection '{collection_name}' with top_k={top_k}"
        )
        distance_mode = self.collection_configs[collection_name].distance
        results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),  # Qdrant expects list[float]
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vectors=True,
            query_filter=filter_,
        )
        logger.info(
            f"[Qdrant] Retrieved {len(results)} results from collection '{collection_name}'"
        )
        outputs: list[ScoredChunk[ChunkWithEmbedding]] = []
        for res in results:
            chunk = ChunkWithEmbedding(
                chunk=Chunk.model_validate(res.payload),
                embedding=np.array(res.vector, dtype=np.float32),
                model_name=res.payload.get("model_name", "default"),
            )
            outputs.append(ScoredChunk(chunk, score=res.score, distance=distance_mode))
        return outputs

    async def remove_chunk_by_id(self, collection_name: str, chunk_id: int) -> None:
        """
        Remove a single chunk by its unique point ID.

        Parameters
        ----------
        collection_name : str
            Name of the collection.

        chunk_id : int
            Unique chunk ID to remove.
        """
        logger.info(
            f"[Qdrant] Removing chunk with ID {chunk_id} from '{collection_name}'"
        )
        await self.client.delete(
            collection_name=collection_name, points_selector={"points": [chunk_id]}
        )
        logger.debug(f"[Qdrant] Chunk {chunk_id} removed")

    async def remove_chunks_by_document_id(
        self, collection_name: str, document_id: int
    ) -> None:
        """
        Remove all chunks associated with a specific document ID.

        Parameters
        ----------
        collection_name : str
            Name of the collection.

        document_id : int
            ID of the document whose chunks should be removed.
        """
        logger.info(
            f"[Qdrant] Removing chunks for document_id={document_id} from '{collection_name}'"
        )
        await self.client.delete(
            collection_name=collection_name,
            points_selector={
                "filter": {
                    "must": [{"key": "document_id", "match": {"value": document_id}}]
                }
            },
        )
        logger.debug(f"[Qdrant] Chunks with document_id={document_id} removed")

    async def clear_collection(self, collection_name: str) -> None:
        """
        Completely remove all data from a collection.

        Parameters
        ----------
        collection_name : str
            Name of the collection to be cleared.
        """
        logger.warning(f"[Qdrant] Clearing all data in collection '{collection_name}'")
        await self.client.delete(
            collection_name=collection_name, points_selector={"all": True}
        )
        logger.info(f"[Qdrant] Collection '{collection_name}' cleared")
