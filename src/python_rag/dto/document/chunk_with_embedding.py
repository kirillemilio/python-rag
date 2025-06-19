"""Contains implementation of vectorized chunk containing embeddings."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from ..chunker import IChunker
from .chunk_interface import IChunk


class ChunkWithEmbedding(IChunk):
    """Wrapper class that combines a document chunk with its embedding vector.

    This class decorates a base `IChunk` implementation with an additional
    NumPy-based embedding and the model name that produced it. It preserves
    the full interface of a chunk and adds vector access for similarity search,
    retrieval, or downstream ML usage.
    """

    chunk: IChunk
    model_name: str
    embedding: NDArray[np.float32]

    def __init__(
        self, chunk: IChunk, embedding: NDArray[np.float32], model_name: str
    ) -> None:
        """
        Initialize a chunk with its corresponding embedding vector and model metadata.

        Parameters
        ----------
        chunk : IChunk
            The base chunk instance containing document metadata and content.
        embedding : NDArray[np.float32]
            The embedding vector representing the semantic content of the chunk.
        model_name : str
            Name of the model that generated the embedding.
        """
        self.chunk = chunk
        self.embedding = embedding
        self.model_name = model_name

    def get_chunk_id(self) -> int:
        """Get chunk id.

        Returns
        -------
        int
            chunk index within document.
        """
        return self.chunk.get_chunk_id()

    def get_document_id(self) -> str:
        """Get document unique identifier.

        Returns
        -------
        str
            unique document string identifier.
        """
        return self.chunk.get_document_id()

    def get_content(self) -> str:
        """Get document content.

        Returns
        -------
        str
            document content string
        """
        return self.chunk.get_content()

    def get_source(self) -> str:
        """Get document source.

        Returns
        -------
        str
            document source.
        """
        return self.chunk.get_source()

    def get_tags(self) -> list[str]:
        """Get tags associated with document.

        Returns
        -------
        list[str]
            list of tags associated with the document.
        """
        return self.chunk.get_tags()

    def get_created_at(self) -> float:
        """Get created at timestamp as float.

        Returns
        -------
        float
            utc based timestamp of document
            creation.
        """
        return self.chunk.get_created_at()

    def gen_chunks(self, chunker: IChunker) -> Iterator[IChunk]:
        """Generate chunks using chunker.

        Parameters
        ----------
        chunker : IChunker
            chunker that will be used
            for generatin chunks from document.

        Yields
        ------
        IChunk
            chunk related to the document.
        """
        yield self.clone()

    def get_point_id(self) -> int:
        """Get unique identifier of qdrant point.

        Returns
        -------
        int
            unique identifier of qdrant point.
        """
        return self.chunk.get_point_id()

    def to_dict(self) -> dict[str, Any]:
        """Transform chunk to dictionary.

        Returns
        -------
        dict[str, Any]
            dictionary representation of chunk
        """
        return {**self.chunk.to_dict(), "embedding": self.embedding.tolist()}

    def get_embedding(self, clone: bool = False) -> NDArray[np.float32]:
        """Get embedding vector.

        Parameters
        ----------
        clone : bool
            whether to return deep copy or shallow
            copy of underlying embedding vector.
            Default is False.

        Returns
        -------
        NDArray[np.float32]
            underlying embedding vector.
        """
        return self.embedding.copy() if clone else self.embedding

    def get_embedding_model_name(self) -> str:
        """Get name of embedding model.

        Returns
        -------
        str
            name of embedding model.
        """
        return self.model_name

    def get_without_embedding(self, clone: bool = False) -> IChunk:
        """Get chunk without embedding.

        Parameters
        ----------
        clone : bool
            whether to clone underlying chunk
            or return a reference.
            Default is False meaning that
            reference to underlying chunk will be returned.
        """
        return self.chunk if not clone else self.chunk.clone()

    def clone(self) -> ChunkWithEmbedding:
        """Copy chunk with embeddings.

        Returns
        -------
        ChunkWithEmbeddings
            copy of original chunk with embeding.
        """
        return ChunkWithEmbedding(
            chunk=self.chunk.clone(),
            embedding=self.embedding.copy(),
            model_name=self.model_name,
        )
