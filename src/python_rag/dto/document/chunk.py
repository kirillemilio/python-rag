"""Contains implementation of basic document chunk class."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Iterator

from pydantic import BaseModel, Field

from ..chunker import IChunker
from .chunk_interface import IChunk


class Chunk(BaseModel, IChunk):
    """Represents a single chunk of a document with metadata.

    This class encapsulates a semantically meaningful fragment of a document,
    including its content, source metadata, and additional tags. It provides
    utility methods for retrieval, chunking, and exporting the chunk.

    Designed to be compatible with vector stores and embedding-based pipelines.
    """

    document_id: str
    chunk_id: int
    title: str | None = None
    content: str
    source: str
    tags: list[str] = Field(default_factory=list)
    source_path: str | None = None
    created_at: float = Field(default_factory=lambda: time.time())

    def get_chunk_id(self) -> int:
        """Get chunk id.

        Returns
        -------
        int
            chunk index within document.
        """
        return self.chunk_id

    def get_document_id(self) -> str:
        """Get document unique identifier.

        Returns
        -------
        str
            unique document string identifier.
        """
        return self.document_id

    def get_content(self) -> str:
        """Get document content.

        Returns
        -------
        str
            document content string
        """
        return self.content

    def get_source(self) -> str:
        """Get document source.

        Returns
        -------
        str
            document source.
        """
        return self.source

    def get_tags(self) -> list[str]:
        """Get tags associated with document.

        Returns
        -------
        list[str]
            list of tags associated with the document.
        """
        return self.tags

    def get_created_at(self) -> float:
        """Get created at timestamp as float.

        Returns
        -------
        float
            utc based timestamp of document
            creation.
        """
        return self.created_at

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
        yield self.model_copy()

    def get_point_id(self) -> int:
        """Get unique identifier of qdrant point.

        Returns
        -------
        int
            unique identifier of qdrant point.
        """
        return int.from_bytes(
            hashlib.sha256(f"{self.document_id}{self.chunk_id:04d}".encode()).digest()
        ) % ((1 << 63) - 1)

    def to_dict(self) -> dict[str, Any]:
        """Transform chunk to dictionary.

        Returns
        -------
        dict[str, Any]
            dictionary representation of chunk
        """
        return self.model_dump()

    def clone(self) -> Chunk:
        """Copy chunk.

        Returns
        -------
        Chunk
            copy of chunk.
        """
        return Chunk(
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            title=self.title,
            content=self.content,
            source=self.source,
            tags=self.tags.copy(),
            source_path=self.source_path,
            created_at=self.created_at,
        )
