"""Contains implementation of base chunk class."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, TypeVar

from .document_interface import IDocument

T = TypeVar("T", bound="IChunk")


class IChunk(IDocument):
    """Chunk interface implementation."""

    @abstractmethod
    def get_chunk_id(self) -> int:
        """Get chunk ordered identifier within document.

        Returns
        -------
        int
            chunk ordered identifier within document.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_point_id(self) -> int:
        """Get unieque identifier of qdrant point.

        Returns
        -------
        int
            unique identifier of qdrant point.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Transform chunk to dictionary.

        Returns
        -------
        dict[str, Any]
            dictionary representation of chunk.
        """
        raise NotImplementedError()

    @abstractmethod
    def clone(self: T) -> T:
        """Copy chunk.

        Returns
        -------
        IChunk
            copy of chunk
        """
        raise NotImplementedError()
