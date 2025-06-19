"""Contains imports related to documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

from ..chunker import IChunker

if TYPE_CHECKING:
    from .chunk_interface import IChunk


class IDocument(ABC):
    """Document interface implementation."""

    @abstractmethod
    def get_document_id(self) -> str:
        """Get document unique identifier.

        Returns
        -------
        str
            unique document string identifier.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_content(self) -> str:
        """Get document content.

        Returns
        -------
        str
            document content string.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_source(self) -> str:
        """Get document source.

        Returns
        -------
        str
            document source.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_tags(self) -> list[str]:
        """Get list of tags associated with document.

        Returns
        -------
        list[str]
            list of tags associated with document.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_created_at(self) -> float:
        """Get created at timestamp as float.

        Returns
        -------
        float
            created at timestamp.
        """
        raise NotImplementedError()

    @abstractmethod
    def gen_chunks(self, chunker: IChunker) -> Iterator[IChunk]:
        """Generate chunks from document.

        Parameters
        ----------
        chunker : IChunker
            chunker instance that will be used for
            chunking document.

        Yields
        ------
        IChunk
            document chunk.
        """
        raise NotImplementedError()
