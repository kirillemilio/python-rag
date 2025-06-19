"""Contains implementation of basic document class."""

from __future__ import annotations

import time
from typing import Iterator

from pydantic import BaseModel, Field

from ..chunker import IChunker
from .chunk import Chunk
from .chunk_interface import IChunk
from .document_interface import IDocument


class Document(BaseModel, IDocument):
    """
    Represents a complete document entry in a SQL-like database.

    Stores metadata and content pointer (if needed) for documents that
    are later split into chunks and indexed in a vector store. Serves
    as a persistent reference for reassembly, source auditing, or UI linking.

    Attributes
    ----------
    document_ id : str
        Unique str identifier for the document.

    title : Optional[str]
        Optional title or label for the document.

    content : str
        text content of the document.

    source : str
        Origin of the document, e.g., 'wikitext', 'upload', or 'url'.

    source_path : Optional[str]
        Optional file path or URL to the source document.

    created_at : float
        POSIX timestamp representing when the document was added to the DB.
    """

    document_id: str
    title: str | None = None
    content: str
    source: str
    tags: list[str] = Field(default_factory=list)
    source_path: str | None = None
    created_at: float = Field(default_factory=lambda: time.time())

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
        for chunk_id, text_chunk in enumerate(chunker.gen_chunks(self.get_content())):
            yield Chunk(
                document_id=self.get_document_id(),
                chunk_id=chunk_id,
                tags=self.get_tags(),
                content=text_chunk,
                source=self.get_source(),
                created_at=self.get_created_at(),
            )
