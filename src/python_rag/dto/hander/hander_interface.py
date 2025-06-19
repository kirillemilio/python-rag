"""Contains imports of components related to handler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ...dto.document import ChunkWithEmbedding
from ...proto.image import ImageRequest
from ...proto.text import TextRequest
from ...qdrant import ScoredChunk
from ..document import IDocument


class IHandler(ABC):
    """Handler interface implementation.

    Handlers are responsible for handling the full pipeline
    for add document to index or searching for related documents by query.
    """

    @abstractmethod
    async def add(self, document: IDocument) -> bool:
        """Process document and add to index.

        Parameters
        ----------
        document : IDocument
            document to add to index.

        Returns
        -------
        bool
            whether adding document was successfull or not.
        """
        raise NotImplementedError()

    @abstractmethod
    async def search_by_text(
        self, request: TextRequest
    ) -> Sequence[ScoredChunk[ChunkWithEmbedding]]:
        """Search for chunks in database.

        Parameters
        ----------
        request : TextRequest
            text request to search in database.

        Returns
        -------
        Sequence[ScoredChunk]
            sequence of found scored chunks.
        """
        raise NotImplementedError()

    @abstractmethod
    async def search_by_image(
        self, request: ImageRequest
    ) -> Sequence[ScoredChunk[ChunkWithEmbedding]]:
        """Search for chunks in database.

        Parameters
        ----------
        request : ImageRequest
            image request to search in database.

        Returns
        -------
        Sequence[ScoredChunk]
            sequence of found scored chunks
        """
        raise NotImplementedError()
