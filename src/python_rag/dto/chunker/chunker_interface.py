"""Contains implementation of chunker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class IChunker(ABC):
    """Chunker interface implementation."""

    @abstractmethod
    def gen_chunks(self, text: str) -> Iterator[str]:
        """Generate chunks from text.

        Parameters
        ----------
        text : str
            input text that will be chunked into
            chunks.

        Yields
        ------
        str
            text chunks.
        """
        raise NotImplementedError()
