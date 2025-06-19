"""Contains implementation of recursive langchain chunker."""

from __future__ import annotations

from typing import Any, Iterator

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

from ...config.chunker import RecursiveCharChunkerConfig
from .base_chunker import BaseChunker
from .chunker_factory import ChunkerFactory


@ChunkerFactory.register_chunker(
    chunker_type="recursive", config_cls=RecursiveCharChunkerConfig
)
class RecursiveCharChunker(BaseChunker):
    """LangChain RecursiveCharacterTextSplitter wrapper."""

    chunk_size: int
    chunk_overlap: int
    splitter: TextSplitter

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

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
        for chunk in self.splitter.split_text(text):
            yield chunk

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> RecursiveCharChunker:
        """Create recursive char chunker from config.

        Parameters
        ----------
        config_dict : dict[str, Any]
            configuration dictionary for chunker.

        Returns
        -------
        RecursiveCharChunker
            chunker instance.
        """
        config = RecursiveCharChunkerConfig.model_validate(config_dict)
        return RecursiveCharChunker(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
