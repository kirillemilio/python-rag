"""Contains implementation of charrecter langchain chunker."""

from __future__ import annotations

from typing import Any, Iterator

from langchain.text_splitter import CharacterTextSplitter, TextSplitter

from ...config.chunker import CharChunkerConfig
from .base_chunker import BaseChunker
from .chunker_factory import ChunkerFactory


@ChunkerFactory.register_chunker(chunker_type="char", config_cls=CharChunkerConfig)
class CharChunker(BaseChunker):
    """LangChain character text splitter wrapper."""

    chunk_size: int
    chunk_overlap: int
    separator: str
    splitter: TextSplitter

    def __init__(
        self, chunk_size: int = 512, chunk_overlap: int = 50, separator: str = "\n\n"
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = str(separator)
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separtorr=separator
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
    def from_config(cls, config_dict: dict[str, Any]) -> CharChunker:
        """Create char chunker from config.

        Parameters
        ----------
        config_dict : dict[str, Any]
            raw configuration dictionary for char chunker.

        Returns
        -------
        CharChunker
            char chunker instance.
        """
        config = CharChunkerConfig.model_validate(config_dict)
        return CharChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator=config.separator,
        )
