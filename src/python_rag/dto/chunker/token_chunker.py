"""Contains implementation of token langchain chunker."""

from __future__ import annotations

from typing import Any, Iterator

from langchain.text_splitter import TextSplitter, TokenTextSplitter

from ...config.chunker import TokenChunkerConfig
from .base_chunker import BaseChunker
from .chunker_factory import ChunkerFactory


@ChunkerFactory.register_chunker(chunker_type="token", config_cls=TokenChunkerConfig)
class TokenChunker(BaseChunker):
    """LangChain token text splitter wrapper."""

    chunk_size: int
    chunk_overlap: int
    encoding_name: str
    splitter: TextSplitter

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "gpt2",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
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
    def from_config(cls, config_dict: dict[str, Any]) -> TokenChunker:
        """Create token chunker from config.

        Parameters
        ----------
        config_dict : dict[str, Any]
            configuration dictionary for chunker.

        Returns
        -------
        TokenChunker
            chunker instance.
        """
        config = TokenChunkerConfig.model_validate(config_dict)
        return TokenChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            encoding_name=config.encoding_name,
        )
