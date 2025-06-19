"""Contains implementation of chunker classes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class BaseChunkerConfig(BaseModel):
    """Base class for all chunker configuration models.

    Serves as the parent model for specific chunker configurations.
    Provides a common interface for runtime chunker creation.

    Attributes
    ----------
    chunker_type : str
        Identifier for the chunker type, used to resolve concrete implementation.
    """

    chunker_type: str


class RecursiveCharChunkerConfig(BaseChunkerConfig):
    """Configuration for recursive character-based chunker.

    Splits text recursively using increasingly fine-grained separators
    (e.g. paragraphs → sentences → words) until the desired chunk size is reached.

    Attributes
    ----------
    chunker_type : Literal["recursive"]
        Identifier for recursive char chunker.
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
    """

    chunker_type: Literal["recursive"]
    chunk_size: int = 512
    chunk_overlap: int = 50


class CharChunkerConfig(BaseChunkerConfig):
    """Configuration for flat character-based chunker.

    Performs straightforward splitting of text into fixed-size chunks
    using a specified separator. Suitable for simple and fast chunking.

    Attributes
    ----------
    chunker_type : Literal["char"]
        Identifier for char chunker.
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of characters to overlap between chunks.
    separator : str
        Separator string used to split the text.
    """

    chunker_type: Literal["char"]
    chunk_size: int = 512
    chunk_overlap: int = 50
    separator: str = "\n\n"


class TokenChunkerConfig(BaseChunkerConfig):
    """Configuration for token-based chunker.

    Splits text based on token count using a specific tokenizer backend.
    Provides better alignment with LLM input requirements.

    Attributes
    ----------
    chunker_type : Literal["token"]
        Identifier for token-based chunker.
    chunk_size : int
        Maximum number of tokens per chunk.
    chunk_overlap : int
        Number of tokens to overlap between chunks.
    encoding_name : str
        Name of the tokenizer encoding (e.g., "gpt2", "cl100k_base").
    """

    chunker_type: Literal["token"]
    chunk_size: int = 512
    chunk_overlap: int = 50
    encoding_name: str = "gpt2"


ChunkerConfigUnion = TokenChunkerConfig | CharChunkerConfig | RecursiveCharChunkerConfig
