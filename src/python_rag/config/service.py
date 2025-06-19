"""Contains service level config implementation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .chunker import ChunkerConfigUnion
from .qdrant import QdrantCollectionConfig


class ServiceConfig(BaseModel):
    """Service config.

    Attributes
    ----------
    models_pool : list[dict[str, Any]]
        each instance must be prased into triton
        model config.
    chunker : ChunkerConfigUnion
        chunker config.
    text_encoders : dict[str, QdrantCollectionConfig]
        mapping from model alias to qdrant collection config.
        Note that collection name must strictly match triton model name.
    image_encoders : dict[str, QdrantColletionConfig]
        mapping from model alias to qdrant collection config.
        Note that collection name must strictly match triton model name.
    """

    models_pool: list[dict[str, Any]]
    chunker: ChunkerConfigUnion
    text_encoders: dict[str, QdrantCollectionConfig]
    image_encoders: dict[str, QdrantCollectionConfig]
