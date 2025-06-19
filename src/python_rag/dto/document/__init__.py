"""Contains imports related to document hierarchy."""

from .chunk import Chunk
from .chunk_interface import IChunk
from .chunk_with_embedding import ChunkWithEmbedding
from .document import Document
from .document_interface import IDocument

__all__ = ["IDocument", "IChunk", "Chunk", "Document", "ChunkWithEmbedding"]
