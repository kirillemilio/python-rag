"""Contains implementation of base handler."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from ...dto.document import IDocument
from ...proto.image import ImageRequest
from ...proto.text import TextRequest
from ...qdrant import QdrantManager, ScoredChunk
from ...triton.image_encoders import BaseImageEncoderTritonModel
from ...triton.text_encoders import BaseTextEncoderTritonModel
from ..chunker import IChunker
from ..document import ChunkWithEmbedding
from .hander_interface import IHandler

logger = logging.getLogger(__name__)


class BaseHandler(IHandler):
    """
    Base implementation of the IHandler interface.

    Provides basic functionality for adding documents to a Qdrant index
    and performing vector-based searches over text and image inputs.
    Supports multiple models per modality (text/image).

    Parameters
    ----------
    text_encoders : Mapping[str, BaseTextEncoderTritonModel]
        Dictionary mapping model names to text encoder instances.
    image_encoders : Mapping[str, BaseImageEncoderTritonModel]
        Dictionary mapping model names to image encoder instances.
    qdrant_manager : QdrantManager
        Manager responsible for interacting with Qdrant collections.
    chunker : IChunker
        Chunker used to split documents into smaller content chunks.
    collection_name : str
        Default name of the collection used for fallback operations.
    """

    image_encoders: Mapping[str, BaseImageEncoderTritonModel]
    text_encoders: Mapping[str, BaseTextEncoderTritonModel]

    chunker: IChunker
    qdrant_manager: QdrantManager

    def __init__(
        self,
        text_encoders: Mapping[str, BaseTextEncoderTritonModel],
        image_encoders: Mapping[str, BaseImageEncoderTritonModel],
        qdrant_manager: QdrantManager,
        chunker: IChunker,
    ) -> None:
        self.chunker = chunker
        self.qdrant_manager = qdrant_manager
        self.image_encoders = image_encoders
        self.text_encoders = text_encoders

    async def add(self, document: IDocument) -> bool:
        """
        Add document to index by chunking and encoding its content.

        Use the provided chunker to split the document into chunks,
        encode each chunk using all available text encoders, and
        store them in Qdrant collections under corresponding model names.

        Parameters
        ----------
        document : IDocument
            Document to add to the vector index.

        Returns
        -------
        bool
            True if all chunks were added successfully, False otherwise.
        """
        exc_happened: bool = False
        for chunk in document.gen_chunks(chunker=self.chunker):
            for encoder_name, text_encoder in self.text_encoders.items():
                chunk_with_embedding = ChunkWithEmbedding(
                    chunk=chunk,
                    embedding=text_encoder.apply(input=chunk.get_content()),
                    model_name=text_encoder.model_name,
                )
                try:
                    await self.qdrant_manager.add_chunk(
                        collection_name=text_encoder.model_name,
                        chunk=chunk_with_embedding,
                    )
                except Exception:
                    logger.error(
                        "Error happened when adding chunk to collection", exc_info=True
                    )
                    exc_happened = True
        return not exc_happened

    async def search_by_text(
        self, request: TextRequest, top_k: int = 8
    ) -> Sequence[ScoredChunk[ChunkWithEmbedding]]:
        """
        Perform vector search using a text query.

        Encode the input query text using the specified text encoder model,
        and retrieve top_k closest chunks from the Qdrant index.

        Parameters
        ----------
        request : TextRequest
            Input request containing query text and model name.
        top_k : int, optional
            Number of top results to return, by default 8.

        Returns
        -------
        Sequence[ScoredChunk]
            List of scored chunks returned by Qdrant.
        """
        if (model := self.text_encoders.get(request.model)) is not None:
            embedding = model.apply(request.text)
            return await self.qdrant_manager.search_with_embeddings(
                collection_name=model.model_name,
                query_vector=embedding,
                top_k=top_k,
            )
        return []

    async def search_by_image(
        self, request: ImageRequest, top_k: int = 8
    ) -> Sequence[ScoredChunk[ChunkWithEmbedding]]:
        """
        Perform vector search using an image query.

        Decode the input image, extract its embedding using the specified
        image encoder model, and retrieve top_k closest chunks from Qdrant.

        Parameters
        ----------
        request : ImageRequest
            Input request containing image bytes and model name.
        top_k : int, optional
            Number of top results to return, by default 8.

        Returns
        -------
        Sequence[ScoredChunk]
            List of scored chunks returned by Qdrant.
        """
        if (model := self.image_encoders.get(request.model)) is not None:
            image: NDArray[np.uint8] = cv2.imdecode(
                np.asarray(bytearray(request.image), dtype=np.uint8),
                cv2.IMREAD_ANYCOLOR,
            )  # type: ignore
            embedding = model.apply(image)
            if (text_encoder := self.text_encoders.get(request.model)) is not None:
                return await self.qdrant_manager.search_with_embeddings(
                    collection_name=text_encoder.model_name,
                    query_vector=embedding,
                    top_k=top_k,
                )

        return []
