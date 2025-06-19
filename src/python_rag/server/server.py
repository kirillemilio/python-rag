"""Contains implementation of main python retriever service.

This module initializes and serves a gRPC server that provides
text and image retrieval based on vector similarity. It manages
Triton-based embedding models and Qdrant vector storage.
"""

import argparse
import asyncio
import logging
import time
import uuid
from typing import Mapping

from grpclib import GRPCError
from grpclib.const import Status
from grpclib.server import Server

from ..config.environment import get_env_settings
from ..config.service import ServiceConfig
from ..dto.chunker import ChunkerFactory
from ..dto.document import Document
from ..dto.hander import BaseHandler, IHandler
from ..proto.image import ImageRequest
from ..proto.retriever import RetrieverBase
from ..proto.text import (
    DocumentChunk,
    TextChunksResponse,
    TextDocumentAddRequest,
    TextDocumentAddResponse,
    TextRequest,
)
from ..qdrant import QdrantManagerHodler
from ..triton import TritonModelsPoolHolder
from ..triton.image_encoders import BaseImageEncoderTritonModel
from ..triton.text_encoders import BaseTextEncoderTritonModel
from .utils import ConfigLoader

logging.basicConfig(format="%(asctime)s-%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__file__)


class Retriever(RetrieverBase):
    """gRPC Retriever service implementation.

    Implements RPC methods for adding documents and performing
    text/image-based semantic search using encoder models and
    vector database.

    Parameters
    ----------
    handler : IHandler
        Handler that manages chunking, encoding and Qdrant operations.
    text_encoder_alias_to_col : Mapping[str, str]
        Mapping from alias to Qdrant collection for text encoders.
    image_encoder_alias_to_col : Mapping[str, str]
        Mapping from alias to Qdrant collection for image encoders.
    """

    handler: IHandler
    text_encoder_alias_to_col: Mapping[str, str]
    image_encoder_alias_to_col: Mapping[str, str]

    def __init__(
        self,
        handler: IHandler,
        text_encoder_alias_to_col: Mapping[str, str],
        image_encoder_alias_to_col: Mapping[str, str],
    ) -> None:
        self.handler = handler
        self.text_encoder_alias_to_col = text_encoder_alias_to_col
        self.image_encoder_alias_to_col = image_encoder_alias_to_col

    async def add_text_document(
        self, message: TextDocumentAddRequest
    ) -> TextDocumentAddResponse:
        """Add a text document to the vector store.

        Parameters
        ----------
        message : TextDocumentAddRequest
            Request containing document content, source and tags.

        Returns
        -------
        TextDocumentAddResponse
            Dummy response (not used currently).

        Raises
        ------
        GRPCError
            Raised for unimplemented or invalid document state.
        """
        document = Document(
            document_id=uuid.uuid4().hex,
            title=None,
            content=message.text,
            source=message.source,
            tags=message.tags,
            created_at=time.time(),
        )
        try:
            await self.handler.add(document=document)
            return TextDocumentAddResponse(
                request_id=message.request_id,
                document_id=document.get_document_id(),
                created_at=document.get_created_at(),
            )
        except Exception as e:
            logger.exception("Failed to add document")
            raise GRPCError(status=Status.INTERNAL, message=str(e))

    async def search_by_image(self, message: ImageRequest) -> TextChunksResponse:
        """Search similar documents based on an image embedding.

        Parameters
        ----------
        message : ImageRequest
            gRPC request with image data and model name.

        Returns
        -------
        TextChunksResponse
            List of document chunks sorted by similarity.

        Raises
        ------
        GRPCError
            If the model or collection is not found.
        """
        collection = self.image_encoder_alias_to_col.get(message.model)
        if collection is None:
            raise GRPCError(
                status=Status.NOT_FOUND,
                message=f"Collection or model not found: `{message.model}`",
            )
        try:
            scored_chunks = await self.handler.search_by_image(request=message)
            chunks = []
            for scored_chunk in scored_chunks:
                chunk = scored_chunk.get_chunk()
                chunks.append(
                    DocumentChunk(
                        document_id=chunk.get_document_id(),
                        content=chunk.get_content(),
                        source=chunk.get_source(),
                        collection=collection,
                        tags=chunk.get_tags(),
                        embedding=chunk.get_embedding().tolist(),  # type: ignore
                        created_at=chunk.get_created_at(),
                    )
                )
            return TextChunksResponse(request_id=message.request_id, chunks=chunks)
        except Exception as e:
            logger.exception("Failed to perform image search")
            raise GRPCError(status=Status.INTERNAL, message=str(e))

    async def search_by_text(self, message: TextRequest) -> TextChunksResponse:
        """Search similar documents based on a text embedding.

        Parameters
        ----------
        message : TextRequest
            gRPC request with text and model name.

        Returns
        -------
        TextChunksResponse
            List of document chunks sorted by similarity.

        Raises
        ------
        GRPCError
            If the model or collection is not found.
        """
        collection = self.text_encoder_alias_to_col.get(message.model)
        if collection is None:
            raise GRPCError(
                status=Status.NOT_FOUND,
                message=f"Text collection or model not found: `{message.model}`",
            )
        try:
            scored_chunks = await self.handler.search_by_text(request=message)
            chunks = []
            for scored_chunk in scored_chunks:
                chunk = scored_chunk.get_chunk()
                chunks.append(
                    DocumentChunk(
                        document_id=chunk.get_document_id(),
                        content=chunk.get_content(),
                        source=chunk.get_source(),
                        collection=collection,
                        tags=chunk.get_tags(),
                        embedding=chunk.get_embedding().tolist(),  # type: ignore
                        created_at=chunk.get_created_at(),
                    )
                )
            return TextChunksResponse(request_id=message.request_id, chunks=chunks)
        except Exception as e:
            logger.exception("Failed to perform text search")
            raise GRPCError(status=Status.INTERNAL, message=str(e))


async def serve(host: str, port: int, config: str) -> None:
    """Serve the gRPC server for document retrieval.

    Load service config, initialize models and Qdrant,
    and run the async server loop.

    Parameters
    ----------
    host : str
        Host address to bind the gRPC server to.
    port : int
        Port number for the gRPC server.
    config : str
        Path to the service configuration file.
    """
    env_settings = get_env_settings()
    logger.info(f"Environment settings: {env_settings}")
    service_config: ServiceConfig = ConfigLoader.load_config(
        config, config_class=ServiceConfig
    )

    col_configs = {
        col_config.name: col_config
        for col_config in service_config.image_encoders.values()
    }
    col_configs |= {
        col_config.name: col_config
        for col_config in service_config.text_encoders.values()
    }
    TritonModelsPoolHolder.init(models_pool_config=service_config.models_pool)
    await QdrantManagerHodler.init(
        qdrant_config=env_settings.qdrant, collection_configs=col_configs
    )

    chunker = ChunkerFactory.create_chunker(service_config.chunker.model_dump())

    image_encoders: dict[str, BaseImageEncoderTritonModel] = {}
    for encoder_name, col_config in service_config.image_encoders.items():
        image_encoders[
            encoder_name
        ] = TritonModelsPoolHolder.get_models_pool().get_model(
            col_config.name, model_type=BaseImageEncoderTritonModel  # type: ignore
        )

    text_encoders: dict[str, BaseTextEncoderTritonModel] = {}
    for encoder_name, col_config in service_config.text_encoders.items():
        text_encoders[
            encoder_name
        ] = TritonModelsPoolHolder.get_models_pool().get_model(
            col_config.name, model_type=BaseTextEncoderTritonModel  # type: ignore
        )

    handler = BaseHandler(
        text_encoders=text_encoders,
        image_encoders=image_encoders,
        chunker=chunker,
        qdrant_manager=QdrantManagerHodler.get(),
    )

    server = Server(
        [
            Retriever(
                handler=handler,
                text_encoder_alias_to_col={
                    k: v.model_name for k, v in text_encoders.items()
                },
                image_encoder_alias_to_col={
                    k: v.model_name for k, v in image_encoders.items()
                },
            )
        ]
    )
    await server.start(host=host, port=port)
    await server.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Python preprocessor service for embeddings")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50081)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    host: str = args.host
    port: int = args.port
    config: str = args.config
    asyncio.get_event_loop().run_until_complete(
        serve(host=host, port=port, config=config)
    )
