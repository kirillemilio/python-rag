"""Api service implemntation for rag system."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, ClassVar

from fastapi import FastAPI
from grpclib.client import Channel

from ..config.environment import get_env_settings
from ..dto.document import Document
from ..proto.retriever import RetrieverStub

logger = logging.getLogger(__name__)


class RetriverClientHolder:
    channel: ClassVar[Channel | None] = None
    retriever: ClassVar[RetrieverStub | None] = None

    @classmethod
    async def init(cls) -> None:
        """Initialize retriever client."""
        env_settings = get_env_settings()
        host = env_settings.grpc_server.host
        port = env_settings.grpc_server.port
        cls.channel = Channel(host, port=port)
        cls.retriever = RetrieverStub(cls.channel)

    @classmethod
    async def get(cls) -> RetrieverStub:
        """Get initilized retriever stub.

        Returns
        -------
        RetrieverStub
            grpc retriever client.

        Raises
        ------
        RuntimeError
            if grpc client was not initialized.
        """
        if cls.retriever is None:
            raise RuntimeError(
                'Retriever grpc client is not initalized. '
                + 'Consider calling init() method first.'
            )
        return cls.retriever

    @classmethod
    async def cleanup(cls) -> None:
        """Clean up resouces consumed by retriever client."""
        if cls.channel is not None:
            cls.channel.close()
            cls.channel = None
            cls.retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan events for fastapi service."""
    await RetriverClientHolder.init()
    yield
    await RetriverClientHolder.cleanup()


app = FastAPI()


# @app.post("/add-document")
# async def add_document(document: Document):
