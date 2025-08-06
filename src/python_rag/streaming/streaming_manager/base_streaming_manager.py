"""Contains implementation of base streaming manager."""

from __future__ import annotations

from typing import Iterator

from ...dto import ChatHistory, LLMStreamItem
from ..streaming_backend import IStreamingBackend
from ..streaming_backend.consumer import BaseStreamConsumer
from ..streaming_backend.producer import BaseStreamProducer
from .streaming_manager_interface import IStreamingManager


class BaseStreamingManager(IStreamingManager):
    """Base streaming manager implementation.

    Attributes
    ----------
    streaming_backend : IStreamingBackend
        streaming backend instance.
    stream_name : str
        stream name that will be used by consumer.
    consumer_group : str
        consumer group that will be used for fetching llm generated
        stream items.
    """

    streaming_backend: IStreamingBackend

    stream_name: str
    consumer_group: str

    def __init__(
        self, streaming_backend: IStreamingBackend, stream_name: str, consumer_group: str
    ) -> None:
        self.streaming_backend = streaming_backend
        self.stream_name = stream_name
        self.consumer_group = consumer_group

    def get_streaming_backend(self) -> IStreamingBackend:
        """Get streaming backend used by streaming manager.

        Returns
        -------
        IStreamingBackend
            streaming backend used by streaming manager.
        """
        return self.streaming_backend

    def get_consumer(self) -> BaseStreamConsumer[LLMStreamItem]:
        """Get stream consumer.

        Returns
        -------
        BaseStreamConsumer[LLMStreamItem]
            stream consumer with llm stream items.
        """
        return self.streaming_backend.get_consumer(
            item_builder=LLMStreamItem,
            stream_name=self.stream_name,
            consumer_group=self.consumer_group,
        )

    def get_producer(self) -> BaseStreamProducer[ChatHistory]:
        """Get stream producer.

        Returns
        -------
        BaseStreamProducer[ChatHistory]
            producer for chat history.
        """
        return self.streaming_backend.get_producer(
            item_builder=ChatHistory, stream_name=self.stream_name
        )
