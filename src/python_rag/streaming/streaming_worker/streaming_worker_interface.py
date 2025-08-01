"""Streaming worker interface implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...dto import ChatHistory, LLMStreamItem
from ...llm import ILLM
from ..streaming_backend import (
    BaseStreamConsumer,
    BaseStreamProducer,
    IStreamingBackend,
)


class IStreamingWorker(ABC):
    """Streaming worker interface implementation."""

    @abstractmethod
    def get_temperature(self) -> float | None:
        """Get optional sampling temperature.

        Returns
        -------
        float | None
            optional sampling temperature.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_consumer(self) -> BaseStreamConsumer[ChatHistory]:
        """Get input consumer.

        Returns
        -------
        BaseStreamConsumer[ChatHistory]
            stream consumer for given item type.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_producer(self, chat_history: ChatHistory) -> BaseStreamProducer[LLMStreamItem]:
        """Get output producer.

        Parameters
        ----------
        chat_history : ChatHistory
            chat history used to get output stream producer.

        Returns
        -------
        BaseStreamProducer[LLMStreamItem]
            stream producer for given item type.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_streaming_backend(self) -> IStreamingBackend:
        """Get streaming backend.

        Returns
        -------
        IStreamingBackend
            streaming backend instance.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_llm(self) -> ILLM:
        """Get underlying llm instance.

        Returns
        -------
        ILLM
            llm instance used for streaming.
        """
        raise NotImplementedError()

    @abstractmethod
    def run(self) -> None:
        """Run loop of streaming worker."""
        raise NotImplementedError()
