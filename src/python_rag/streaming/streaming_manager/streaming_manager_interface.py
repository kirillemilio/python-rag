"""Contains implementation of streaming manager interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ...dto import ChatHistory, LLMStreamItem
from ..streaming_backend import IStreamingBackend
from ..streaming_backend.consumer import BaseStreamConsumer
from ..streaming_backend.producer import BaseStreamProducer


class IStreamingManager(ABC):
    """Streaming manager interface implementation."""

    @abstractmethod
    def get_streaming_backend(self) -> IStreamingBackend:
        """Get streaming backend used by streaming manager.
        
        Returns
        -------
        IStreamingBackend
            streaming backend used by streaming manager. 
        """
        raise NotImplementedError()

    @abstractmethod
    def get_consumer(self) -> BaseStreamConsumer[LLMStreamItem]:
        """Get stream consumer.
        
        Returns
        -------
        BaseStreamConsumer[LLMStreamItem] 
            stream consumer with llm stream items.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_producer(self) -> BaseStreamProducer[ChatHistory]:
        """Get stream producer.
        
        Returns
        -------
        BaseStreamProducer[ChatHistory]
            stream producer with chat history for llm
        """
        raise NotImplementedError()

    @abstractmethod
    def schedule_to_worker(self, chat_history: ChatHistory) -> None:
        """Schedule chat history to worker.

        Parameters
        ----------
        chat_history : ChatHistory
            chat history to process with worker.
        """
        raise NotImplementedError()

    @abstractmethod
    def read_stream_from_worker(self) -> Iterator[LLMStreamItem]:
        """Generate items from worker stream."""
        raise NotImplementedError()