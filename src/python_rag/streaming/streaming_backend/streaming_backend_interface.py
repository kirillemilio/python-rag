"""Contains implementation of streaming backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

from .consumer import BaseStreamConsumer
from .producer import BaseStreamProducer

T = TypeVar('T', bound=BaseModel)


class IStreamingBackend(ABC):
    """Implements streaming backend."""

    @abstractmethod
    def get_producer(self, item_builder: Type[T], stream_name: str) -> BaseStreamProducer[T]:
        """Get producer for given stream.

        Parameters
        ----------
        item_builder : Type[T]
            item builder to use.
        stream_name : str
            stream name for which producer will be created.

        Returns
        -------
        BaseStreamProducer
            stream producer with given stream name.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_consumer(
        self, item_builder: Type[T], stream_name: str, consumer_group: str
    ) -> BaseStreamConsumer[T]:
        """Get stream consumer for given stream.

        Parameters
        ----------
        item_builder : Type[T]
            item builder to use.
        stream_name : str
            stream name for which consumer will be created.
        consumer_group : str
            consumer group for given consumer.

        Returns
        -------
        BaseStreamConsumer
            stream consumer with given stream name and consumer group.
        """
        raise NotImplementedError()
