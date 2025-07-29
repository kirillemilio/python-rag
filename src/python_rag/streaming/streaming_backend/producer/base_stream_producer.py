"""Contains implementation of base producer class."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Generator, Generic, Self, Type, TypeVar

from pydantic import BaseModel

from ....errors import StreamProducerSendError

T = TypeVar('T', bound=BaseModel)


class BaseStreamProducer(ABC, Generic[T]):
    """Base stream producer implementation."""

    item_builder: Type[T]
    stream_name: str

    def __init__(self, item_builder: Type[T], stream_name: str) -> None:
        """Initialize stream producer.

        Parameters
        ----------
        item_builder : Type[T]
            pydantic model item builder class.
        stream_name : str
            stream name.
        """
        self.item_builder = item_builder
        self.stream_name = stream_name

    def get_stream_name(self) -> str:
        """Get stream name.

        Returns
        -------
        str
            producer stream name.
        """
        return self.stream_name

    @abstractmethod
    def send_raw(self, data: str) -> None:
        """Send raw data.

        Parameters
        ----------
        data : str
            raw data to send.
        """
        raise NotImplementedError()

    def __lshift__(self, other: Any) -> Self:
        """Right lshift operator producer << item.

        Parameters
        ----------
        other : Type[T]
            pydantic model item of type Type[T].

        Returns
        -------
        self
            self stream producer.
        """
        if isinstance(other, self.item_builder):
            raw_data = other.model_dump_json()
            try:
                self.send_raw(raw_data)
            except Exception as e:
                raise StreamProducerSendError(
                    message=f'Error when sending data: {raw_data} to stream `{self.stream_name}`'
                ) from e
            return self
        elif inspect.isgenerator(other):
            for item in other:
                raw_data: str = item.model_dump_json()
                if not isinstance(item, self.item_builder):
                    raise TypeError(f'Generator must have the type of item `{self.item_builder}`')
                try:
                    self.send_raw(raw_data)
                except Exception as e:
                    raise StreamProducerSendError(
                        message=f'Error when sending data: {raw_data} to stream `{self.stream_name}`'
                    ) from e
        raise TypeError(f'Parameter other must have type `{self.item_builder}`')

    def __rrshift__(self, other: Any) -> Self:
        """Right shfit operator for right operatore item >> producer.

        Parameters
        ----------
        other : Type[T]
            pydantic model item of type Type[T].

        Returns
        -------
        Self
            self stream producer.
        """
        return self.__lshift__(other)

    def __call__(self, other: Generator[T] | T) -> Self:
        """Send values to producer.

        Parmaeters
        ----------
        other : Generator[T] | T
            values to add to send.

        Returns
        -------
        Self
            self stream producer.
        """
        return self.__lshift__(other)

    def __repr__(self) -> str:
        """Get string representation of stream producer.

        Returns
        -------
        str
            string representation of stream producer.
        """
        return f'<{self.__class__.__name__}(stream={self.stream_name})>'
