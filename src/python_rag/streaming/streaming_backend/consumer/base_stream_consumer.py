"""Contains implementation of base consumer class."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Self, Type, TypeVar

from pydantic import BaseModel

from ..confirmration_state import EConsumerConfirmationState

T = TypeVar('T', bound=BaseModel)


logger = logging.getLogger(__name__)


class BaseStreamConsumer(ABC, Generic[T]):
    """Base stream consumer implementation."""

    item_builder: Type[T]
    stream_name: str
    consumer_group: str
    consumer_name: str

    def __init__(
        self, item_builder: Type[T], stream_name: str, consumer_group: str, consumer_name: str
    ) -> None:
        """Initialize stream consumer.

        Parameters
        ----------
        item_builder : Type[T]
            pydantic model item builder class.
        stream_name : str
            stream name.
        consumer_group : str
            consumer group name.
        consumer_name : str
            consumer name within group.
        """
        self.item_builder = item_builder
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name

    @abstractmethod
    def on_ack(self) -> None:
        """Run on last item acknoledgement."""
        raise NotImplementedError()

    @abstractmethod
    def on_fail(self) -> None:
        """Run on last item failed."""
        raise NotImplementedError()

    @abstractmethod
    def next_raw(self) -> str | None:
        """Get next str item or None from stream consumer.

        Returns
        -------
        str
            raw string encoded item from stream consumer
            if it was able to fetch or None.
        """
        raise NotImplementedError()

    def get_consumer_name(self) -> str:
        """Get name of consumer within consumer group.

        Returns
        -------
        str
            consumer name within group.
        """
        return self.consumer_name

    def get_consumer_group(self) -> str:
        """Get name of consumer group.

        Returns
        -------
        str
            consumer group name.
        """
        return self.consumer_group

    def get_stream_name(self) -> str:
        """Get stream name.

        Returns
        -------
        str
            stream name.
        """
        return self.stream_name

    def __next__(self) -> T:
        """Get next item from stream consumer.

        Returns
        -------
        BaseModel
            pydantic item.
        """
        while True:
            try:
                raw_item = self.next_raw()
            except StopIteration as e:
                raise e
            except Exception:
                logger.error(
                    f'Error when fetching raw item: {raw_item},'
                    + f' stream `{self.stream_name}`, consumer_group `{self.consumer_group}`, '
                    + f' consumer `{self.consumer_name}`',
                    exc_info=True,
                )
                continue

            if raw_item is None:
                time.sleep(0.05)
                continue

            try:
                item = self.item_builder.model_validate_json(raw_item)
            except Exception:
                logger.error(
                    f'Error when parsing item: {raw_item} to {self.item_builder}, '
                    + f' stream `{self.stream_name}`, consumer_group `{self.consumer_group}`, '
                    + f' consumer `{self.consumer_name}`',
                    exc_info=True,
                )
                continue

            logger.info(f'Fetched item from stream: {item}')
            return item

    def __iter__(self) -> Self:
        """Get iterator of stream consumer.

        Returns
        -------
        Self
            always self.
        """
        return self

    def __lshift__(self, other: Any) -> Self:
        """Left shift operator consumer << confirmation_state.

        Parameters
        ----------
        other : EConsumerConfirmationState
            consumer confirmation state to send.

        Returns
        -------
        Self
            self stream consumer.
        """
        if isinstance(other, EConsumerConfirmationState):
            match other:
                case EConsumerConfirmationState.ACK:
                    self.on_ack()
                    return self
                case EConsumerConfirmationState.FAILED:
                    self.on_fail()
                    return self
                case _:
                    raise ValueError(f'Unknown consumer confirmation state: {other}')

        raise TypeError('Value must be of type EConsumerConfirmationState')

    def __rrshift__(self, other: Any) -> Self:
        """Right shift operator for right operator confirmation_state >> consumer.

        Parameters
        ----------
        other : EConsumerConfirmationState
            consumer confirmation state to send.

        Returns
        -------
        Self
            self stream consumer.
        """
        return self.__lshift__(other)

    def __repr__(self) -> str:
        """Get string representation of stream consumer.

        Returns
        -------
        str
            string representation of stream consumer.
        """
        return (
            f'<{self.__class__.__name__}(stream={self.stream_name}, group={self.consumer_group})>'
        )
