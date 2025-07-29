"""Contains implementation of kafka stream consumer."""

from __future__ import annotations

import logging
from typing import Generic, Type, TypeVar

import kafka
from pydantic import BaseModel

from ..item_with_ack_status import ItemWithAckStatus
from .base_stream_consumer import BaseStreamConsumer

T = TypeVar('T', bound=BaseModel)


logger = logging.getLogger(__name__)


class KafkaStreamConsumer(BaseStreamConsumer[T], Generic[T]):
    """Kafka stream consumer implementation.

    Attributes
    ----------
    kafka_servers : list[str]
        list of kafka boostrap servers.
    auto_offset_reset : str
        auto offset reset policy.
        Default is 'earliest'.
    enable_autocommit : bool
        whether to enable autocommit or not.
        Default is False.
    """

    auto_offset_reset: str
    enable_autocommit: bool
    kafka_servers: list[str]
    consumer: kafka.KafkaConsumer

    last_raw_item: str | None
    last_item_ack: bool

    def __init__(
        self,
        item_builder: Type[T],
        stream_name: str,
        consumer_group: str,
        consumer_name: str,
        kafka_servers: list[str],
        auto_offset_reset: str = 'earliest',
        enable_autocommit: bool = False,
    ) -> None:
        super().__init__(
            item_builder=item_builder,
            stream_name=stream_name,
            consumer_group=consumer_group,
            consumer_name=consumer_name,
        )
        self.auto_offset_reset = auto_offset_reset
        self.enable_autocommit = enable_autocommit
        self.consumer = kafka.KafkaConsumer(
            self.stream_name,
            group_id=self.consumer_group,
            bootstrap_servers=kafka_servers,
            auto_offset_reset=self.auto_offset_reset,
            enable_autocommit=self.enable_autocommit,
            value_deserializer=lambda m: m.decode('utf-8'),
        )
        self.last_item_ack = True
        self.last_raw_item = None

    def next_raw(self) -> str:
        """Get next str item from stream consumer.

        Returns
        -------
        str
            raw string encoded item from stream consumer.
        """
        item = self.consumer.__next__().value
        self.last_raw_item = item
        self.last_item_ack = False
        return item

    def get_last_raw_item(self) -> str | None:
        """Get last processed item.

        Returns
        -------
        str | None
            last processed item data or None if
            no data was processed.
        """
        return self.last_raw_item

    def get_last_item(self) -> ItemWithAckStatus[T] | None:
        """Get last parsed item with ack status.

        Returns
        -------
        ItemWithAckStatus[T] | None
            item with ack status if any item was processed
            or None otherwise.
        """
        if self.last_raw_item is None:
            return None
        return ItemWithAckStatus(
            item=self.item_builder.model_validate_json(self.last_raw_item),
            ack_status=self.last_item_ack,
        )

    def on_ack(self) -> None:
        """Run on latest item acknoledgement."""
        logger.info(f'Commiting on item: `{self.get_last_item()}`')
        self.consumer.commit()
        self.last_item_ack = True

    def on_fail(self) -> None:
        """Run on latest item failed."""
        logger.info(f'Failed to process item: `{self.get_last_item()}`')
