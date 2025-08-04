"""Contains implementation of redis stream consumer."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Generic, Type, TypeVar

import redis
from pydantic import BaseModel

from ..item_with_ack_status import ItemWithAckStatus
from .base_stream_consumer import BaseStreamConsumer

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger(__name__)


class RedisStreamConsumer(BaseStreamConsumer[T], Generic[T]):
    """Redis stream consumer implementation.

    Attributes
    ----------
    redis : redis.Redis | redis.RedisCluster
        redis instance or cluster that will be used for streaming.
    """

    redis: redis.Redis | redis.RedisCluster

    timeout: int
    buffer_max_size: int
    buffer: Deque[tuple[bytes, bytes]]

    last_raw_item: bytes | None
    last_item_ack: bool
    last_message_id: bytes | None

    def __init__(
        self,
        item_builder: Type[T],
        stream_name: str,
        consumer_group: str,
        consumer_name: str,
        redis_instance: redis.Redis | redis.RedisCluster,  # type: ignore
        timeout: int,
        buffer_max_size: int = 10,
    ) -> None:
        super().__init__(
            item_builder=item_builder,
            stream_name=stream_name,
            consumer_group=consumer_group,
            consumer_name=consumer_name,
        )
        self.timeout = timeout
        self.buffer_max_size = buffer_max_size
        self.buffer = deque(maxlen=self.buffer_max_size)
        self.redis = redis_instance

        self.last_raw_item = None
        self.last_message_id = None
        self.last_item_ack = True

        self.init_consumer_group()

    def init_consumer_group(self) -> None:
        """Initialize consumer group if not exists."""
        stream_groups = set()
        if self.redis.exists(self.stream_name):
            stream_groups = {
                d['name'].decode('utf-8')
                for d in self.redis.xinfo_groups(self.stream_name)  # type: ignore
            }
        if self.consumer_group not in stream_groups:
            self.redis.xgroup_create(self.stream_name, self.consumer_group, mkstream=True)

    def reclaim_pending_messages(self) -> None:
        """Reclaim pending message."""
        messages: list[Any] = self.redis.xpending_range(  # type: ignore
            self.stream_name,
            self.consumer_group,
            min='-',
            max='+',
            count=self.buffer_max_size - len(self.buffer),
            idle=self.timeout,
        )
        if len(messages):
            claimed: list[tuple[bytes, dict[bytes, bytes]]] = self.redis.xclaim(  # type: ignore
                self.stream_name,
                self.consumer_group,
                self.consumer_name,
                min_idle_time=int(self.timeout),
                message_ids=[m['message_id'] for m in messages],
                force=True,
            )
            for message_id, m in claimed:
                if (data := m.get(b'data')) is not None:
                    self.buffer.append((message_id, data))

    def next_raw(self) -> str | None:
        """Get next str item or None from stream consumer.

        Returns
        -------
        str
            raw string encoded item from stream consumer
            if it was able to fetch or None.
        """
        self.reclaim_pending_messages()
        if len(self.buffer) < self.buffer_max_size:
            response: list[bytes, tuple[bytes, dict[bytes, bytes]]] = self.redis.xreadgroup(  # type: ignore
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_name: '>'},
                count=self.buffer_max_size - len(self.buffer),
            )
            for _, messages in response:
                for message_id, fields in messages:
                    if (data := fields.get(b'data')) is not None:
                        self.buffer.append((message_id, data))

        if not len(self.buffer):
            return None

        message_id, item = self.buffer.popleft()
        self.last_raw_item = item
        self.last_message_id = message_id
        self.last_item_ack = False
        return item.decode()

    def get_last_raw_item(self) -> str | None:
        """Get last processed item.

        Returns
        -------
        str | None
            last processed item data or None if
            no data was processed.
        """
        return self.last_raw_item.decode('utf-8') if self.last_raw_item is not None else None

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
            item=self.item_builder.model_validate_json(self.last_raw_item.decode('utf-8')),
            ack_status=self.last_item_ack,
        )

    def on_ack(self) -> None:
        """Run on latest item acknoledgement."""
        logger.info(f'Commiting on item: `{self.get_last_item()}`')
        self.last_item_ack = True
        if self.last_message_id:
            self.redis.xack(self.stream_name, self.consumer_group, self.last_message_id)

    def on_fail(self) -> None:
        """Run on latest item failed."""
        logger.info(f'Failed to process item: `{self.get_last_item()}`')
