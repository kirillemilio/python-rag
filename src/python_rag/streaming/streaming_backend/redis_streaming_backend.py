"""Contains implementation of redis streaming backend."""

from __future__ import annotations

import uuid
from typing import Any, Type, TypeVar

import redis
from pydantic import BaseModel

from ...config.streaming_backend import RedisStreamingBackendConfig
from .base_streaming_backend import BaseStreamingBackend
from .consumer import RedisStreamConsumer
from .producer import RedisStreamProducer
from .streaming_backend_factory import StreamingBackendFactory

T = TypeVar('T', bound=BaseModel)


@StreamingBackendFactory.register_streaming_backend('redis', config_cls=RedisStreamingBackendConfig)
class RedisStreamingBackend(BaseStreamingBackend):
    """
    Redis-based implementation of the streaming backend.

    This backend provides stream producer and consumer interfaces
    over Redis streams. It supports configurable buffering and
    acknowledgment timeout settings.

    Attributes
    ----------
    redis : redis.Redis | redis.RedisCluster
        Instance of Redis client used for streaming operations.
    buffer_max_size : int
        Maximum number of buffered messages in memory.
    stream_ack_timeout : int
        Time (in milliseconds) after which unacknowledged messages
        are considered stale and eligible for reclaim.
    """

    redis: redis.Redis | redis.RedisCluster
    buffer_max_size: int
    stream_ack_timeout: int

    def __init__(
        self,
        redis_instance: redis.Redis | redis.RedisCluster,
        buffer_max_size: int = 10,
        stream_ack_timeout: int = 30000,
    ) -> None:
        """
        Initialize the Redis streaming backend.

        Parameters
        ----------
        redis_instance : redis.Redis | redis.RedisCluster
            A Redis client instance used for reading and writing to streams.
        buffer_max_size : int, optional
            Maximum number of items to keep in the consumer buffer
            (default is 10).
        stream_ack_timeout : int, optional
            Idle time in milliseconds after which a message will be reclaimed
            if not acknowledged (default is 30000 ms).
        """
        super().__init__()
        self.redis = redis_instance
        self.buffer_max_size = buffer_max_size
        self.stream_ack_timeout = stream_ack_timeout

    def get_consumer(
        self, item_builder: Type[T], stream_name: str, consumer_group: str
    ) -> RedisStreamConsumer[T]:
        """
        Create a new Redis stream consumer instance.

        Parameters
        ----------
        item_builder : Type[T]
            A Pydantic model used to parse consumed data.
        stream_name : str
            Name of the Redis stream.
        consumer_group : str
            Name of the consumer group.

        Returns
        -------
        RedisStreamConsumer[T]
            A configured Redis stream consumer.
        """
        rnd_name = uuid.uuid4().hex
        return RedisStreamConsumer(
            item_builder=item_builder,
            stream_name=stream_name,
            consumer_group=consumer_group,
            consumer_name=f'{consumer_group}-{rnd_name[:16]}',
            redis_instance=self.redis,
            timeout=self.stream_ack_timeout,
            buffer_max_size=self.buffer_max_size,
        )

    def get_producer(self, item_builder: Type[T], stream_name: str) -> RedisStreamProducer[T]:
        """
        Create a new Redis stream producer instance.

        Parameters
        ----------
        item_builder : Type[T]
            A Pydantic model used to validate and serialize data.
        stream_name : str
            Name of the Redis stream to produce to.

        Returns
        -------
        RedisStreamProducer[T]
            A configured Redis stream producer.
        """
        return RedisStreamProducer(
            item_builder=item_builder, stream_name=stream_name, redis_instance=self.redis
        )

    def get_redis(self) -> redis.Redis | redis.RedisCluster:
        """Get underlying redis instance.

        Returns
        -------
        redis.Redis | redis.RedisCluster
            redis instance.
        """
        return self.redis

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> RedisStreamingBackend:
        """Create redis streaming backend from raw configuration dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            configuration dictionary to use to create redis streaming backend.

        Returns
        -------
        RedisStreamingBackend
            redis streaming backend instance created from config.
        """
        config = RedisStreamingBackendConfig.model_validate(config_dict)
        redis_instance = redis.Redis(
            host=config.host, port=config.port, db=config.db, password=config.password
        )
        return RedisStreamingBackend(
            redis_instance=redis_instance,
            buffer_max_size=config.buffer_max_size,
            stream_ack_timeout=config.stream_ack_timeout,
        )
