"""Contains implementation of redis stream producer."""

from __future__ import annotations

from typing import Generic, Type, TypeVar

import redis
from pydantic import BaseModel

from .base_stream_producer import BaseStreamProducer

T = TypeVar('T', bound=BaseModel)


class RedisStreamProducer(BaseStreamProducer[T], Generic[T]):
    """Implementation of redis stream producer.

    Attributes
    ----------
    redis : redis.Redis | redis.RedisCluster
        redis instance that will be used for pushing stream data.
    """

    redis: redis.Redis | redis.RedisCluster

    def __init__(
        self,
        item_builder: Type[T],
        stream_name: str,
        redis_instance: redis.Redis | redis.RedisCluster,
    ) -> None:
        super().__init__(item_builder=item_builder, stream_name=stream_name)
        self.redis = redis_instance

    def get_redis(self) -> redis.Redis | redis.RedisCluster:
        """Get underlying redis instance.

        Returns
        -------
        redis.RedisCluster
            redis cluster.
        """
        return self.redis

    def send_raw(self, data: str) -> None:
        """Send raw data.

        Parameters
        ----------
        data : str
            raw data to send.
        """
        self.redis.xadd(name=self.stream_name, fields={b'data': data.encode()})
