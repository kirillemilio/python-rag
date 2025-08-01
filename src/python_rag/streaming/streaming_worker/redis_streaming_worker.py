"""Contains implementation of redis streaming worker."""

from __future__ import annotations

import time

import redis

from ...llm import ILLM
from ..streaming_backend import (
    EConsumerConfirmationState,
    IStreamingBackend,
)
from .base_streaming_worker import BaseStreamingWorker


class RedisStreamingWorker(BaseStreamingWorker):
    """Redis streaming worker implementation.

    Redis streaming worker uses redis for streaming state synchronization
    while streaming backend itself can be different (e.g. Kafka)

    Attributes
    ----------
    sampling_temperature : float | None
        sampling temperature for llm.
    redis : redis.Redis | redis.RedisCluster
        redis instance that will be used for streaming
        state updates.
    llm : ILLM
        llm instance that will be used for response generation.
    """

    sampling_temperature: float | None
    redis: redis.Redis | redis.RedisCluster
    llm: ILLM

    streaming_backend: IStreamingBackend

    def __init__(
        self,
        llm: ILLM,
        streaming_backend: IStreamingBackend,
        redis: redis.Redis | redis.RedisCluster,  # type: ignore
        sampling_temperature: float | None = None,
    ) -> None:
        super().__init__(
            sampling_temperature=sampling_temperature, streaming_backend=streaming_backend, llm=llm
        )
        self.redis = redis

    def get_redis(self) -> redis.Redis | redis.RedisCluster:  # type: ignore
        """Get redis.

        Get redis instance that is used for streaming state
        and stats update.

        Returns
        -------
        redis.Redis | redis.RedisCluster
            redis instance.
        """
        return self.redis

    def run(self) -> None:
        """Run llop of streaming worker."""
        stream_consumer = self.get_consumer()
        llm = self.get_llm()
        redis_instance = self.get_redis()
        while True:
            for chat_history in stream_consumer:
                chat_id = chat_history.get_chat_id()
                stream_producer = self.get_producer(chat_history=chat_history)
                ts = time.time()

                redis_instance.zadd('llm:chat:zactivity', {chat_history.get_chat_id(): ts})
                redis_instance.hset(
                    f'llm:chat:{chat_id}:hstats',
                    mapping={
                        'stream_start_ts': ts,
                        'stream_end_ts': ts,
                        'model_name': llm.get_model_name(),
                        'num_chunks': 0,
                        'num_tokens': 0,
                        'is_done': 0,
                    },
                )
                total_tokens, total_chunks = 0, 0
                for response_item in llm.stream_response_on_chat(
                    chat_history=chat_history, temperature=self.get_temperature()
                ):
                    ts = time.time()
                    total_tokens += response_item.get_num_tokens()
                    total_chunks += 1

                    response_item >> stream_producer

                    redis_instance.zadd('llm:chat:zactivity', {chat_history.get_chat_id(): ts})
                    redis_instance.hset(
                        f'llm:chat:{chat_id}:hstats',
                        mapping={
                            'stream_end_ts': ts,
                            'num_tokens': total_tokens,
                            'num_chunks': total_chunks,
                            'is_done': int(response_item.is_done()),
                        },
                    )
            EConsumerConfirmationState.ACK >> stream_consumer
