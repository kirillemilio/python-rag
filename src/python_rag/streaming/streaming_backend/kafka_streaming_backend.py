"""Contains implementation of kafka streaming backend."""

from __future__ import annotations

import uuid
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from ...config.streaming_backend import KafkaStreamingBackendConfig
from .base_streaming_backend import BaseStreamingBackend
from .consumer import KafkaStreamConsumer
from .producer import KafkaStreamProducer
from .streaming_backend_factory import StreamingBackendFactory

T = TypeVar('T', bound=BaseModel)


@StreamingBackendFactory.register_streaming_backend('kafka', config_cls=KafkaStreamingBackendConfig)
class KafkaStreamingBackend(BaseStreamingBackend):
    """Kafka-based implementation of streaming backend.

    This backend provides stream producer and stream consumer interfaces
    over kafka streams.

    Attributes
    ----------
    kafka_servers : list[str]
        list of kafka servers.
    """

    kafka_servers: list[str]

    def __init__(self, kafka_servers: list[str]) -> None:
        super().__init__()
        self.kafka_servers = kafka_servers

    def get_consumer(
        self, item_builder: Type[T], stream_name: str, consumer_group: str
    ) -> KafkaStreamConsumer[T]:
        """Create a new kafka stream consumer instance.

        Parameters
        ----------
        item_builder : Type[T]
            A Pydantic model used to parse consumed data.
        stream_name : str
            Name of kafka stream.
        consumer_group : str
            name of kafka consumer group.

        Returns
        -------
        KafkaStreamConsumer[T]
            A configured Kafka stream consumer.
        """
        rnd_name = uuid.uuid4().hex
        return KafkaStreamConsumer(
            item_builder=item_builder,
            stream_name=stream_name,
            consumer_group=consumer_group,
            consumer_name=f'{consumer_group}-{rnd_name[:16]}',
            kafka_servers=self.kafka_servers,
            auto_offset_reset='earliest',
            enable_autocommit=False,
        )

    def get_producer(self, item_builder: Type[T], stream_name: str) -> KafkaStreamProducer[T]:
        """Get a new kafka stream producer instance.

        Parameters
        ----------
        item_builder : Type[T]
            A Pydantic model used to parse produced data.
        stream_name : str
            Name of kafka stream.

        Returns
        -------
        KafkaStreamProducer[T]
            A configured Kafka stream producer.
        """
        return KafkaStreamProducer(
            item_builder=item_builder, stream_name=stream_name, kafka_servers=self.kafka_servers
        )

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> KafkaStreamingBackend:
        """Create kafka streaming backend from raw configuration dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            configuration dictionary to use to create kafka streaming backend.

        Returns
        -------
        KafkaStreamingBackend
            kafka streaming backend instance created from config.
        """
        config = KafkaStreamingBackendConfig.model_validate(config_dict)
        return KafkaStreamingBackend(config.kafka_servers)
