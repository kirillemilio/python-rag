"""Contains implementation of kafka stream producer."""

from __future__ import annotations

from typing import Generic, Type, TypeVar

import kafka
from pydantic import BaseModel

from .base_stream_producer import BaseStreamProducer

T = TypeVar('T', bound=BaseModel)


class KafkaStreamProducer(BaseStreamProducer[T], Generic[T]):
    """Implementation of kafka stream producer.

    Attributes
    ----------
    kafka_servers : list
        list of kafka boostrap servers.
    producer : kafka.KafkaProducer
        kafka producer.
    """

    def __init__(self, item_builder: Type[T], stream_name: str, kafka_servers: list[str]) -> None:
        """Initialize kafka stream producer.

        Parameters
        ----------
        item_builder : Type[T]
            pydantic model item builder class.
        stream_name : str
            stream name.
        """
        self.item_builder = item_builder
        self.stream_name = stream_name
        self.kafka_servers = kafka_servers
        self.producer = kafka.KafkaProducer(
            bootstrap_servers=kafka_servers, value_serializer=lambda v: v.encode('utf-8')
        )

    def get_kafka_producer(self) -> kafka.KafkaProducer:
        """Get underlying kafka producer.

        Returns
        -------
        kafka.KafkaProducer
            underlying kafka producer.
        """
        return self.producer

    def send_raw(self, data: str) -> None:
        """Send raw data.

        Parameters
        ----------
        data : str
            raw data to send.
        """
        self.producer.send(topic=self.stream_name, value=data)
