"""Contains implementation of streaming backend configs."""

from __future__ import annotations

from typing import Literal, NotRequired, Required, TypedDict

from pydantic import BaseModel


class BaseStreamingBackendConfig(BaseModel):
    """Base streaming backend config model.

    Attributes
    ----------
    backend_type : str
        backend type used for streaming.
    """

    backend_type: str


class RedisStreamingBackendConfig(BaseStreamingBackendConfig):
    """Redis streaming backend config.

    Attributes
    ----------
    backend_type : Literal["redis"]
        backend type. Must be "redis" for redis backend.
    host : str
        redis host.
        Default is localhost.
    port : int
        redis port.
        Default is 6379.
    db : int
        redis database index.
        Default is 0.
    password : str | None
        redis password.
        Default is None.
    buffer_max_size : int
        stream buffer max size.
        Default is 10.
    stream_ack_timeout : int
        stream ack timeout in ms.
        Default is 30000.
    """

    backend_type: Literal['redis']
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: str | None = None
    buffer_max_size: int = 10
    stream_ack_timeout: int = 30000


class KafkaStreamingBackendConfig(BaseStreamingBackendConfig):
    """Kafka streaming backend config.

    Attributes
    ----------
    backend_type : Literal["kafka"]
        backend type. Must be "kafka" for kafka backend.
    kafka_servers : list[str]
        list of kafka servers address.
    """

    backend_type: Literal['kafka']
    kafka_servers: list[str]


class RedisStreamingBackendConfigTypedDict(TypedDict):
    """Redis streaming backend config typed dictionary.

    Attributes
    ----------
    backend_type : Required[Literal["redis"]]
        backend type must be "redis".
    host : NotRequired[str]
        redis host. Default is "localhost".
    port : NotRequired[int]
        redis port.
        Default is 6379.
    db : NotRequired[int]
        redis database index.
        Default is 0.
    password : NotRequired[str | None]
        redis password.
        Default is None.
    buffer_max_size : NotRequired[int]
        stream buffer max size.
        Default is 10.
    stream_ack_timeout : NotRequired[int]
        stream ack timeout in ms.
        Default is 30000.
    """

    backend_type: Required[Literal['redis']]
    host: NotRequired[str]
    port: NotRequired[int]
    db: NotRequired[int]
    password: NotRequired[str | None]
    buffer_max_size: NotRequired[int]
    stream_ack_timeout: NotRequired[int]


class KafkaStreamingBackendConfigTypedDict(TypedDict):
    """Kafka streaming backend config typed dictionary.

    Attributes
    ----------
    backend_type : Required[Literal["kafka"]]
        backend type must be "kafka".
    kafka_servers : Required[list[str]]
        kafka server list.
    """

    backend_type: Required[Literal['kafka']]
    kafka_servers: Required[list[str]]


StreamingBackendConfigUnion = RedisStreamingBackendConfig | KafkaStreamingBackendConfig


StreamingBackendConfigTypedDict = (
    RedisStreamingBackendConfigTypedDict | KafkaStreamingBackendConfigTypedDict
)
