"""Contains implementation of redis config classes."""

from __future__ import annotations

from typing import Literal, Required, TypedDict

from pydantic import BaseModel, Field


class BaseRedisConfig(BaseModel):
    """Implements base redis config model."""

    backend_type: str


class RedisSingleInstanceConfig(BaseRedisConfig):
    """Implements single-instance redis config."""

    backend_type: Literal['redis']
    host: str = Field(..., description='Redis host address')
    port: int = Field(6379, description='Redis port number')
    db: int = Field(0, description='Redis database index')
    password: str | None = Field(None, description='Redis password if required')
    decode_responses: bool = Field(True, description='Whether to decode responses as strings')


class RedisClusterConfig(BaseModel):
    """Implements Redis cluster config."""

    backend_type: Literal['redis-cluster']
    startup_nodes: list[str] = Field(
        ..., description="List of Redis cluster node addresses, e.g. ['host1:port1', 'host2:port2']"
    )
    decode_responses: bool = Field(True, description='Whether to decode responses as strings')
    password: str | None = Field(
        None, description='Password if Redis cluster is password-protected'
    )
    skip_full_coverage_check: bool = Field(
        True, description='Allows to skip full coverage check (useful in dev environments)'
    )


class RedisSingleInstanceConfigTypedDict(TypedDict):
    """TypedDict for single-instance redis config."""

    backend_type: Required[Literal['redis']]
    host: Required[str]
    port: Required[int]
    db: Required[int]
    password: str | None


class RedisClusterConfigTypedDict(TypedDict):
    """TypedDict for Redis cluster config."""

    backend_type: Required[Literal['redis-cluster']]
    startup_nodes: Required[list[str]]
    password: str | None
    skip_full_coverage_check: bool


RedisConfigTypedDictUnion = RedisSingleInstanceConfigTypedDict | RedisClusterConfigTypedDict
RedisConfigUnion = RedisSingleInstanceConfig | RedisClusterConfig
