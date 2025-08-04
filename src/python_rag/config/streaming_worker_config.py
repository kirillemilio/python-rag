"""Contains implementation of streaming worker config."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .llm import LLMConfigUnion
from .redis import RedisConfigUnion
from .streaming_backend import StreamingBackendConfigUnion


class StreamingWorkerConfig(BaseModel):
    """Streaming worker config.

    Attributes
    ----------
    llm : LLMConfigUnion
        llm config.
    redis : RedisConfigUnion
        redis config union.
    streaming_backend : StreamingBackendConfigUnion
        streaming backend config.
    worker_type : Literal["base", "redis"]
        worker type. Can be one of 'base', 'redis'.
    sampling_temperature : float | None
        sampling temperature for llm.
        Default is None meaning that argmax mode will be used.
    """

    llm: LLMConfigUnion
    redis: RedisConfigUnion
    streaming_backend: StreamingBackendConfigUnion
    worker_type: Literal['base', 'redis'] = 'redis'
    sampling_temperature: float | None = None
