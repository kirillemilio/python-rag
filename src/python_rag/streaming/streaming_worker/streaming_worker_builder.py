"""Contains implementation of streaming worker builder."""

from __future__ import annotations

from typing import Literal, Self, cast

import redis

from ...config.llm import LLMConfigTypedDictUnion, LLMConfigUnion
from ...config.redis import (
    RedisClusterConfig,
    RedisConfigTypedDictUnion,
    RedisConfigUnion,
    RedisSingleInstanceConfig,
)
from ...config.streaming_backend import StreamingBackendConfigTypedDict, StreamingBackendConfigUnion
from ...llm import ILLM, LLMFactory
from ..streaming_backend import IStreamingBackend, RedisStreamingBackend, StreamingBackendFactory
from .base_streaming_worker import BaseStreamingWorker
from .redis_streaming_worker import RedisStreamingWorker
from .streaming_worker_interface import IStreamingWorker


class StreamingWorkerBuilder:
    """Build streaming worker from components.

    This class provides a flexible way to build a streaming worker using
    configurations for LLM, Redis, and the streaming backend. It supports
    multiple worker types and allows customization through the builder pattern.

    Attributes
    ----------
    llm : ILLM | None
        Language model instance.
    redis : redis.Redis | redis.RedisCluster | None
        Redis instance for synchronization and stats.
    streaming_backend : IStreamingBackend | None
        Streaming backend to be used (e.g. Redis, Kafka).
    """

    llm: ILLM | None
    redis: redis.Redis | redis.RedisCluster | None
    streaming_backend: IStreamingBackend | None

    def __init__(
        self,
        llm: ILLM | None = None,
        redis_instance: redis.Redis | redis.RedisCluster | None = None,  # type: ignore
        streaming_backend: IStreamingBackend | None = None,
    ) -> None:
        """Initialize streaming worker builder.

        Parameters
        ----------
        llm : ILLM | None, optional
            LLM instance to use, by default None.
        redis_instance : redis.Redis | redis.RedisCluster | None, optional
            Redis instance for tracking stats, by default None.
        streaming_backend : IStreamingBackend | None, optional
            Backend for stream-based communication, by default None.
        """
        self.llm = llm
        self.redis = redis_instance
        self.streaming_backend = streaming_backend

    def with_streaming_backend_config(
        self, config: StreamingBackendConfigUnion | StreamingBackendConfigTypedDict
    ) -> Self:
        """Initialize backend from config.

        Parameters
        ----------
        config : StreamingBackendConfigUnion | StreamingBackendConfigTypedDict
            Configuration dictionary for streaming backend.

        Returns
        -------
        Self
            Updated builder instance with backend initialized.
        """
        if isinstance(config, StreamingBackendConfigUnion):
            config_dict = cast(StreamingBackendConfigTypedDict, config.model_dump())
        else:
            config_dict = config
        self.streaming_backend = StreamingBackendFactory.create_streaming_backend(
            config_dict=config_dict
        )
        return self

    def with_llm_config(self, config: LLMConfigTypedDictUnion | LLMConfigUnion) -> Self:
        """Initialize LLM instance from config.

        Parameters
        ----------
        config : LLMConfigTypedDictUnion | LLMConfigUnion
            Configuration dictionary or model for LLM.

        Returns
        -------
        Self
            Updated builder instance with LLM initialized.
        """
        if isinstance(config, LLMConfigUnion):
            config_dict = cast(LLMConfigTypedDictUnion, config.model_dump())
        else:
            config_dict = config
        self.llm = LLMFactory.create_llm(config_dict=config_dict)
        return self

    def with_redis_config(
        self,
        config: RedisConfigTypedDictUnion | RedisConfigUnion,
    ) -> Self:
        """Initialize Redis instance from config.

        Parameters
        ----------
        config : RedisConfigTypedDictUnion | RedisConfigUnion
            Redis configuration.

        Returns
        -------
        Self
            Updated builder instance with Redis initialized.
        """
        if isinstance(config, RedisConfigUnion):
            config_dict = cast(RedisConfigTypedDictUnion, config.model_dump())
        else:
            config_dict = config
        if config_dict['backend_type'] == 'redis':
            config = RedisSingleInstanceConfig.model_validate(config_dict)
            self.redis = redis.Redis(
                host=config.host,
                port=config.port,
                db=config.db,
                password=config.password,
                decode_responses=False,
            )
        elif config_dict['backend_type'] == 'redis-cluster':
            config_cluster = RedisClusterConfig.model_validate(config_dict)
            self.redis = redis.RedisCluster(
                startup_nodes=config_cluster.startup_nodes,
                password=config_cluster.password,
                require_full_coverage=config_cluster.skip_full_coverage_check,
            )
        else:
            raise ValueError(f'Unable to initialize redis: {config_dict}')
        return self

    def get_streaming_worker(
        self,
        worker_type: Literal['base', 'redis'] = 'redis',
        sampling_temperature: float | None = None,
    ) -> IStreamingWorker:
        """Build a streaming worker instance.

        Parameters
        ----------
        worker_type : {'base', 'redis'}, optional
            Type of worker to build. Use 'base' for stateless worker and
            'redis' for Redis-tracked streaming, by default 'redis'.
        sampling_temperature : float | None, optional
            Temperature to use for sampling from the LLM, by default None.

        Returns
        -------
        IStreamingWorker
            Initialized streaming worker ready to run.

        Raises
        ------
        RuntimeError
            If LLM or streaming backend is not initialized.
        ValueError
            If Redis instance is required but not set.
        """
        if self.llm is None:
            raise RuntimeError(
                'Attribute `llm` must be initialized with `with_llm_config` method call'
            )
        if self.streaming_backend is None:
            raise RuntimeError(
                'Attribute `streaming_backend` must be initialized '
                + 'with `with_streaming_backned_config` method call'
            )
        if worker_type == 'base':
            return BaseStreamingWorker(
                llm=self.llm,
                streaming_backend=self.streaming_backend,
                sampling_temperature=sampling_temperature,
            )
        elif worker_type == 'redis':
            redis_instance = self.redis
            if redis_instance is None and isinstance(self.streaming_backend, RedisStreamingBackend):
                redis_instance = self.streaming_backend.get_redis()
            if redis_instance is None:
                raise ValueError(
                    'Worker type redis requires redis instance to be initialized '
                    + 'with `with_redis_config` method call'
                    + ' or use RedisStreamingWorker class with redis instance.'
                )
            return RedisStreamingWorker(
                llm=self.llm,
                streaming_backend=self.streaming_backend,
                redis=redis_instance,
                sampling_temperature=sampling_temperature,
            )
