"""Contains implementation of triton models pool holder."""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.environment import get_env_settings
from .models_factory import TritonModelFactory
from .models_pool import TritonModelsPool


class TritonModelsPoolHolder:
    """Triton models pool holder.

    Singleton for triton models pool.

    Attributes
    ----------
    models_pool : ClassVar[TritonModelsPool | None]
        models pool if were initialize or None.
    """

    models_pool: ClassVar[TritonModelsPool | None] = None

    @classmethod
    def init(cls, models_pool_config: list[dict[str, Any]]) -> None:
        """Initialize triton models pool using config.

        Parameters
        ----------
        models_pool_config : list[dict[str, Any]]
            models pool config. Each instance must be parsable
            into triton model config.
        """
        env_settings = get_env_settings()
        model_factory = TritonModelFactory.from_config(env_settings.triton)
        cls.models_pool = TritonModelsPool.from_config(
            config=models_pool_config, model_factory=model_factory
        )

    @classmethod
    def get_models_pool(cls) -> TritonModelsPool:
        """Get triton models pool.

        Returns
        -------
        TritonModelsPool
            initialized triton models pool instance.

        Raises
        ------
        RuntimeError
            if triton models pool is not initialized.
        """
        if cls.models_pool is None:
            raise RuntimeError(
                "Triton models pool is not initialized. "
                + "Consider calling `init` method first."
            )
        return cls.models_pool
