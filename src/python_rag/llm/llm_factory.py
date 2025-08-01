"""Contains implementation of llm model factory class."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Type, TypeVar, cast

from ..config.llm import BaseLLMConfig, LLMConfigTypedDictUnion
from .base_llm import BaseLLM

T = TypeVar('T', bound=BaseLLM)


class LLMFactory:
    """Implementation of llm model factory-registry class."""

    _models: ClassVar[dict[str, Type[BaseLLM]]] = {}
    _configs: ClassVar[dict[str, Type[BaseLLMConfig]]] = {}
    _names: ClassVar[dict[Type[BaseLLM], str]] = {}

    @classmethod
    def register_llm_model(
        cls, model_type: str, config_cls: Type[BaseLLMConfig]
    ) -> Callable[[Type[T]], Type[T]]:
        """Register llm class in factory.

        Parameters
        ----------
        model_type : str
            llm model type to register.
        config_cls : Type[BaseLLMConfig]
            config class associated with streaming backend.

        Returns
        -------
        Callable[[Type[T]], Type[T]]
            decorator for wrapping streaming backend.
        """

        def _decorator(llm_cls: Type[T]) -> Type[T]:
            if model_type in cls._models:
                raise ValueError(f'LLM model with name `{model_type}` is already registered')
            cls._models[model_type] = llm_cls
            cls._configs[model_type] = config_cls
            cls._names[llm_cls] = model_type
            return llm_cls

        return _decorator

    @classmethod
    def create_llm(cls, config_dict: dict[str, Any] | LLMConfigTypedDictUnion) -> BaseLLM:
        """Create llm from config dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any] | LLMConfigTypedDictUnion
            configuration dictionary for llm.

        Returns
        -------
        BaseLLM
            llm instance built from configuration dictionary.
        """
        base_config = BaseLLMConfig.model_validate(config_dict)
        llm_cls = cls._models[base_config.model_type]
        return llm_cls.from_config(cast(dict[str, Any], config_dict))

    @classmethod
    def get_model_type(cls, llm_cls: Type[BaseLLM]) -> str:
        """Get model type corresponding to given llm class.

        Parameters
        ----------
        llm_cls : Type[BaseLLM]
            llm class for which model type
            will be fetched.

        Returns
        -------
        str
            model type corresponding to given llm class.
        """
        return cls._names[llm_cls]

    @classmethod
    def get_llm_cls(cls, model_type: str) -> Type[BaseLLM]:
        """Get llm class corresponding to given model type.

        Parameters
        ----------
        model_type : str
            model type for which registered llm class
            will be fetched.

        Returns
        -------
        Type[BaseLLM]
            llm class correspondign to model type.
        """
        return cls._models[model_type]

    @classmethod
    def get_config_cls(cls, model_type: str) -> Type[BaseLLMConfig]:
        """Get llm config class corresponding to given model type.

        Parameters
        ----------
        model_type : str
            model type for which config class will be fetched.

        Returns
        -------
        Type[BaseLLMConfig]
            llm config corresponding to given model type.
        """
        return cls._configs[model_type]

    @classmethod
    def has_model_type(cls, model_type: str) -> bool:
        """Check whether model type is registered in factory.

        Parameters
        ----------
        model_type : str
            model type to check.

        Returns
        -------
        bool
            True if model type is registered.
        """
        return model_type in cls._models
