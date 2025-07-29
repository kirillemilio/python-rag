"""Contains implementation of streaming backend factory."""

from __future__ import annotations

from typing import Callable, ClassVar, Type, TypeVar

from ...config.streaming_backend import BaseStreamingBackendConfig, StreamingBackendConfigTypedDict
from .base_streaming_backend import BaseStreamingBackend

T = TypeVar('T', bound=BaseStreamingBackend)


class StreamingBackendFactory:
    """Implementation of streaming backend factory class."""

    _configs: ClassVar[dict[str, Type[BaseStreamingBackendConfig]]] = {}
    _backends: ClassVar[dict[str, Type[BaseStreamingBackend]]]
    _names: ClassVar[dict[Type[BaseStreamingBackend], str]] = {}

    @classmethod
    def register_streaming_backend(
        cls, backend_type: str, config_cls: Type[BaseStreamingBackendConfig]
    ) -> Callable[[Type[T]], Type[T]]:
        """Register streaming backend in factory.

        Parameters
        ----------
        backend_type : str
            backend type to register.
        config_cls : Type[BaseStreamingBackendConfig]
            config class associated with streaming backend.

        Returns
        -------
        Callable[[Type[T]], Type[T]]
            decorator for warpping streaming backend.
        """

        def _decorator(backend_cls: Type[T]) -> Type[T]:
            if backend_type in cls._backends:
                raise ValueError(
                    f'Streaming backend with type: `{backend_type}` is already registered'
                )
            cls._names[backend_cls] = backend_type
            cls._backends[backend_type] = backend_cls
            cls._configs[backend_type] = config_cls
            return backend_cls

        return _decorator

    @classmethod
    def create_streaming_backend(
        cls, config_dict: StreamingBackendConfigTypedDict
    ) -> BaseStreamingBackend:
        """Create streaming backend from config dictionary.

        Parameters
        ----------
        config_dict : StreamingBackendConfigTypedDict
            configuration dictionary for streaming backend.

        Returns
        -------
        BaseStreamingBackend
            streaming backend built from configuration dictionary.
        """
        base_config = BaseStreamingBackendConfig.model_validate(config_dict)
        backend_cls = cls._backends[base_config.backend_type]
        return backend_cls.from_config(config_dict)

    @classmethod
    def get_backend_type(cls, backend_cls: Type[T]) -> str:
        """Get backend type name by backend class.

        Parameters
        ----------
        backend_cls : Type[T]
            backend class for which type name will be fetched.

        Returns
        -------
        str
            backend type.
        """
        return cls._names[backend_cls]

    @classmethod
    def get_backend_cls(cls, backend_type: str) -> Type[BaseStreamingBackend]:
        """Get backend class by backend type.

        Parameters
        ----------
        backend_type : str
            backend type for which backend class
            will be fetched.

        Returns
        -------
        Type[T]
            backend type class corresponding to
            given backend type alias.
        """
        return cls._backends[backend_type]

    @classmethod
    def get_config_cls(cls, backend_type: str) -> Type[BaseStreamingBackendConfig]:
        """Get backend config class for given backend type.

        Parameters
        ----------
        backend_type : str
            backend type for which config class will be fetched.

        Returns
        -------
        Type[BaseStreamingBackendConfig]
            streaming backend config class corresponding to given backend
            type.
        """
        return cls._configs[backend_type]

    @classmethod
    def has_backend_type(cls, backend_type: str) -> bool:
        """Check whether backend type is registered in factory.

        Parameters
        ----------
        backend_type : str
            backend type to check.

        Returns
        -------
        bool
            True if backend type is registered.
        """
        return backend_type in cls._backends
