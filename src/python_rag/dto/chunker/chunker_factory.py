"""Contains implementation of chunker factory class."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Type, TypeVar

from ...config.chunker import BaseChunkerConfig
from .base_chunker import BaseChunker
from .chunker_interface import IChunker

T = TypeVar("T", bound=BaseChunker)


class ChunkerFactory:
    """Chunker factory class implementation."""

    _chunkers: ClassVar[dict[str, Type[BaseChunker]]] = {}
    _configs: ClassVar[dict[str, Type[BaseChunkerConfig]]] = {}
    _names: ClassVar[dict[Type[BaseChunker], str]] = {}

    @classmethod
    def register_chunker(
        cls, chunker_type: str, config_cls: Type[BaseChunkerConfig]
    ) -> Callable[[Type[T]], Type[T]]:
        """Register chunker class with ginve name and config class."""

        def _decorator(chunker_cls: Type[T]) -> Type[T]:

            if chunker_type in cls._chunkers:
                raise RuntimeError(
                    f"Chunker with type `{chunker_type}` is already reigstered"
                )

            cls._chunkers[chunker_type] = chunker_cls
            cls._configs[chunker_type] = config_cls
            cls._names[chunker_cls] = chunker_type
            return chunker_cls

        return _decorator

    @classmethod
    def get_available_chunker_types(cls) -> list[str]:
        """Get list of available chunker types.

        Returns
        -------
        list[str]
            list of available chunker types.
        """
        return list(cls._chunkers.keys())

    @classmethod
    def get_registered_chunker_classes(cls) -> list[Type[BaseChunker]]:
        """Get all registered chunker classes.

        Returns
        -------
        list[Type[BaseChunker]]
            list of registered chunker classes.
        """
        return list(cls._chunkers.values())

    @classmethod
    def get_config_cls(cls, chunker_type: str) -> Type[BaseChunkerConfig]:
        """Get config class for given chunker type.

        Parameters
        ----------
        chunker_type : str
            Type of chunker.

        Returns
        -------
        Type[BaseChunkerConfig]
            Corresponding config class.
        """
        if chunker_type not in cls._configs:
            raise ValueError(f"Unknown chunker type: `{chunker_type}`")
        return cls._configs[chunker_type]

    @classmethod
    def get_chunker_cls(cls, chunker_type: str) -> Type[BaseChunker]:
        """Get chunker class for given chunker type.

        Parameters
        ----------
        chunker_type : str
            Type of chunker.

        Returns
        -------
        Type[BaseChunker]
            Corresponding chunker class.
        """
        if chunker_type not in cls._chunkers:
            raise ValueError(f"Unknown chunker type: `{chunker_type}`")
        return cls._chunkers[chunker_type]

    @classmethod
    def create_chunker(cls, config_dict: dict[str, Any]) -> IChunker:
        """Create chunker from raw config dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            raw configuration dictionary that will be used
            for chunker creation.

        Returns
        -------
        IChunker
            chunker instance corresponding to given config.
        """
        base_config = BaseChunkerConfig.model_validate(config_dict)
        try:
            config_cls = cls._configs[base_config.chunker_type]
            _ = config_cls.model_validate(config_dict)
        except Exception as e:
            raise ValueError("Invalid chunker config") from e
        if base_config.chunker_type not in cls._chunkers:
            raise ValueError(f"Uknown chunker type: `{base_config.chunker_type}`")
        chunker_cls = cls._chunkers[base_config.chunker_type]
        return chunker_cls.from_config(config_dict=config_dict)

    @classmethod
    def has_chunker_type(cls, chunker_type: str) -> bool:
        """Check whether chunker with given chunker type is registered.

        Parameters
        ----------
        chunker_type : str
            chunker type to check.

        Returns
        -------
        bool
            True if chunker type is registered.
        """
        return chunker_type in cls._chunkers

    @classmethod
    def has_chunker_cls(cls, chunker_cls: Type[BaseChunker]) -> bool:
        """Check whether given chunker class is registered.

        Parameters
        ----------
        chunker_cls : Type[BaseChunker]
            chunker class to check.

        Returns
        -------
        bool
            True if chunker_cls is registered.
        """
        return chunker_cls in cls._names
