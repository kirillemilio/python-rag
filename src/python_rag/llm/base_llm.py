"""Contains implementation of base llm class."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from .llm_interface import ILLM


class BaseLLM(ILLM):
    """Base llm class implementation."""

    @classmethod
    @abstractmethod
    def from_config(cls, config_dict: dict[str, Any]) -> BaseLLM:
        """Create llm from raw configuration dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            configuration dictionary that will be used
            from llm building.
            Must be parsable into BaseLLMConfig instance.

        Returns
        -------
        BaseLLM
            constructed BaseLLM instance.
        """
        raise NotImplementedError()
