"""Contains implementation of base streaming backend class."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from .streaming_backend_interface import IStreamingBackend


class BaseStreamingBackend(IStreamingBackend):
    """Base streaming backend implementation."""

    @classmethod
    @abstractmethod
    def from_config(cls, config_dict: dict[str, Any]) -> BaseStreamingBackend:
        """Create streaming backend from config dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            raw configuration dictionary that will be used
            for streaming backend creation.

        Returns
        -------
        BaseStreamingBackend
            streaming backend instance created from config.
        """
        raise NotImplementedError()
