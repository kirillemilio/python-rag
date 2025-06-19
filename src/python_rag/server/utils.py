"""Contains utils functions and classes used by RAG service."""

from __future__ import annotations

import logging
from typing import Type, TypeVar

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


class ConfigLoader:
    """Utility class for loading configurations from YAML files."""

    @staticmethod
    def load_config(file_path: str, config_class: Type[T]) -> T:
        """Parse config from yaml file."""
        try:
            return parse_yaml_file_as(config_class, file_path)
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            raise
