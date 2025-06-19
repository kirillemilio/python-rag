"""Contains implementation of pydantic environment settings class."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# from .postgres import PostgresConfig
from .qdrant import QdrantConfig
from .triton import TritonConfig


class EnvSettings(BaseSettings):
    """Service environment settings."""

    service_name: str = Field(default="python-rag")
    service_env: Literal["dev", "prod"] = "dev"
    logging_dir: str = Field(default="./logs")
    triton: TritonConfig = TritonConfig(host="localhost", port=8081)
    qdrant: QdrantConfig = QdrantConfig()
    #    postgres: PostgresConfig = PostgresConfig()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__", env_file=".env", extra="allow"
    )


@lru_cache
def get_env_settings() -> EnvSettings:
    """Get environment settings.

    Returns
    -------
    EnvSettings
        environment settings instance.
    """
    return EnvSettings()
