"""Contains implementation of llm config classes."""

from __future__ import annotations

from typing import Literal, Required, TypedDict

from pydantic import BaseModel


class BaseLLMConfig(BaseModel):
    """Implements base llm config model."""

    model_type: str
    model_name: str


class HuggingFaceLLMConfig(BaseLLMConfig):
    """Implements hugging face llm config model."""

    model_type: Literal['huggingface']
    model_name: str
    hub_name: str
    use_cuda: bool = True


class HuggingFaceLLMConfigTypedDict(TypedDict):
    """Implements hugging face llm config typed dictionary."""

    model_type: Required[Literal['huggingface']]
    model_name: Required[str]
    hub_name: Required[str]
    use_cuda: Required[bool]


LLMConfigTypedDictUnion = HuggingFaceLLMConfigTypedDict
LLMConfigUnion = HuggingFaceLLMConfig
