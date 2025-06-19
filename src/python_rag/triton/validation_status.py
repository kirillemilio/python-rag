"""Contains implementation of validation status enum for triton models."""

from enum import Enum, auto


class EValidationStatus(Enum):
    """Enum with validation statuses of triton models."""

    SUCCESS = auto()
    INVALID_ARGUMENT = auto()
    INVALID_INPUT = auto()
    INVALID_OUTPUT = auto()
    INVALID_INPUT_SHAPE = auto()
    EMPTY_OUTPUT = auto()
    CONNECTION_ERROR = auto()
    MODEL_NOT_FOUND = auto()
    UNKNOWN_ERROR = auto()
