"""Contains implementation of triton models error for multitrack."""

from .base_error import BaseError


class TritonError(BaseError):
    """Base class for all custom triton related exceptions."""

    def __init__(self, error_type: str, model_name: str, message: str):
        super().__init__(subsystem=f"triton-error-{error_type}", method=model_name, message=message)
