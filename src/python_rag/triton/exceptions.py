"""Contains implementation of triton related exceptions."""

from typing import Tuple

from ..errors import TritonError


class TritonInvalidShapeError(TritonError):
    """Exception raised for invalid input shapes."""

    def __init__(
        self, model_name: str, input_name: str, expected: Tuple[int, ...], received: Tuple[int, ...]
    ):
        message = (
            f"Invalid shape for input `{input_name}`: expected {expected}," + f" but got {received}"
        )
        super().__init__(error_type="invalid-shape", model_name=model_name, message=message)


class TritonEmptyOutputError(TritonError):
    """Exception raised when Triton server returns an empty output."""

    def __init__(self, model_name: str, model_version: str):
        message = f"No output returned by model '{model_name}' version '{model_version}'"
        super().__init__(model_name=model_name, error_type="empty-output", message=message)


class TritonCudaSharedMemoryError(TritonError):
    """Triton cuda shared memory exception."""

    def __init__(self, model_name: str, message: str):
        super().__init__(model_name=model_name, error_type="cushm", message=message)


class TritonConnectionError(TritonError):
    """Exception raised for connection errors with the Triton server."""

    def __init__(self, model_name: str, url: str):
        message = f"Unable to connect to the Triton server at {url}"
        super().__init__(model_name=model_name, error_type="connection", message=message)


class TritonUnknownInputNameError(TritonError):
    """Exception raised for invalid input name for trition model."""

    def __init__(self, model_name: str, argument_name: str):
        message = f"Uknown input name '{argument_name}'"
        super().__init__(model_name=model_name, error_type="unknown-input", message=message)


class TritonUnknownOutputNameError(TritonError):
    """Exception raised for invalid output name for triton model."""

    def __init__(self, model_name: str, argument_name: str):
        message = f"Unknown output name '{argument_name}'"
        super().__init__(model_name=model_name, error_type="unknown-output", message=message)


class TritonInvalidArgumentError(TritonError):
    """Exception raised for invalid arguments sent to the Triton server."""

    def __init__(self, model_name: str, argument_name: str, detail: str):
        message = f"Invalid argument '{argument_name}': {detail}"
        super().__init__(model_name=model_name, error_type="invalid-input", message=message)


class TritonModelNotFoundError(TritonError):
    """Exception raised when the specified model or version is not found on the server."""

    def __init__(self, model_name: str, model_version: str):
        message = f"Model '{model_name}' with version '{model_version}' not found on Triton server"
        super().__init__(model_name=model_name, error_type="model-not-found", message=message)
