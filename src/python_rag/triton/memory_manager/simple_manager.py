"""Contains implementation of simple triton model memory manager."""

from typing import Dict, Tuple

import numpy as np
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray

from ..exceptions import TritonInvalidShapeError
from .manager_interface import IMemoryManager


class SimpleMemoryManager(IMemoryManager):
    """Simple memory manager for handling input tensors without CUDA shared memory.

    Directly setting numpy arrays as input data for Triton inference requests.

    Attributes
    ----------
    client : grpcclient.InferenceServerClient
        The Triton Inference Server client instance used for making inference requests.
    model_name : str
        name of the model associated with memory manager.
    allow_spatial_adjustment : bool
        whether to allow spacial adjustment or not.
    """

    client: grpcclient.InferenceServerClient
    model_name: str
    allow_spatial_adjustment: bool

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        allow_spatial_adjustment: bool = False,
    ):
        """
        Initialize the SimpleMemoryManager with a Triton client.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            Client to interact with the Triton Inference Server.
        model_name : str
            name of the model associated with memory manager.
        allow_spatial_adjustment : bool
            whether allow spacial adjustment or not.
            Default is False.
        """
        self.client = client
        self.model_name = model_name
        self.allow_spatial_adjustment = allow_spatial_adjustment

    def init(self, triton_inputs: Dict[str, grpcclient.InferInput]) -> None:
        """Initialize any necessary components for the memory manager.

        This implementation does nothing as no initialization is required
        for simple numpy handling.

        Parameters
        ----------
        triton_inputs : Dict[str, grpcclient.InferInput]
            A dictionary of Triton input objects, not utilized in this simple manager.
        """
        return

    def set_inputs(
        self,
        triton_inputs: Dict[str, grpcclient.InferInput],
        data: Dict[str, NDArray[np.float32]],
    ) -> None:
        """
        Set the inputs for a Triton inference request by directly applying numpy arrays.

        Parameters
        ----------
        triton_inputs : Dict[str, grpcclient.InferInput]
            A dictionary of Triton input configurations.
        data : Dict[str, NDArray[np.float32]]
            A dictionary mapping input names to their respective numpy arrays.

        Raises
        ------
        TritonInvalidShapeError
            If the numpy array shapes do not match the expected Triton input configurations.
        """
        for name, value in data.items():
            input_ = triton_inputs[name]
            input_shape = tuple(input_.shape())

            shape_mismatch = tuple(input_shape[1:]) != tuple(value.shape[1:])
            batch_size_mismatch = int(input_shape[0]) != int(value.shape[0])

            if batch_size_mismatch and not shape_mismatch:
                self.set_batch_size(input_=input_, batch_size=value.shape[0])
            elif shape_mismatch and self.allow_spatial_adjustment:
                self.set_full_shape(input_=input_, shape=value.shape)
            elif shape_mismatch and not self.allow_spatial_adjustment:
                raise TritonInvalidShapeError(
                    input_name=name,
                    model_name=self.model_name,
                    expected=tuple(input_shape[1:]),
                    received=tuple(value.shape[1:]),
                )
            input_.set_data_from_numpy(value)

    def cleanup(self) -> None:
        """Clean up any resources used by the memory manager.

        This implementation does nothing as no resources are allocated.
        """
        return

    def set_full_shape(self, input_: grpcclient.InferInput, shape: Tuple[int, int, int, int]):
        """
        Set full shape for sepcific Triton input configuration.

        Parameters
        ----------
        input_ : grpcclient.InferInput
            The Triton input object whose spatial size is to be set.
        shape : Tuple[int, int, int, int]
            Full shape to set for input_.

        Raises
        ------
        TritonInvalidShapeError
            If the input's initial shape is invalid (e.g., empty).
        """
        old_shape = tuple(input_.shape())
        if len(old_shape) == 0:
            raise TritonInvalidShapeError(
                input_name=input_.name,
                model_name=self.model_name,
                expected=(1,),
                received=old_shape,
            )
        input_.set_shape(shape)

    def set_batch_size(self, input_: grpcclient.InferInput, batch_size: int):
        """
        Set the batch size for a specific Triton input configuration.

        Parameters
        ----------
        input_ : grpcclient.InferInput
            The Triton input object whose batch size is to be set.
        batch_size : int
            The batch size to set for the input.

        Raises
        ------
        TritonInvalidShapeError
            If the input's initial shape is invalid (e.g., empty).
        """
        shape = tuple(input_.shape())
        if len(shape) == 0:
            raise TritonInvalidShapeError(
                input_name=input_.name, model_name=self.model_name, expected=(1,), received=shape
            )
        new_shape = (batch_size, *shape[1:])
        input_.set_shape(new_shape)
