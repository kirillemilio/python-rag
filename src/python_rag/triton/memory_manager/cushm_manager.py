"""Contains implementation of cudasharememory manager for triton models."""

import logging
import threading
import uuid
from typing import ClassVar, Dict, List, Tuple

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.utils.cuda_shared_memory as cudashm
from numpy.typing import NDArray
from tritonclient.utils.cuda_shared_memory import CudaSharedMemoryRegion

from ..exceptions import TritonCudaSharedMemoryError, TritonInvalidShapeError
from .simple_manager import SimpleMemoryManager


class CudaShmMemoryManager(SimpleMemoryManager):
    """A memory manager that uses CUDA shared memory.

    Optimizes the data transfer
    between the client and a Triton Inference Server, extending the SimpleMemoryManager.

    Attributes
    ----------
    client : grpcclient.InferenceServerClient
        The client used to interact with the Triton Inference Server.
    model_name : str
        name of the model associated with memory manager.
    cushm_inputs : List[str]
        List of input names that should use CUDA shared memory.
    device_id : int
        The GPU device ID that will be used for CUDA shared memory.
    """

    cushm_inputs: List[str]
    cushm_regions: Dict[str, str]
    cushm_handles: Dict[str, CudaSharedMemoryRegion]
    cushm_shapes: Dict[str, Tuple[int, ...]]

    device_id: int

    lock: ClassVar[threading.RLock] = threading.RLock()
    _unregister_memory_init_called: ClassVar[bool] = False

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        cushm_inputs: List[str],
        device_id: int = 0,
        allow_spatial_adjustment: bool = False,
    ):
        """Initialize the CUDA shared memory manager with Triton client, inputs, and device ID.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            The Triton server client.
        cushm_inputs : List[str]
            Names of the inputs that will use CUDA shared memory.
        device_id : int, optional
            The GPU device ID for CUDA operations, by default 0.
        """
        super().__init__(
            client=client, model_name=model_name, allow_spatial_adjustment=allow_spatial_adjustment
        )
        self.device_id = device_id
        self.cushm_inputs = cushm_inputs
        self.cushm_regions = {}
        self.cushm_handles = {}
        self.cushm_shapes = {}

    def init(self, triton_inputs: Dict[str, grpcclient.InferInput]) -> None:
        """Initialize the necessary CUDA shared memory regions for specified inputs.

        Parameters
        ----------
        triton_inputs : Dict[str, grpcclient.InferInput]
            Dictionary mapping input names to their Triton server configurations.
        """
        self._initial_unregister_cushm()
        self._init_regions(self.cushm_inputs, triton_inputs)
        for name in self.cushm_inputs:
            region = self.cushm_regions[name]
            handle = self.cushm_handles[name]
            input_ = triton_inputs[name]
            data_size = np.float32().itemsize * np.prod(input_.shape())
            self.client.register_cuda_shared_memory(
                name=region,
                raw_handle=cudashm.get_raw_handle(handle),
                byte_size=data_size,
                device_id=self.device_id,
            )

    def cleanup(self):
        """Clean up all CUDA shared memory regions by unregistering them from the server."""
        self._cleanup()

    def set_inputs(
        self,
        triton_inputs: Dict[str, grpcclient.InferInput],
        data: Dict[str, NDArray[np.float32]],
    ) -> None:
        """Set inputs for the Triton inference request managing both regular and CUDA shared memory.

        Parameters
        ----------
        triton_inputs : Dict[str, grpcclient.InferInput]
            A dictionary of Triton input configurations.
        data : Dict[str, NDArray[np.float32]]
            A dictionary mapping input names to their respective numpy arrays.
        """
        default_data = {}
        cushm_data = {}
        for name, value in data.items():
            if name in self.cushm_inputs:
                cushm_data[name] = value
            else:
                default_data[name] = value
        super().set_inputs(triton_inputs=triton_inputs, data=default_data)
        self.set_cushm_inputs(triton_inputs=triton_inputs, data=cushm_data)

    def set_cushm_inputs(
        self, triton_inputs: Dict[str, grpcclient.InferInput], data: Dict[str, NDArray[np.float32]]
    ):
        """
        Specifically sets CUDA shared memory inputs for the Triton inference request.

        Parameters
        ----------
        triton_inputs : Dict[str, grpcclient.InferInput]
            A dictionary of Triton input configurations.
        data : Dict[str, NDArray[np.float32]]
            A dictionary mapping input names to their respective data arrays which
            are to be set in CUDA shared memory.
        """
        for name, value in data.items():
            input_ = triton_inputs[name]
            input_shape = tuple(input_.shape())

            data_size = self._get_data_size(value)

            old_cushm_shape = self.cushm_shapes.get(name, (0, 0, 0, 0))
            old_cushm_batch_size, *_ = old_cushm_shape

            shape_mismatch = tuple(old_cushm_shape[1:]) != tuple(value.shape[1:])
            batch_size_mismatch = int(old_cushm_batch_size) != int(value.shape[0])

            if batch_size_mismatch and not shape_mismatch and old_cushm_batch_size < value.shape[0]:
                logging.warning(
                    "Resetting batch size of cudashm input can lead to performance degradation"
                )
                new_batch_size = value.shape[0]
                self.set_batch_size(input_=input_, batch_size=new_batch_size)
                # Unregister and delete old regions
                self._cleanup_cushm_input(input_name=name)
                # Register new regions
                self._init_regions(names=[name], inputs={name: input_})

                region = self.cushm_regions[name]
                handle = self.cushm_handles[name]
                self.client.register_cuda_shared_memory(
                    name=region,
                    raw_handle=cudashm.get_raw_handle(handle),
                    byte_size=data_size,
                    device_id=self.device_id,
                )
            elif (
                shape_mismatch
                and self.allow_spatial_adjustment
                and np.prod(old_cushm_shape) < np.prod(input_shape)
            ):
                logging.warning(
                    "Resetting spatial size of cudashm input"
                    + " can lead to performance degradation"
                )
                self.set_full_shape(input_=input_, shape=value.shape)
                # Unregister and delete old regions
                self._cleanup_cushm_input(name)
                # Register new regions
                self._init_regions(names=[name], inputs={name: input_})

                region = self.cushm_regions[name]
                handle = self.cushm_handles[name]
                self.client.register_cuda_shared_memory(
                    name=region,
                    raw_handle=cudashm.get_raw_handle(handle),
                    byte_size=data_size,
                    device_id=self.device_id,
                )

            elif shape_mismatch and not self.allow_spatial_adjustment:
                raise TritonInvalidShapeError(
                    input_name=name,
                    model_name=self.model_name,
                    expected=tuple(input_shape[1:]),
                    received=tuple(value.shape[1:]),
                )
            elif shape_mismatch or batch_size_mismatch:
                self.set_full_shape(input_=input_, shape=value.shape)
            region = self.cushm_regions[name]
            handle = self.cushm_handles[name]
            cudashm.set_shared_memory_region(handle, [value])
            input_.set_shared_memory(region, data_size)

    @classmethod
    def _set_unregister_memory_flag(cls):
        """
        Specifically sets CUDA shared memory inputs for the Triton inference request.

        Parameters
        ----------
        triton_inputs : Dict[str, grpcclient.InferInput]
            A dictionary of Triton input configurations.
        data : Dict[str, NDArray[np.float32]]
            A dictionary mapping input names to their respective data arrays which
            are to be set in CUDA shared memory.
        """
        cls._unregister_memory_init_called = True

    @classmethod
    def _get_data_size(cls, data: np.ndarray) -> int:
        """Calculate the total byte size of the data array based on its dtype and shape.

        Parameters
        ----------
        data : np.ndarray
            The numpy array for which to calculate the byte size.

        Returns
        -------
        int
            The total byte size of the data array.
        """
        return data.size * data.itemsize

    @classmethod
    def _gen_region_name(cls, prefix: str = "") -> str:
        """
        Generate a unique region name for CUDA shared memory.

        This method generates a unique identifier to be used as part of the name
        for a CUDA shared memory region, optionally prefixed by a specified string.

        Parameters
        ----------
        prefix : str, optional
            A prefix to prepend to the generated unique identifier.

        Returns
        -------
        str
            A unique CUDA shared memory region name with the optional prefix.
        """
        return prefix + uuid.uuid4().hex

    def _initial_unregister_cushm(self):
        """
        Unregister all CUDA and system shared memory regions at the start.

        This method ensures that no shared memory regions are left registered
        from previous sessions, thereby cleaning up before initializing new regions.
        It prevents double registration and potential memory leaks.
        """
        if self._unregister_memory_init_called:
            return
        with self.lock:
            self.client.unregister_cuda_shared_memory()
            self.client.unregister_system_shared_memory()
            self._set_unregister_memory_flag()

    def _init_regions(self, names: List[str], inputs: Dict[str, grpcclient.InferInput]):
        """
        Initialize CUDA shared memory regions for specified inputs.

        This method sets up CUDA shared memory regions for the inputs specified
        that are intended to use CUDA shared memory. It raises an error if any
        specified input names do not exist.

        Parameters
        ----------
        names : List[str]
            Names of the inputs for which CUDA shared memory regions are to be created.
        inputs : Dict[str, grpcclient.InferInput]
            A dictionary of input configurations from Triton.

        Raises
        ------
        ValueError
            If any of the specified names are not present in the inputs dictionary.
        TritonCudaSharedMemoryError
            If there is a failure in creating any of the CUDA shared memory regions.
        """
        if not set(inputs.keys()).issuperset(set(names)):
            raise ValueError("cushm inputs must be a subset of input names")
        with self.lock:
            for name in names:
                region_name = self._gen_region_name(prefix=name + "_")
                self.cushm_regions[name] = region_name
                try:
                    shape = inputs[name].shape()
                    byte_size = np.float32().itemsize * np.prod(shape)
                    self.cushm_handles[name] = cudashm.create_shared_memory_region(
                        triton_shm_name=region_name,
                        byte_size=byte_size,
                        device_id=self.device_id,
                    )
                    self.cushm_shapes[name] = shape
                except Exception as e:
                    logging.error(f"Failed to create CUDA shared memory region: {str(e)}")
                    del self.cushm_regions[name]
                    raise TritonCudaSharedMemoryError(
                        model_name=self.model_name,
                        message=f"Failed to create CUDA shared memory region for {name}",
                    )

    def _cleanup_cushm_input(self, input_name: str):
        """Clean up on cushm input.

        This method unregisters corresponding region_name(self.region_names[input_name])
        from triton client and destroys shared memory regions using cudashm.

        Parameters
        ----------
        input_name : str
            cudashared memory input.
        """
        with self.lock:
            region_name = self.cushm_regions[input_name]
            region_handle = self.cushm_handles[input_name]
            try:
                self.client.unregister_cuda_shared_memory(region_name)
                cudashm.destroy_shared_memory_region(region_handle)
            except Exception as e:
                logging.error(
                    "Failed to cleanup CUDA shared " + f"memory region '{region_name}': {str(e)}"
                )
            finally:
                del self.cushm_regions[input_name]
                del self.cushm_handles[input_name]

    def _cleanup(self):
        """
        Clean up all CUDA shared memory regions.

        This method unregisters all CUDA shared memory regions managed by this manager,
        ensuring that there are no resources left allocated after the operation is complete.

        Exceptions are logged rather than raised to avoid disruption in cleanup operations.
        """
        with self.lock:
            for input_name in self.cushm_inputs:
                self._cleanup_cushm_input(input_name)
