"""Contains implementation of base triton image model."""

import inspect
import logging
import multiprocessing
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Literal, Optional, Tuple, TypeVar

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore
from numpy.typing import NDArray
from prometheus_client import Counter, Gauge, Histogram

from ..config.triton import TritonConfig
from ..monitoring import MetricsHolder
from .exceptions import (
    TritonConnectionError,
    TritonEmptyOutputError,
    TritonError,
    TritonInvalidArgumentError,
    TritonInvalidShapeError,
    TritonModelNotFoundError,
    TritonUnknownInputNameError,
    TritonUnknownOutputNameError,
)
from .memory_manager import CudaShmMemoryManager, IMemoryManager, SimpleMemoryManager
from .validation_status import EValidationStatus

T = TypeVar("T")
U = TypeVar("U")


LOGGER = logging.getLogger(__name__)


class BaseTritonModel(ABC, Generic[T, U]):
    """Base class for implementing Triton models.

    This class handles setup,
    preprocessing, and postprocessing of Triton inference requests.

    Parameters
    ----------
    client : grpcclient.InferenceServerClient
        Triton Inference Server instance.
    model_name : str
        Name of the model to use.
    inputs : Dict[str, Tuple[int, ...]]
        Specifications of input tensors.
        Keys are names of inputs and values are tuples of ints
        representing tensors shapes excluding batch dimension.
    outputs : List[str]
        List of output tensor names.
    datatype : str
        Data type of the inputs, e.g., 'FP32'.
    client_timeout : int
        Timeout in seconds for the client to wait for a response.
    model_version : str, optional
        Version of the model to use, by default "1".
    device_id : int, optional
        Device ID for CUDA shared memory, by default 0.
    cushm_inputs : Optional[List[str]], optional
        List of input names to use CUDA shared memory, by default None.
    allow_spatial_adjustment : bool
        whether to allow spatial adjustment in way similar
        to batch size adjustment. Default is False.
    **kwargs : dict
        Additional keyword arguments for Triton client.
    """

    model_name: str
    model_version: str
    datatype: str

    memory_manager: IMemoryManager

    inputs: Dict[str, grpcclient.InferInput]
    outputs: Dict[str, grpcclient.InferRequestedOutput]

    device_id: int
    cushm_inputs: List[str]
    client_timeout: Optional[float]
    compression_algorithm: Optional[Literal["deflate", "gzip"]]

    _pr_hist: MetricsHolder[Histogram]
    _pr_counter: MetricsHolder[Counter]
    _pr_gauge: MetricsHolder[Gauge]

    _use_cushm: bool

    _model_cleaned_up: bool

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        inputs: Dict[str, Tuple[int, ...]],
        outputs: List[str],
        datatype: str,
        client_timeout: Optional[float] = None,
        compression_algorithm: Optional[Literal["deflate", "gzip"]] = None,
        model_version: str = "1",
        device_id: int = 0,
        cushm_inputs: Optional[List[str]] = None,
        allow_spatial_adjustment: bool = False,
        **kwargs,
    ):
        self.model_name = str(model_name)
        self.model_version = str(model_version)
        self.datatype = str(datatype)

        self.inputs = self.create_inputs(inputs=inputs, datatype=datatype, batch_size=1)
        self.outputs = self.create_outputs(outputs)
        self.client_timeout = None if client_timeout is None else abs(client_timeout)
        self.compression_algorithm = compression_algorithm
        self.client = client
        self.url = self.client._channel._channel.target().decode()

        self.device_id = device_id
        self.cushm_inputs = [] if cushm_inputs is None else list(cushm_inputs)

        if len(self.cushm_inputs) > 0:
            self.memory_manager = self.create_memory_manager(
                manager_type="cushm", allow_spatial_adjustment=allow_spatial_adjustment
            )
            self._use_cushm = True
        else:
            self.memory_manager = self.create_memory_manager(
                manager_type="simple", allow_spatial_adjustment=allow_spatial_adjustment
            )
            self._use_cushm = False
        self.memory_manager.init(triton_inputs=self.inputs)
        self._model_cleaned_up = False

        # Prometheus metrics
        self._pr_hist = MetricsHolder.get_default_hist()
        self._pr_counter = MetricsHolder.get_default_counter()
        self._pr_gauge = MetricsHolder.get_default_gauge()

    @property
    def use_cushm(self) -> bool:
        """Whether model uses cuda shared memory or not.

        Returns
        -------
        bool
            True if underlying model
            uses cuda shared memory
            False otherwise.
        """
        return self._use_cushm

    @abstractmethod
    def preprocess(self, inputs: List[T]) -> Dict[str, NDArray[np.float32]]:
        """Abstract method to preprocess input data.

        Parameters
        ----------
        inputs : List[T]
            List of inputs to preprocess.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            Dictionary of preprocessed data arrays.
        """
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, raw_outputs: Dict[str, NDArray[np.float32]]) -> List[U]:
        """
        Abstract method to postprocess model outputs.

        Parameters
        ----------
        raw_outputs : Dict[str, NDArray[np.float32]]
            Raw outputs from the Triton model.

        Returns
        -------
        List[U]
            Postprocessed outputs.
        """
        raise NotImplementedError()

    def create_memory_manager(
        self,
        manager_type: Literal["simple", "cushm"],
        allow_spatial_adjustment: bool = False,
    ) -> IMemoryManager:
        """
        Create a memory manager based on the specified type.

        Parameters
        ----------
        manager_type : Literal["simple", "cushm"]
            Specifies the type of memory manager to create. 'simple' uses
            basic numpy arrays, while 'cushm' utilizes CUDA shared memory.
        allow_spatial_adjustment : bool
            whether to allow spatial adjustment or not. Default is False.

        Returns
        -------
        IMemoryManager
            An instance of the memory manager.
        """
        if manager_type == "simple":
            return SimpleMemoryManager(
                client=self.client,
                model_name=self.model_name,
                allow_spatial_adjustment=allow_spatial_adjustment,
            )
        elif manager_type == "cushm":
            return CudaShmMemoryManager(
                client=self.client,
                model_name=self.model_name,
                cushm_inputs=self.cushm_inputs,
                device_id=self.device_id,
                allow_spatial_adjustment=allow_spatial_adjustment,
            )

    @classmethod
    def create_outputs(
        cls, outputs: List[str]
    ) -> Dict[str, grpcclient.InferRequestedOutput]:
        """
        Create output specifications for the Triton model.

        Parameters
        ----------
        outputs : List[str]
            A list of strings naming the output tensors expected from the model.

        Returns
        -------
        Dict[str, grpcclient.InferRequestedOutput]
            A dictionary mapping output names to Triton output request objects.
        """
        return {name: grpcclient.InferRequestedOutput(name=name) for name in outputs}

    @classmethod
    def create_inputs(
        cls, inputs: Dict[str, Tuple[int, ...]], datatype: str, batch_size: int
    ) -> Dict[str, grpcclient.InferInput]:
        """
        Create input specifications for the Triton model.

        Parameters
        ----------
        inputs : Dict[str, Tuple[int, ...]]
            A dictionary specifying each input's name and shape.
        datatype : str
            The data type of the inputs (e.g., 'FP32').
        batch_size : int
            The batch size to set for each input tensor.

        Returns
        -------
        Dict[str, grpcclient.InferInput]
            A dictionary mapping input names to Triton input objects.
        """
        return {
            name: grpcclient.InferInput(
                name=name, shape=(batch_size, *size), datatype=datatype
            )
            for name, size in inputs.items()
        }

    def get_batch_size(self, input_name: str) -> int:
        """
        Retrieve the batch size for a specified input.

        Parameters
        ----------
        input_name : str
            The name of the input tensor.

        Returns
        -------
        int
            The batch size of the specified input.

        Raises
        ------
        TritonInvalidShapeError
            If the input tensor's shape is not properly configured.
        """
        shape = self.inputs[input_name].shape()
        if len(shape) == 0:
            raise TritonInvalidShapeError(
                model_name=self.model_name,
                input_name=input_name,
                expected=(1,),
                received=(),
            )
        return shape[0]

    def set_batch_size(self, input_name: str, batch_size: int):
        """
        Set the batch size for a specified input.

        Parameters
        ----------
        input_name : str
            The name of the input tensor.
        batch_size : int
            The new batch size to be set.

        Raises
        ------
        KeyError
            If the input name is not found within the inputs dictionary.
        TritonInvalidShapeError
            If the input tensor's shape is not properly configured.
        """
        if input_name not in self.inputs:
            raise KeyError(f"No input {input_name} found in inputs dict")
        shape = self.inputs[input_name].shape()
        if len(shape) == 0:
            raise TritonInvalidShapeError(
                model_name=self.model_name,
                input_name=input_name,
                expected=(1,),
                received=(),
            )
        new_shape = (batch_size, *shape[1:])
        self.inputs[input_name].set_shape(new_shape)

    def run_model(
        self, inputs: Dict[str, NDArray[np.float32]], priority: int
    ) -> Dict[str, NDArray[np.float32]]:
        """
        Run inference on the Triton model with the provided inputs.

        Parameters
        ----------
        inputs : Dict[str, NDArray[np.float32]]
            Dictionary of input data arrays.
        priority : int
            Priority level for the inference request.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            Dictionary of output data arrays.

        Raises
        ------
        TritonConnectionError
            If there is a connection issue with the server.
        TritonInvalidArgumentError
            If there are invalid arguments in the request.
        TritonModelNotFoundError
            If the model or version is not found.
        TritonInvalidShapeError
            If there is a shape mismatch in the inputs.
        TritonEmptyOutputError
            If no output is returned by the model.
        """
        start_ts = time.time()
        try:
            self.memory_manager.set_inputs(triton_inputs=self.inputs, data=inputs)
            # NOTE Use prirority parameter with caution here
            # Triton client has a bug regrading parameter
            # it must have int64 type
            # https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/_utils.py#L113
            res = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                client_timeout=self.client_timeout,
                priority=int(priority),
                inputs=list(self.inputs.values()),
                outputs=list(self.outputs.values()),
                compression_algorithm=self.compression_algorithm,
            )

        except grpcclient.InferenceServerException as e:
            fmt_e = str(e).lower()
            if "statuscode.deadline_exceeded" in fmt_e:
                raise TritonConnectionError(url=self.url, model_name=self.model_name)
            elif "statuscode.invalid_argument" in fmt_e:
                parsed_error = self._parse_triton_invalid_arg_error(str(e))
                raise parsed_error
            elif "not found" in fmt_e or "uknown_model" in fmt_e:
                raise TritonModelNotFoundError(self.model_name, self.model_version)
            else:
                raise TritonError(
                    error_type="other", model_name=self.model_name, message=str(e)
                )

        if res is None:
            raise TritonEmptyOutputError(
                model_name=self.model_name, model_version=self.model_version
            )
        outputs = {}
        for name in self.outputs.keys():
            v = res.as_numpy(name)
            if v is None:
                raise TritonEmptyOutputError(
                    model_name=self.model_name, model_version=self.model_version
                )
            outputs[name] = v

        self._update_metrics(delta=time.time() - start_ts)
        return outputs

    def _update_metrics(self, delta: float):
        """Update metrics in prometheus.

        Parameters
        ----------
        delta : float
            single run delta.
        """
        batch_size = self.get_batch_size(list(self.inputs.keys())[0])
        self._pr_hist.with_method(
            subsystem="triton-batch-latency", method=self.model_name
        ).observe(delta)
        self._pr_hist.with_method(
            subsystem="triton-latency", method=self.model_name
        ).observe(delta / batch_size)
        self._pr_gauge.with_method(
            subsystem="triton-batch-size", method=self.model_name
        ).set(batch_size)

    def apply(self, input: T, priority: int = 0) -> U:
        """
        Apply the model to process a single input.

        Parameters
        ----------
        input : T
            A single input instance to process.
        priority : int, optional
            Priority level for the inference request. Default is 0.

        Returns
        -------
        U
            The processed output from the model.
        """
        return self.apply_batch(inputs=[input], priority=priority)[0]

    def apply_batch(self, inputs: List[T], priority: int = 0) -> List[U]:
        """
        Apply the model to process a batch of inputs.

        Parameters
        ----------
        inputs : List[T]
            A list of inputs to process.
        priority : int, optional
            Priority level for the inference requests. Default is 0.

        Returns
        -------
        List[U]
            List of outputs processed by the model.
        """
        raw_inputs = self.preprocess(inputs)
        raw_outputs = self.run_model(inputs=raw_inputs, priority=priority)
        return self.postprocess(raw_outputs)

    def __call__(self, input: T, priority: int = 0) -> U:
        """
        Make the model callable so it can be used directly.

        Parameters
        ----------
        input : T
            A single input instance to process.
        priority : int, optional
            Priority level for the inference request. Default is 0.

        Returns
        -------
        U
            The processed output from the model.
        """
        return self.apply(input=input)

    def __del__(self) -> None:
        """Cleanup resources used by memory managers."""
        self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources used by memory managers."""
        if self._model_cleaned_up:
            return
        elif hasattr(self, "memory_manager") and self.memory_manager is not None:
            frame = inspect.currentframe()
            outer_frame = inspect.getouterframes(frame)[1]
            file_name = outer_frame.filename
            line_number = outer_frame.lineno
            process_name = multiprocessing.current_process().name
            thread_name = threading.current_thread().name
            LOGGER.info(
                f"Cleanup called from file: {file_name}, line: {line_number}, "
                f"process: {process_name}, thread: {thread_name}, model: {self.model_name}"
            )
            self.memory_manager.cleanup()
        self._model_cleaned_up = True

    def validate_model(self) -> Tuple[EValidationStatus, str]:
        """
        Validate the model by running inference on fake data.

        Returns
        -------
        Tuple[EValidationStatus, str]
            The validation status and a descriptive message.
        """
        try:
            fake_inputs = self.generate_fake_data()
            self.run_model(fake_inputs, priority=0)
            return EValidationStatus.SUCCESS, "success"
        except TritonInvalidShapeError as e:
            return EValidationStatus.INVALID_INPUT_SHAPE, str(e)
        except TritonEmptyOutputError as e:
            return EValidationStatus.EMPTY_OUTPUT, str(e)
        except TritonConnectionError as e:
            return EValidationStatus.CONNECTION_ERROR, str(e)
        except TritonUnknownInputNameError as e:
            return EValidationStatus.INVALID_INPUT, str(e)
        except TritonUnknownOutputNameError as e:
            return EValidationStatus.INVALID_OUTPUT, str(e)
        except TritonInvalidArgumentError as e:
            return EValidationStatus.INVALID_ARGUMENT, str(e)
        except TritonModelNotFoundError as e:
            return EValidationStatus.MODEL_NOT_FOUND, str(e)
        except Exception as e:
            return EValidationStatus.UNKNOWN_ERROR, str(e)

    def generate_fake_data(self) -> Dict[str, NDArray[np.float32]]:
        """
        Generate fake data for model validation.

        This method creates synthetic data matching the input specifications
        of the model to validate model setup and execution.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            A dictionary of fake input data arrays, with keys as input names.
        """
        fake_data = {}
        for input_name, infer_input in self.inputs.items():
            data_shape = infer_input.shape()
            fake_data[input_name] = np.random.rand(*data_shape).astype(np.float32)
        return fake_data

    @classmethod
    def create_client(
        cls, triton_config: TritonConfig
    ) -> grpcclient.InferenceServerClient:
        """Create triton client from config.

        Parameters
        ----------
        triton_config : TritonConfig
            triton server config.

        Returns
        -------
        grpcclient.InfernceServerClient
            inference server client instance.
        """
        return grpcclient.InferenceServerClient(
            f"{triton_config.host}:{triton_config.port}"
        )

    def _parse_triton_invalid_arg_error(self, error_message: str) -> TritonError:
        input_pattern = (
            r"\[statuscode\.invalid_argument\] unexpected inference input '(.+)' "
            + r"for model '(.+)'"
        )
        output_pattern = r"\[statuscode\.invalid_argument\] unexpected inference output '(.+)' for model '(.+)'"
        shape_pattern = (
            r"\[statuscode\.invalid_argument\] unexpected shape for input '(.+)' "
            + r"for model '(.+)'. expected \[(.+)\], got \[(.+)\]"
        )

        if match := re.match(input_pattern, error_message, flags=re.IGNORECASE):
            input_name, _ = match.groups()
            return TritonUnknownInputNameError(
                model_name=self.model_name, argument_name=input_name
            )
        if match := re.match(output_pattern, error_message, flags=re.IGNORECASE):
            output_name, _ = match.groups()
            return TritonUnknownOutputNameError(
                model_name=self.model_name, argument_name=output_name
            )
        if match := re.match(shape_pattern, error_message, flags=re.IGNORECASE):
            input_name, _, expected_shape_str, got_shape_str = match.groups()
            expected_shape = tuple(int(s) for s in expected_shape_str.split(","))
            got_shape = tuple(int(s) for s in got_shape_str.split(","))
            return TritonInvalidShapeError(
                input_name=input_name,
                model_name=self.model_name,
                expected=expected_shape,
                received=got_shape,
            )
        return TritonInvalidArgumentError(
            model_name=self.model_name, argument_name="", detail=error_message
        )
