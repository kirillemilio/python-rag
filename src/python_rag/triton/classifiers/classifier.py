"""Contains implementation of base classifier triton model."""

from typing import Callable, Dict, List, Optional, Set

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore
from numpy.typing import NDArray

from ...dto import Size
from ..models_factory import TritonModelFactory
from ..transforms import get_transform_sequence
from ..triton_model import BaseTritonModel


@TritonModelFactory.register_model(model_type="classifier", arch_type="classifier")
class ClassifierTritonModel(BaseTritonModel[NDArray[np.float32], Dict[str, NDArray[np.float32]]]):
    """
    A specialized Triton model class for image classification tasks.

    This class handles the preprocessing of images,
    setting up necessary transformations,
    and manages interactions with a Triton inference server.

    Attributes
    ----------
    image_input_name : str
        Name of the input tensor expected by the Triton model.
    transform : Callable
        Transformation function applied to input images.
    _batch_size : int
        Batch size used for processing; adjusted dynamically.
    input_size : Size
        Size object specifying the dimensions to which input images are resized.
    """

    image_input_name: str
    transform: Callable[[NDArray[np.float32]], NDArray[np.float32]]
    _batch_size: int
    input_size: Size

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        input_name: str,
        input_size: Size,
        outputs: List[str],
        client_timeout: float | None,
        outputs_map: Optional[Dict[str, str]] = None,
        model_version: str = "1",
        device_id: int = 0,
        use_cushm: bool = False,
        **kwargs,
    ):
        """Initialize the ClassifierTritonModel.

        Initialization with specific settings for image processing
        and Triton server communication.

        Default preprocessing config is (0.0, 0.0, 0.0) as mean,
        (1.0, 1.0, 1.0) as std and crop_mode equal to 'center'.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            Triton Inference Server instance.
        model_name : str
            The name of the model to use on the Triton server.
        input_name : str
            The name of the input tensor as expected by the model.
        input_size : Size
            The dimensions (width and height) to which input images are resized.
        outputs : List[str]
            Names of the output tensors as expected from the model.
        client_timeout : float | None
            Timeout in seconds for the Triton client when making requests.
        outputs_map : Optional[Dict[str, str]]
            optional mapping from triton's output head name
            to new key name. Can be useful for remapping
            certain outputs names to new names.
            Defualt is None meaning that not remapping
            will be appied. If key not present
            as key in outputs_map dict that it will
            be returned by original key.
        model_version : str, optional
            The version of the model to use.
        device_id : int, optional
            The GPU device ID to use for CUDA operations.
        use_cushm : bool, optional
            Whether to use CUDA shared memory for input tensors.
        **kwargs : dict
            Additional arguments passed to the base Triton model initialization.
        """
        super().__init__(
            client=client,
            model_name=model_name,
            inputs={input_name: (3, input_size.h, input_size.w)},
            outputs=outputs,
            datatype="FP32",
            cushm_inputs=[input_name] if use_cushm else [],
            client_timeout=client_timeout,
            model_version=model_version,
            device_id=device_id,
            **kwargs,
        )
        self.input_size = input_size
        self.image_input_name = input_name
        self.outputs_map = {} if outputs_map is None else dict(outputs_map)
        self.transform = self.get_transform()
        self._batch_size = 1

    def get_heads_names(self) -> Set[str]:
        """Get set of heads names.

        Returns
        -------
        Set[str]
            heads names.
        """
        return {self.outputs_map.get(output, output) for output in self.outputs}

    def get_transform(self) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
        """Get transform for classifier.

        Default config is (0.0, 0.0, 0.0) as mean,
        (1.0, 1.0, 1.0) as std and crop_mode equal to 'center'.

        Returns
        -------
        Callable[[List[NDArray[np.float32]]], NDArray[np.float32]]
            transform sequence.
        """
        return get_transform_sequence(
            size=self.input_size,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            crop_mode="center",
        )

    def preprocess(self, inputs: List[NDArray[np.float32]]) -> Dict[str, NDArray[np.float32]]:
        """Preprocess input image for yolo v5 model.

        Parameters
        ----------
        inputs : np.ndarray
            input image for yolo v5 model represented
            by numpy array of shape (m, n, 3) and type np.uint8.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            mapping from classifiers output's name
            to array with probs of shape (batch_size, num_classes)
        """
        raw_inputs = []
        for input in inputs:
            raw_inputs.append(self.transform(input))
        self._batch_size = len(raw_inputs)
        return {self.image_input_name: np.stack(raw_inputs, axis=0).astype(np.float32)}

    def postprocess(
        self, raw_outputs: Dict[str, NDArray[np.float32]]
    ) -> List[Dict[str, NDArray[np.float32]]]:
        """Postprocess multihead classifier raw outputs.

        Parameters
        ----------
        raw_outputs : Dict[str, NDArray[np.float32]]
            raw outputs from triton model
            Output dict is a mapping from
            model's outputs names to raw
            numpy arrays.

        Returns
        -------
        List[Dict[str, NDArray[np.float32]]]
            batched outputs. Each element of list
            is a dict with key corresponding to multihead
            output name and values representing probabilities
            of labels of this output.
        """
        res = []
        for i in range(self._batch_size):
            x = {}
            for key, value in raw_outputs.items():
                new_key = key
                if key in self.outputs_map:
                    new_key = self.outputs_map[key]
                x[new_key] = value[i, ...]
            res.append(x)
        return res
