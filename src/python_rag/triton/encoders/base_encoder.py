"""Contains implementation of base encoder model."""

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray

from ...dto import Size
from ..triton_model import BaseTritonModel


class ImageWithBoxes(TypedDict):
    """Image with boxes typed dict."""

    image: NDArray[np.float32]
    boxes: NDArray[np.float32]


class BaseEncoderTritonModel(BaseTritonModel[ImageWithBoxes, NDArray[np.float32]]):
    """
    Base class for all encoders triton models.

    Attributes
    ----------
    embedding_size : int
        embedding size of the encoder.

    Methods
    -------
    get_image_input_name()
        abstract method to return the input name for the image data.
    get_image_input_size()
        abstract method to return the input image size.
    get_embedding_size()
        returns the size of the embeddings.
    """

    embedding_size: int

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        inputs: Dict[str, Tuple[int, ...]],
        outputs: List[str],
        datatype: str,
        client_timeout: float | None,
        embedding_size: int,
        model_version: str = "1",
        device_id: int = 0,
        cushm_inputs: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the base encoder Triton model with specific configurations.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            Triton Inference Server instance.
        model_name : str
            The name of the model to deploy.
        inputs : Dict[str, Tuple[int, ...]]
            Definitions of the input data shapes.
        outputs : List[str]
            Names of the outputs to be produced by the model.
        datatype : str
            Data type of the inputs, e.g., 'FP32'.
        client_timeout : float | None
            Timeout in seconds for the client to wait for a response.
        embedding_size : int
            Size of the embeddings produced by the model.
        model_version : str, optional
            Version of the model to use, by default "1".
        device_id : int, optional
            Device ID for CUDA shared memory, by default 0.
        cushm_inputs : Optional[List[str]], optional
            List of input names to use CUDA shared memory, by default None.
        **kwargs : dict
            Additional keyword arguments for Triton client configuration.
        """
        super().__init__(
            client=client,
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            cushm_inputs=cushm_inputs,
            datatype=datatype,
            client_timeout=client_timeout,
            model_version=model_version,
            device_id=device_id,
            **kwargs,
        )

        self.embedding_size = int(embedding_size)

    @abstractmethod
    def get_image_input_name(self) -> str:
        """Abstract method that returns image input name.

        Returns
        -------
        str
            image input's name.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_image_input_size(self) -> Size:
        """Abstract method that returns image input size.

        Returns
        -------
        Size
            size of image input.
        """
        raise NotImplementedError()

    def get_embedding_size(self) -> int:
        """
        Get the size of the embeddings produced by the model.

        Returns
        -------
        int
            The embedding size.
        """
        return self.embedding_size
