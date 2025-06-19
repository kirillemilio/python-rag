"""Contains implementation of image encoder triton model."""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore
from numpy.typing import NDArray
from transformers import AutoImageProcessor

from ...dto import Size
from ..triton_model import BaseTritonModel


class BaseImageEncoderTritonModel(BaseTritonModel[NDArray[np.uint8], NDArray[np.float32]]):
    """Defines a base class for image encoder models deployed on Triton.

    This class provides an interface for preprocessing image inputs,
    managing Triton inference, and extracting embeddings from the model's
    outputs.

    Attributes
    ----------
    image_input_name : str
        Name of the model's image input field.
    hidden_output_name : str
        Name of the hidden state output (optional).
    embeddings_output_name : str
        Name of the output containing final image embeddings.
    input_size : Size
        Expected input resolution (height and width).
    embedding_size : int
        Size of the output embedding vector.
    preprocessor : AutoImageProcessor
        Hugging Face image processor used for preprocessing.

    Methods
    -------
    create_preprocessor()
        Create an image processor (to be implemented by subclass).
    preprocess(inputs)
        Convert raw image data into input tensors.
    postprocess(raw_outputs)
        Extract and return image embeddings from Triton output.
    get_embedding_size()
        Return the dimensionality of the output embeddings.
    """

    image_input_name: str
    hidden_output_name: str
    embeddings_output_name: str

    input_size: Size

    embedding_size: int
    preprocesor: AutoImageProcessor

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        image_input_name: str,
        hidden_output_name: str,
        embeddings_output_name: str,
        input_size: Size,
        datatype: str,
        client_timeout: float | None,
        embedding_size: int,
        model_version: str = "1",
        device_id: int = 0,
        cushm_inputs: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the base image encoder Triton model.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            Triton Inference Server client instance.
        model_name : str
            Name of the model as registered in the Triton server.
        image_input_name : str
            Name of the input field for image data.
        hidden_output_name : str
            Name of the model's hidden state output (optional).
        embeddings_output_name : str
            Name of the output containing final image embeddings.
        input_size : Size
            Expected input resolution (height and width).
        datatype : str
            Data type of the inputs (e.g., "FP32").
        client_timeout : float or None
            Timeout in seconds for inference requests.
        embedding_size : int
            Size of the embedding vectors produced by the model.
        model_version : str, optional
            Version of the model to use. Default is "1".
        device_id : int, optional
            CUDA device ID for shared memory. Default is 0.
        cushm_inputs : list of str or None, optional
            Input names to use CUDA shared memory for. Default is None.
        **kwargs : dict
            Additional arguments passed to the base Triton model class.
        """
        super().__init__(
            client=client,
            model_name=model_name,
            inputs={image_input_name: (input_size.h, input_size.w)},
            outputs=[hidden_output_name, embeddings_output_name],
            cushm_inputs=cushm_inputs,
            datatype=datatype,
            client_timeout=client_timeout,
            model_version=model_version,
            device_id=device_id,
            allow_spatial_adjustment=True,
            **kwargs,
        )

        self.image_input_name = image_input_name
        self.input_size = input_size
        self.hidden_output_name = str(hidden_output_name)
        self.embeddings_output_name = str(embeddings_output_name)
        self.embedding_size = int(embedding_size)
        self.preprocesor = self.create_preprocessor()

    @abstractmethod
    def create_preprocessor(self) -> AutoImageProcessor:
        """
        Create and return a Hugging Face image processor.

        This method must be implemented in a subclass to initialize an image
        processor appropriate for the deployed model.

        Returns
        -------
        AutoImageProcessor
            Hugging Face image processor instance.
        """
        raise NotImplementedError()

    def preprocess(self, inputs: List[NDArray[np.uint8]]) -> Dict[str, NDArray[np.float32]]:
        """
        Convert input image data into model-compatible tensors.

        Parameters
        ----------
        inputs : list of NDArray[np.uint8]
            List of raw input images in uint8 format (H x W x C).

        Returns
        -------
        dict of str to NDArray[np.float32]
            Dictionary containing model input tensors.
        """
        res = self.preprocesor(inputs, return_tensors="pt")  # type: ignore
        return {
            self.image_input_name: res["pixel_values"].float().numpy(),
        }

    def postprocess(self, raw_outputs: Dict[str, NDArray[np.float32]]) -> List[NDArray[np.float32]]:
        """
        Extract embeddings from Triton model output.

        Parameters
        ----------
        raw_outputs : dict of str to NDArray[np.float32]
            Raw output from Triton inference containing image embeddings.

        Returns
        -------
        list of NDArray[np.float32]
            List of embedding vectors, one per input image.
        """
        x = raw_outputs[self.embeddings_output_name]
        return [x[i, ...] for i in range(x.shape[0])]

    def get_embedding_size(self) -> int:
        """
        Return the size of the output embedding vectors.

        Returns
        -------
        int
            Dimensionality of each output embedding.
        """
        return self.embedding_size
