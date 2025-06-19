"""Contains implementation of base text recognizer model."""

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore
from numpy.typing import NDArray

from ...dto import Size
from ..triton_model import BaseTritonModel


class TextRecognizerTritonModel(BaseTritonModel[NDArray[np.float32], str]):
    """Base class for all text recognizer models.

    This class serves as a base for text recognizer models utilizing the Triton Inference Server.
    It handles common configurations and initializations required for text recognition tasks.

    Attributes
    ----------
    input_name : str
        Name of the image input in the Triton model.
    output_name : str
        Name of the recognized text output in the Triton model.
    input_size : Size
        Size of the input image.
    """

    input_name: str
    output_name: str
    input_size: Size

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        input_name: str,
        output_name: str,
        client_timeout: float | None,
        input_size: Size,
        model_version: str = "1",
        device_id: int = 0,
        use_cushm: bool = False,
        **kwargs,
    ):
        """
        Initialize the TextRecognizerTritonModel.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            The Triton inference server client.
        model_name : str
            The name of the model to be used on the Triton server.
        input_name : str
            The name of the input tensor.
        output_name : str
            The name of the output tensor.
        client_timeout : float | None
            Timeout for the client requests in milliseconds.
        input_size : Size
            The expected size of the input images.
        model_version : str, optional
            The version of the model to use, by default "1".
        device_id : int, optional
            The ID of the device to use, by default 0.
        use_cushm : bool, optional
            Whether to use CUDA Shared Memory (CUSHM) for inputs, by default False.
        **kwargs
            Additional keyword arguments to pass to the base class.
        """
        super().__init__(
            client=client,
            model_name=model_name,
            inputs={input_name: (3, input_size.h, input_size.w)},
            outputs=[output_name],
            datatype="FP32",
            client_timeout=client_timeout,
            model_version=model_version,
            device_id=device_id,
            cushm_inputs=[input_name] if use_cushm else None,
            **kwargs,
        )
        self.input_name = input_name
        self.output_name = output_name
        self.input_size = input_size
