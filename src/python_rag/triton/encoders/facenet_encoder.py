"""Contains implementation of facenet encoder."""

import cv2
import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore
from numpy.typing import NDArray

from ...dto import Size
from ..models_factory import TritonModelFactory
from .crop_encoder import CropEncoderTritonModel


@TritonModelFactory.register_model(model_type="crop-encoder", arch_type="facenet")
class FacenetCropEncoderTritonModel(CropEncoderTritonModel):
    """
    Facenet based encoder model for facial recognition.

    This model uses cropped facial images to produce embeddings. It processes input
    images by cropping based on bounding boxes and then encoding them to
    produce embeddings, which are typically used for facial verification
    or identificatio purposes.
    """

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        image_input_name: str,
        input_size: Size,
        output_name: str,
        client_timeout: float | None,
        embedding_size: int = 768,
        model_version: str = "1",
        **kwargs,
    ):
        """Init facenet crop encoder.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            Triton Inference Server instance.
        model_name : str
            Name of the model on the Triton server.
        image_input_name : str
            The input name for image data.
        input_size : Size
            The expected HxW dimensions of the input images.
        output_name : str
            The name of the output tensor for embeddings.
        client_timeout : int
            Timeout in seconds for the client to wait for a response.
        embedding_size : int, optional
            The size of the embedding vector, default is 768.
        model_version : str, optional
            The version of the model to use, defaults to "1".
        **kwargs : dict
            Additional arguments to configure the Triton client.
        """
        super().__init__(
            client=client,
            model_name=model_name,
            image_input_name=image_input_name,
            input_size=input_size,
            output_name=output_name,
            client_timeout=client_timeout,
            embedding_size=embedding_size,
            model_version=model_version,
            **kwargs,
        )

    def process_crop(self, crop: NDArray[np.float32]) -> NDArray[np.float32]:
        """Process crop.

        Parameters
        ----------
        crop : NDArray[np.float32]
            input crop to resize.

        Returns
        -------
        NDArray[np.float32]
            resized crop.
        """
        crop = cv2.resize(crop, (self.input_size.w, self.input_size.h))  # type: ignore
        return np.transpose(crop.astype(np.float32), (2, 0, 1)) / 255.0
