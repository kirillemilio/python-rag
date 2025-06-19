"""Contains implementation of base class for crop encoders models."""

from typing import Dict, List

import cv2
import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore
from numpy.typing import NDArray

from ...dto import BBox, Size
from ..models_factory import TritonModelFactory
from .base_encoder import BaseEncoderTritonModel, ImageWithBoxes


@TritonModelFactory.register_model(model_type="crop-encoder", arch_type="crop-encoder")
class CropEncoderTritonModel(BaseEncoderTritonModel):
    """
    A specialized encoder model that processes image crops for encoding.

    This model takes images along with their bounding box data,
    extracts the relevant crops, and encodes them into embeddings of a specified size.

    Attributes
    ----------
    image_input_name : str
        The name of the input tensor for the image data.
    output_name : str
        The name of the output tensor for the embeddings.
    input_size : Size
        The dimensions to which input images are resized.
    embedding_size : int
        The dimensionality of the output embeddings.
    split_indices : List[int]
        Indices at which the batch of crops is split into individual samples.
    """

    image_input_name: str
    output_name: str
    input_size: Size

    embedding_size: int

    split_indices: List[int]

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        image_input_name: str,
        input_size: Size,
        output_name: str,
        client_timeout: float | None,
        embedding_size: int,
        model_version: str = "1",
        device_id: int = 0,
        use_cushm: bool = False,
        **kwargs,
    ):
        """
        Initialize the CropEncoderTritonModel with necessary configuration.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            Triton Inference Server instance.
        model_name : str
            The name of the model to deploy.
        image_input_name : str
            The name of the image input tensor.
        input_size : Size
            The size to which images are resized before processing.
        output_name : str
            The name of the output tensor for the embeddings.
        client_timeout : float | None
            Timeout in seconds for the client to wait for a response.
        embedding_size : int
            Size of the embeddings produced by the model.
        model_version : str, optional
            Version of the model to use, default is "1".
        **kwargs : dict
            Additional keyword arguments for Triton client configuration.
        """
        super().__init__(
            client=client,
            model_name=model_name,
            inputs={
                image_input_name: (3, input_size.h, input_size.w),
            },
            outputs=[output_name],
            cushm_inputs=[image_input_name] if use_cushm else [],
            datatype="FP32",
            client_timeout=client_timeout,
            model_version=model_version,
            embedding_size=embedding_size,
            device_id=device_id,
            **kwargs,
        )
        self.input_size = input_size
        self.image_input_name = image_input_name
        self.output_name = output_name
        self.split_indices = []

    def get_image_input_name(self) -> str:
        """Get image input name.

        Returns
        -------
        str
            image input's name.
        """
        return self.image_input_name

    def get_image_input_size(self) -> Size:
        """Get image input's size.

        Returns
        -------
        Size
            size of image input.
        """
        return self.input_size

    def process_crop(self, crop: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Process a single crop to prepare it for model input.

        This method resizes the crop to the target input size and adjusts its
        data format and type appropriately.

        Parameters
        ----------
        crop : NDArray[np.float32]
            A crop from the input image.

        Returns
        -------
        NDArray[np.float32]
            The processed crop suitable for input to the model.
        """
        crop = cv2.resize(crop, (self.input_size.w, self.input_size.h))  # type: ignore
        return np.transpose(crop.astype(np.float32), (2, 0, 1))

    def preprocess(self, inputs: List[ImageWithBoxes]) -> Dict[str, NDArray[np.float32]]:
        """
        Prepare the input data for model inference.

        This method handles preprocessing steps required to format the inputs according
        to the model's expectations. This includes cropping the images based on the
        provided bounding boxes, resizing them, and formatting the data.

        Parameters
        ----------
        inputs : List[ImageWithBoxes]
            A list of dictionaries, each containing keys for 'image' and 'boxes'.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            A dictionary with model input names as keys and preprocessed data arrays as values.
        """
        self.split_indices.clear()
        crops = []
        bboxes = []
        for i, img_with_box in enumerate(inputs):
            bboxes_raw = img_with_box["boxes"]
            for k in range(bboxes_raw.shape[0]):
                bbox = BBox(
                    x1=bboxes_raw[k, 0],
                    y1=bboxes_raw[k, 1],
                    x2=bboxes_raw[k, 2],
                    y2=bboxes_raw[k, 3],
                )
                crop = self.process_crop(bbox.crop(img_with_box["image"]))  # type: ignore
                crops.append(crop)
                bboxes.append(bboxes_raw)
            if i < len(inputs) - 1:
                self.split_indices.append(bboxes_raw.shape[0])

        return {
            self.image_input_name: np.stack(crops, axis=0),
        }

    def postprocess(self, raw_outputs: Dict[str, NDArray[np.float32]]) -> List[NDArray[np.float32]]:
        """
        Postprocess the outputs of the model to convert raw embedding arrays.

        This method normalizes and rescales the outputs back to the original
        image dimensions and splits the batch of features into a list of
        arrays corresponding to each input image.

        Parameters
        ----------
        raw_outputs : Dict[str, NDArray[np.float32]]
            Dictionary of raw outputs from the model.

        Returns
        -------
        List[NDArray[np.float32]]
            A list of processed outputs where each entry corresponds to the
            features of an individual image, adjusted for original size.
        """
        features = raw_outputs[self.output_name]

        features = features.astype(np.float64)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        np.clip(features, a_min=-1e20, a_max=1e20, out=features)
        if np.any(np.isnan(features)):
            raise ValueError(f"Some features from osnet are nan: {features}")
        if np.linalg.norm(features) == np.nan:
            raise ValueError(f"Features from osnet has nan norm: {features}")
        if np.linalg.norm(features) == 0.0:
            raise ValueError(f"Features from osnet has 0.0 norm: {features}")
        if np.linalg.norm(features) == np.inf:
            raise ValueError(f"Features from osnet has inf norm: {features}")
        res = np.split(features, np.cumsum(self.split_indices), axis=0)
        self.split_indices.clear()
        return res
