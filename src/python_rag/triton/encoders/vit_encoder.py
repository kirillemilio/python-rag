"""Contains implementation of visual transformer encoder."""

from typing import Dict, List, Tuple

import cv2
import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore
from numpy.typing import NDArray

from ...dto import BBox, Size
from ..models_factory import TritonModelFactory
from .crop_encoder import CropEncoderTritonModel, ImageWithBoxes


@TritonModelFactory.register_model(model_type="crop-encoder", arch_type="vit")
class VitCropEncoderTritonModel(CropEncoderTritonModel):
    """
    Visual Transformer (ViT) based model for extracting features from image crops.

    This model adapts the transformer architecture for vision
    tasks, particularly effective in understanding the global context of
    images, suitable for tasks like image classification and object detection.

    Attributes
    ----------
    mean : List[float]
        Normalization mean values.
    std : List[float]
        Normalization standard deviation values.
    """

    mean: List[float]
    std: List[float]

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
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
        **kwargs,
    ):
        """Init vit based crop encoder trition model.

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
        client_timeout : float | None
            Timeout in seconds for the client to wait for a response.
        model_version : str, optional
            The version of the model to use, defaults to "1".
        mean : Tuple[float, float, float], optional
            Mean normalization values used for pre-processing images.
        std : Tuple[float, float, float], optional
            Standard deviation for normalization used in pre-processing.
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
        self.mean = list(map(float, mean))
        self.std = list(map(float, std))

    def preprocess(self, inputs: List[ImageWithBoxes]) -> Dict[str, NDArray[np.float32]]:
        """
        Preprocess input images and bounding boxes for model inference.

        This method applies normalization, resizing, and format conversion
        to each crop extracted using bounding boxes from the input images. Each
        crop is resized to the predefined input size of the model, normalized using
        the specified mean and standard deviation, and transposed to match the
        expected input format of the model.

        Parameters
        ----------
        inputs : List[ImageWithBoxes]
            A list of dictionaries containing 'image' and 'boxes'. Each 'image' is
            an array representing an image, and 'boxes' is an array of coordinates
            for bounding boxes within that image.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            A dictionary with the processed image tensors under their respective
            input names. The tensors are batched along the first dimension if
            multiple images or crops are provided.

        Notes
        -----
        The bounding box coordinates should be provided in the format:
        [x1, y1, x2, y2], where (x1, y1) are the coordinates of the top-left corner,
        and (x2, y2) are the coordinates of the bottom-right corner of the bounding box.

        This preprocessing step is crucial for ensuring that the inputs are in the
        correct format and size for optimal model performance.
        """
        self.split_indices.clear()
        crops = []
        bboxes = []
        for i, img_with_box in enumerate(inputs):
            bboxes_raw = img_with_box["boxes"]
            for i in range(bboxes_raw.shape[0]):
                bbox = BBox(
                    x1=bboxes_raw[i, 0],
                    y1=bboxes_raw[i, 1],
                    x2=bboxes_raw[i, 2],
                    y2=bboxes_raw[i, 3],
                )
                crop = cv2.resize(
                    bbox.crop(img_with_box["image"]),
                    (self.input_size.w, self.input_size.h),  # type: ignore
                )
                crop = crop / 255.0  # type: ignore
                crop = (crop - np.array(self.mean)) / np.array(self.std)
                crop = np.transpose(crop.astype(np.float32), (2, 0, 1))
                crops.append(crop)
                bboxes.append(bboxes_raw)
            self.split_indices.append(img_with_box["boxes"].shape[0])

        return {
            self.image_input_name: np.stack(crops, axis=0),
        }
