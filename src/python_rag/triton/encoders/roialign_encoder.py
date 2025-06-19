"""Contains implementation of base class for roi align encoder models."""

from typing import Dict, List, cast

import numpy as np
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray

from ...dto import Size
from ..models_factory import TritonModelFactory
from .base_encoder import BaseEncoderTritonModel, ImageWithBoxes


@TritonModelFactory.register_model(model_type="roialign-encoder", arch_type="roialign-encoder")
class RoiAlignEncoderTritonModel(BaseEncoderTritonModel):
    """
    Implement a Triton model encoder using ROI (Region of Interest) alignment.

    This model is suitable for tasks that require extracting features from specific regions of
    an input image.

    Attributes
    ----------
    image_input_name : str
        The name of the input tensor for images.
    bbox_input_name : str
        The name of the input tensor for bounding boxes.
    output_name : str
        The name of the output tensor for embeddings.
    input_size : Size
        The expected size of the input images.
    embedding_size : int
        The dimensionality of the output embeddings.
    split_indices : List[int]
        Indices to split the batched outputs into individual results.

    Parameters
    ----------
    client : grpcclient.InferenceServerClient
        Triton Inference Server instance.
    model_name : str
        The name of the model on the Triton server.
    image_input_name : str
        The input name for image data.
    bbox_input_name : str
        The input name for bounding box data.
    input_size : Size
        The expected HxW dimensions of the input images.
    output_name : str
        The name of the model output used for embeddings.
    client_timeout : float | None
        Timeout in seconds for the client to wait for a response from the server.
    embedding_size : int
        The size of the embedding vector.
    model_version : str, optional
        The version of the model to use.
    device_id : int, optional
        The GPU device ID to use for inference, if applicable.
    use_cushm : bool, optional
        Indicates whether to use CUDA shared memory for input/output.
    **kwargs
        Additional keyword arguments for the Triton client configuration.
    """

    image_input_name: str
    bbox_input_name: str
    output_name: str
    input_size: Size

    embedding_size: int

    split_indices: List[int]

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        image_input_name: str,
        bbox_input_name: str,
        input_size: Size,
        output_name: str,
        client_timeout: float | None,
        embedding_size: int,
        model_version: str = "1",
        device_id: int = 0,
        use_cushm: bool = False,
        **kwargs,
    ):
        """Initialize roi align encoder triton model.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            Triton Inference Server instance.
        model_name : str
            Name of the model deployed on Triton server.
        image_input_name : str
            Input name for the image data.
        bbox_input_name : str
            Input name for the bounding box data.
        input_size : Size
            Size (height, width) of the images.
        output_name : str
            Output tensor name for the embeddings.
        client_timeout : float | None
            Timeout for the client when making requests.
        embedding_size : int
            Size of the embedding vector.
        model_version : str, optional
            Version of the model to use, defaults to "1".
        device_id : int, optional
            Device ID used for GPU computations.
        use_cushm : bool, optional
            Whether to use CUDA shared memory.
        **kwargs : dict
            Additional arguments to configure the Triton client.
        """
        super().__init__(
            client=client,
            model_name=model_name,
            inputs={
                image_input_name: (3, input_size.h, input_size.w),
                bbox_input_name: (5,),
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
        self.bbox_input_name = bbox_input_name
        self.output_name = output_name
        self.embedding_size = int(embedding_size)
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

    def preprocess(self, inputs: List[ImageWithBoxes]) -> Dict[str, NDArray[np.float32]]:
        """
        Convert input data into the format expected by the Triton server.

        Specifically formatting and batching images and their corresponding
        bounding boxes for ROI alignment processing.

        Parameters
        ----------
        inputs : List[ImageWithBoxes]
            A list of dictionaries, each containing an image and its corresponding
            bounding boxes.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            A dictionary of preprocessed data arrays keyed by input tensor names.
        """
        self.split_indices.clear()
        images, bboxes = [], []
        for i, img_with_box in enumerate(inputs):
            images.append(self.preprocess_image(img_with_box["image"], i))
            bboxes.append(self.preprocess_bbox(img_with_box["boxes"], i))
            self.split_indices.append(img_with_box["boxes"].shape[0])

        return {
            self.image_input_name: np.stack(images, axis=0),
            self.bbox_input_name: np.concatenate(bboxes, axis=0),
        }

    def preprocess_image(self, img: NDArray[np.float32], image_id: int) -> NDArray[np.float32]:
        """Preprocess single input image.

        Parameters
        ----------
        img : NDArray[np.float32]
            input image.
        image_id : int
            input image id in batch.

        Returns
        -------
        NDArray[np.float32]
            input image after preprocessing.
        """
        return np.transpose(img.astype(np.float32), (2, 0, 1))

    def preprocess_bbox(self, bboxes: NDArray[np.float32], image_id: int) -> NDArray[np.float32]:
        """Preprocess single image bounding boxes.

        Parameters
        ----------
        bboxes : NDArray[np.float32]
            input image bounding boxes.
        image_id : int
            input image id.

        Returns
        -------
        NDArray[np.float32]
            preprocessed bounding boxes.
        """
        indices = image_id * np.ones((bboxes.shape[0], 1), dtype=np.float32)
        return np.concatenate([indices, bboxes], axis=1)

    def postprocess(self, raw_outputs: Dict[str, NDArray[np.float32]]) -> List[NDArray[np.float32]]:
        """
        Postprocess the raw outputs from the Triton server.

        Normalize and prepare the embeddings for further use or analysis.

        Parameters
        ----------
        raw_outputs : Dict[str, NDArray[np.float32]]
            The raw outputs from the Triton inference.

        Returns
        -------
        List[NDArray[np.float32]]
            A list of processed and normalized embedding arrays.
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
        res = np.split(features, self.split_indices, axis=0)
        self.split_indices.clear()
        return res
