"""Contains implementationf of base points detection model."""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Literal, TypedDict

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore
from numpy.typing import NDArray

from ...dto import Size
from ..triton_model import BaseTritonModel


class PointsOutputDict(TypedDict):
    """Point detection model output dictionary.

    Attributes
    ----------
    points : NDArray[np.float32]
        points are represented by numpy array of dtype=np.float32
        and shape (n, k, 2). n size is corresponding to index
        of detected polygon, next dimension(k)
        is responsible for indexing points of this polygon,
        while the last dimension(2) is responsible for x,y
        coordinates discrimination.
    probs : NDArray[np.float32]
        probabilities array represented by array of dtype=np.float32
        and shape n.
    """

    points: NDArray[np.float32]  # (n, k, 2) shape, where last dimension is in x,y format
    probs: NDArray[np.float32]  # (n, ) shape corresponding to probability


class PointsDetectorTritonModel(BaseTritonModel[NDArray[np.float32], PointsOutputDict]):
    """Base model for points detection models."""

    input_name: str
    output_name: str
    input_size: Size

    iou_threshold: float
    conf_threshold: float

    original_size_list: List[Size]
    channels_last: bool

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        input_name: str,
        input_size: Size,
        output_name: str,
        client_timeout: float | None,
        iou_threshold: float,
        conf_threshold: float,
        model_version: str = "1",
        device_id: int = 0,
        use_cushm: bool = False,
        channels_last: bool = False,
        **kwargs,
    ):
        self.channel_last = channels_last
        if channels_last:
            triton_input_shape = (input_size.h, input_size.w, 3)
        else:
            triton_input_shape = (3, input_size.h, input_size.w)
        super().__init__(
            client=client,
            model_name=model_name,
            inputs={input_name: triton_input_shape},
            outputs=[output_name],
            datatype="FP32",
            client_timeout=client_timeout,
            model_version=model_version,
            device_id=device_id,
            cushm_inputs=[input_name] if use_cushm else None,
            **kwargs,
        )
        self.original_size_list = []
        self.input_name = input_name
        self.output_name = output_name
        self.input_size = input_size
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def set_batch_size(self, input_name: str, batch_size: int) -> None:
        """Set batch size for encoder models.

        Parameters
        ----------
        batch_size : int
            batch size to set for input
            of triton model in order
            to pass inputs in batched mode.
        """
        return

    def set_original_size_list(self, images: List[NDArray[np.float32]]):
        """Store original input images sizes for inference.

        Stored sizes will be used for outputs rescale and callibration
        purposes typically inside postprocess method.

        Parameters
        ----------
        images : List[NDArray[np.float32]]
            list of input images whose sizes will
            be saved.
        """
        for image in images:
            self.original_size_list.append(Size(w=image.shape[1], h=image.shape[0]))

    def clear_original_size_list(self):
        """Clear the list containing original images sizes.

        This method must be called after each completion
        of model inference, typically in postprocess method.
        """
        self.original_size_list.clear()

    @abstractmethod
    def get_crops(
        self,
        image: NDArray[np.float32],
        output: PointsOutputDict,
        crop_mode: Literal["perspective", "borders"],
    ) -> List[NDArray[np.float32]]:
        """Get list of crops from image using iwpodnet points.

        Parameters
        ----------
        image : NDArray[np.float32]
            source image from which crops will be extracted.
        output : PointsOutputDict
            detected points dict.
        crop_mode : Literal["perspective", "borders"]
            switch to select iwpodnet postprocessing method

        Returns
        -------
        List[NDArray[np.float32]]
            list of crops from the image.
        """
        raise NotImplementedError()
