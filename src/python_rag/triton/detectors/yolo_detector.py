"""Contains implementation of yolo family models."""

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from ..models_factory import TritonModelFactory
from .detector import DetectorInputDict, DetectorTritonModel
from .utils import letterbox, scale_coords


@TritonModelFactory.register_model(model_type="detector", arch_type="yolov5")
class YoloV5TritonModel(DetectorTritonModel):
    """
    A Triton model subclass for YOLOv5 object detection which adapts the general detector model.

    Framework for specific preprocessing and postprocessing
    required by YOLOv5 models.

    This model handles specific transformations and scaling involved in
    YOLOv5's detection pipeline, ensuring images are correctly formatted and
    postprocessed to output detected objects with bounding boxes.
    """

    def preprocess(self, inputs: List[DetectorInputDict]) -> Dict[str, NDArray[np.float32]]:
        """Preprocess input images by resizing to a target size, normalizing.

        Also restructuring for batch processing, alongside preparing threshold
        values for object detection.

        Parameters
        ----------
        inputs : List[DetectorInputDict]
            A list of input dictionaries each containing
            an image and detection thresholds.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            Dictionary of processed images and thresholds
            formatted for Triton inference server.
        """
        self.set_original_size_list(inputs)
        images = []
        iou_thresholds = []
        conf_thresholds = []
        for input_dict in inputs:
            preproc_input, _, _ = letterbox(
                input_dict["image"], new_shape=(self.input_size.h, self.input_size.w), auto=False
            )
            preproc_input = preproc_input.transpose(2, 0, 1)
            preproc_input = np.ascontiguousarray(preproc_input).astype(np.float32)
            preproc_input /= 255.0
            preproc_input = np.expand_dims(preproc_input, axis=0)
            images.append(preproc_input)
            iou_thresholds.append(np.array([[input_dict["iou_threshold"]]], dtype=np.float32))
            conf_thresholds.append(np.array([[input_dict["conf_threshold"]]], dtype=np.float32))

        return {
            self.input_name: np.concatenate(images, axis=0),
            self.iou_threshold_name: np.concatenate(iou_thresholds, axis=0),
            self.conf_threshold_name: np.concatenate(conf_thresholds, axis=0),
        }

    def postprocess(self, raw_outputs: Dict[str, NDArray[np.float32]]) -> List[NDArray[np.float32]]:
        """Postprocess raw outputs from the Triton server.

        Typically scale coordinates
        of detected objects back to the original image dimensions.

        Parameters
        ----------
        raw_outputs : Dict[str, NDArray[np.float32]]
            The raw outputs from the Triton server.

        Returns
        -------
        List[NDArray[np.float32]]
            List of processed detection results with adjusted
            bounding box coordinates.
        """
        res = raw_outputs[self.output_name]
        output = []
        for i in range(self.get_batch_size(self.input_name)):
            orig_size = self.original_size_list[i]
            det = res[i, :, :]
            det = det[det[:, 0] != -1, :]
            det[:, 1:5] = scale_coords(
                coords=det[:, 1:5],
                input_shape=(self.input_size.h, self.input_size.w),
                output_shape=(orig_size.h, orig_size.w),
            ).round()
            output.append(det)
        return output


@TritonModelFactory.register_model(model_type="detector", arch_type="yolov7")
class YoloV7TritonModel(YoloV5TritonModel):
    """A specific implementation of YoloV5TritonModel for YOLOv7.

    This model adapts postprocessing
    to the output structure of YOLOv7 models, ensuring correct extraction and scaling
    of bounding boxes based on model outputs.

    Inherits the preprocessing method from YoloV5TritonModel and only adjusts the postprocessing
    to handle differences in output structure of YOLOv7.
    """

    def postprocess(self, raw_outputs: Dict[str, NDArray[np.float32]]) -> List[NDArray[np.float32]]:
        """
        Postprocess raw outputs specifically for YOLOv7 model outputs.

        Bounding boxes are adapted to the original image sizes.

        Parameters
        ----------
        raw_outputs : Dict[str, NDArray[np.float32]]
            The raw outputs from the Triton server.

        Returns
        -------
        List[NDArray[np.float32]]
            List of processed detection results with adjusted
            bounding box coordinates for YOLOv7 specifics.
        """
        res = raw_outputs[self.output_name]
        output = []
        for i in range(self.get_batch_size(self.input_name)):
            orig_size = self.original_size_list[i]
            det = res[res[:, 0] == i, :]
            det = det[:, [5, 1, 2, 3, 4, 6]]
            det = det[det[:, 0] != -1, :]
            det[:, 1:5] = scale_coords(
                coords=det[:, 1:5],
                input_shape=(self.input_size.h, self.input_size.w),
                output_shape=(orig_size.h, orig_size.w),
            ).round()
            output.append(det)
        return output
