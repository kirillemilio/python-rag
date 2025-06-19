"""Contains implementation of base model for detector triton models."""

from typing import Dict, List, Optional, TypedDict

import numpy as np
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray

from ...dto import Size
from ..triton_model import BaseTritonModel


class DetectorInputDict(TypedDict):
    """Input dict type for detector model."""

    image: NDArray[np.float32]
    iou_threshold: float
    conf_threshold: float


class DetectorTritonModel(BaseTritonModel[DetectorInputDict, NDArray[np.float32]]):
    """
    A specialized model class designed for object detection tasks.

    Model handles both the preprocessing and postprocessing steps necessary
    for interfacing with a Triton
    Inference Server. This class manages detection-specific configurations like
    IOU and confidence thresholds.

    Attributes
    ----------
    input_name : str
        The name of the input tensor for image data.
    output_name : str
        The name of the output tensor where detection results are stored.
    input_size : Size
        Dimensions (height and width) to which input images are resized.
    iou_threshold_name : str
        The name of the input tensor for the IOU threshold.
    conf_threshold_name : str
        The name of the input tensor for the confidence threshold.
    iou_threshold_default : float
        Default value for the IOU threshold if not specified at runtime.
    conf_threshold_default : float
        Default value for the confidence threshold if not specified at runtime.
    original_size_list : List[Size]
        A list that stores the original sizes of images for processing output data.
    """

    input_name: str
    output_name: str
    input_size: Size

    iou_threshold_name: str
    conf_threshold_name: str

    iou_threshold_default: float
    conf_threshold_default: float

    original_size_list: List[Size]

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        input_name: str,
        input_size: Size,
        output_name: str,
        client_timeout: float | None,
        iou_threshold_default: float,
        conf_threshold_default: float,
        iou_threshold_name: str = "iou_threshold",
        conf_threshold_name: str = "conf_threshold",
        model_version: str = "1",
        device_id: int = 0,
        use_cushm: bool = False,
        **kwargs,
    ):
        """Initialize a detector model with specific configuration.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            Triton Inference Server instance.
        model_name : str
            Name of the model as registered in the Triton server.
        input_name : str
            Name of the model input tensor for image data.
        input_size : Size
            Size object specifying the height and width to which input images will be resized.
        output_name : str
            Name of the output tensor for detection results.
        client_timeout : float | None
            Timeout in seconds for the server to respond.
        iou_threshold_default : float
            Default IOU threshold for filtering detections.
        conf_threshold_default : float
            Default confidence threshold for filtering detections.
        iou_threshold_name : str, optional
            Name of the input tensor for IOU threshold, by default "iou_threshold".
        conf_threshold_name : str, optional
            Name of the input tensor for confidence threshold, by default "conf_threshold".
        model_version : str, optional
            Version of the model to use, by default "1".
        device_id : int, optional
            Device ID for CUDA shared memory operations, by default 0.
        use_cushm : bool, optional
            Flag to indicate whether CUDA shared memory should be used, by default False.
        **kwargs : dict
            Additional keyword arguments for configuration.
        """
        super().__init__(
            client=client,
            model_name=model_name,
            inputs={
                input_name: (3, input_size.h, input_size.w),
                iou_threshold_name: (1,),
                conf_threshold_name: (1,),
            },
            outputs=[output_name],
            cushm_inputs=[input_name] if use_cushm else [],
            client_timeout=client_timeout,
            model_version=model_version,
            datatype="FP32",
            device_id=device_id,
            **kwargs,
        )
        self.input_name = input_name
        self.output_name = output_name

        self.iou_threshold_name = iou_threshold_name
        self.conf_threshold_name = conf_threshold_name

        self.iou_threshold_default = iou_threshold_default
        self.conf_threshold_default = conf_threshold_default

        self.input_size = input_size
        self.original_size_list = []

    def create_input_dict(
        self,
        image: NDArray[np.float32],
        iou_threshold: Optional[float] = None,
        conf_threshold: Optional[float] = None,
    ) -> DetectorInputDict:
        """
        Create and return a dictionary suitable for input to the detector model.

        This method prepares the input dictionary required for detecting objects using the model.
        It includes the image, IOU threshold, and confidence threshold. If IOU or confidence
        thresholds are not provided, default values are used. Both thresholds must
        be within the range [0.0, 1.0].

        Parameters
        ----------
        image : NDArray[np.float32]
            The image data as a numpy array with the format specified by the model
            (e.g., HWC format).
        iou_threshold : float, optional
            The intersection over union (IOU) threshold for detection filtering. If not specified,
            the model's default IOU threshold is used. Must be between 0.0 and 1.0, inclusive.
        conf_threshold : float, optional
            The confidence threshold for detections. If not specified,
            the model's default confidence
            threshold is used. Must be between 0.0 and 1.0, inclusive.

        Returns
        -------
        DetectorInputDict
            A dictionary containing the image, IOU threshold, and confidence threshold ready to be
            processed by the detector model.

        Raises
        ------
        ValueError
            If either `iou_threshold` or `conf_threshold` is outside the range [0.0, 1.0].

        Examples
        --------
        >>> image_array = np.random.rand(224, 224, 3).astype(np.float32)
        >>> input_dict = model.create_input_dict(image=image_array,
                                                 iou_threshold=0.5,
                                                 conf_threshold=0.5)
        >>> print(input_dict)
        {'image': array([...]), 'iou_threshold': 0.5, 'conf_threshold': 0.5}
        """
        iou_threshold_ = self.iou_threshold_default if iou_threshold is None else iou_threshold
        conf_threshold_ = self.conf_threshold_default if conf_threshold is None else conf_threshold
        if iou_threshold_ < 0.0 or iou_threshold_ > 1.0:
            raise ValueError("Parameter `iou_threshold` must be in [0.0, 1.0] range")
        if conf_threshold_ < 0.0 or conf_threshold_ > 1.0:
            raise ValueError("Parameter `conf_threshold` must be in [0.0, 1.0] range")
        return {"image": image, "iou_threshold": iou_threshold_, "conf_threshold": conf_threshold_}

    def set_original_size_list(self, inputs: List[DetectorInputDict]):
        """Store the original dims of each input image in the list.

        Parameters
        ----------
        inputs : List[DetectorInputDict]
            List of input dictionaries containing the images being processed.

        Notes
        -----
        This method is crucial for accurately scaling the detection outputs back to
        the original image dimensions.
        """
        for input_dict in inputs:
            image = input_dict["image"]
            self.original_size_list.append(Size(h=image.shape[0], w=image.shape[1]))

    def clear_original_size_list(self):
        """
        Clear the list that holds the original sizes of processed images.

        This method should be called after completing the processing of a batch
        to reset the list for the next batch of images.
        """
        self.original_size_list.clear()

    def preprocess(self, inputs: List[DetectorInputDict]) -> Dict[str, NDArray[np.float32]]:
        """
        Prepare the input data for model inference.

        This method handles preprocessing steps required to format the inputs according to
        the model's expectations. This includes stacking images into a batch for the model
        and possibly converting image formats and data types.

        Parameters
        ----------
        inputs : List[DetectorInputDict]
            A list of dictionaries, each containing keys for 'image', 'iou_threshold',
            and 'conf_threshold'.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            A dictionary with model input names as keys and preprocessed data arrays as values.

        Examples
        --------
        >>> inputs = [{'image': np.random.rand(224, 224, 3).astype(np.float32),
                       'iou_threshold': 0.5, 'conf_threshold': 0.5}]
        >>> preprocessed_inputs = model.preprocess(inputs)
        >>> print(list(preprocessed_inputs.keys()))
        ['input_name', 'iou_threshold_name', 'conf_threshold_name']
        """
        self.set_original_size_list(inputs)
        return {
            self.input_name: np.stack([input_dict["image"] for input_dict in inputs]).astype(
                np.float32
            ),
            self.iou_threshold_name: np.array(
                [[input_dict["iou_threshold"] for input_dict in inputs]], dtype=np.float32
            ),
            self.conf_threshold_name: np.array(
                [[input_dict["conf_threshold"] for input_dict in inputs]], dtype=np.float32
            ),
        }

    def postprocess(self, raw_outputs: Dict[str, NDArray[np.float32]]) -> List[NDArray[np.float32]]:
        """Convert the raw output from the Triton model into a structured format.

        Parameters
        ----------
        raw_outputs : Dict[str, NDArray[np.float32]]
            The raw outputs from the Triton inference call, typically including
            detected classes, scores, and bounding box coordinates.

        Returns
        -------
        List[NDArray[np.float32]]
            A list of processed outputs, each corresponding to an input image.
        """
        self.clear_original_size_list()
        return [raw_outputs[self.output_name]]
