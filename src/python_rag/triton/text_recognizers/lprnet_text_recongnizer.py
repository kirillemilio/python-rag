"""Contains implementation of lprnet for licence plate recognition."""

from typing import ClassVar, Dict, List

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray

from ...dto import Size
from ..models_factory import TritonModelFactory
from .text_recognizer import TextRecognizerTritonModel


@TritonModelFactory.register_model(model_type="text-recognizer", arch_type="lprnet")
class LPRNetTextRecognizerTritonModel(TextRecognizerTritonModel):
    """Triton model for license plate recognition using LPRNet.

    This class extends the base text recognizer model to provide specific functionality
    for recognizing license plates using the LPRNet architecture.

    Attributes
    ----------
    chars : ClassVar[List[str]]
        List of characters that can be recognized by the model.
    """

    chars: ClassVar[List[str]] = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "А",
        "В",
        "С",
        "Е",
        "Н",
        "К",
        "М",
        "О",
        "Р",
        "Т",
        "Х",
        "У",
        "-",
    ]

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        input_name: str,
        output_name: str,
        client_timeout: int,
        input_size: Size = Size(h=50, w=100),
        model_version: str = "1",
        device_id: int = 0,
        use_cushm: bool = False,
        **kwargs,
    ):
        """
        Initialize the LPRNetTriton model.

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
        client_timeout : int
            Timeout for the client requests in milliseconds.
        input_size : Size, optional
            The expected size of the input images, by default Size(h=50, w=100).
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
            input_name=input_name,
            output_name=output_name,
            client_timeout=client_timeout,
            input_size=input_size,
            model_version=model_version,
            device_id=device_id,
            use_cushm=use_cushm,
            **kwargs,
        )

    def _preprocess_one_image(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        """Preprocess one image.

        Preprocess one image by resizing to input size,
        substracing 127.5 and scaling.
        Image is converted to channels first format
        with batch size added.

        Parameters
        ----------
        image : NDArray[np.float32]
            input image with licence plate.

        Returns
        -------
        NDArray[np.float32]
            preprocessed image with licence plate.
        """
        image = cv2.resize(
            image, (self.input_size.w, self.input_size.h), interpolation=cv2.INTER_CUBIC
        )
        image = image.astype("float32")
        image -= 127.5
        image *= 0.0078125
        image = np.transpose(image, (2, 0, 1))
        image = image[np.newaxis, :]
        return image

    def preprocess(self, inputs: List[NDArray[np.float32]]) -> Dict[str, NDArray[np.float32]]:
        """Run preprocessing on list of input images.

        Parameters
        ----------
        inputs : List[NDArray[np.float32]]
            input images to preprocess.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            feed dict for triton model
            where key is the name of input
            and values is numpy array for this inputs.
        """
        preprocessed_images_list = []
        for image in inputs:
            preprocessed_images_list.append(self._preprocess_one_image(image))
        return {self.input_name: np.concatenate(preprocessed_images_list, axis=0)}

    def _indices_to_label(self, indices: NDArray[np.int_]) -> str:
        pre_c, no_repeat_blank_label = "", []
        for c in indices:  # dropout repeated label and blank label
            if (pre_c == c) or (c == len(self.chars) - 1):
                if c == len(self.chars) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        # Add filtering by regexpr
        # re.findall(r"[А-Я]{1}\d{3}[А-Я]{2}\d{2,3}", final_label)
        return "".join([self.chars[l] for l in no_repeat_blank_label])

    def _post_process_one(self, preds: NDArray[np.float32]) -> str:
        """Postprocess one image prediction and get licence plate text.

        Parameters
        ----------
        preds : NDArray[np.float32]
            array of shape (23, 19)

        Returns
        -------
        str
            predicted licence plate.
        """
        max_prob_indices = np.argmax(preds, axis=0)
        return self._indices_to_label(max_prob_indices)

    def postprocess(self, raw_outputs: Dict[str, NDArray[np.float32]]) -> List[str]:
        """Postprocess raw outputs from triton model.

        Postprocess raw outputs from triton model
        and convert them into labels.

        Parameters
        ----------
        raw_outputs : Dict[str, NDArray[np.float32]]
            raw outupts from triton model.
            Typically of size (n, 23, 19).

        Returns
        -------
        List[str]
            list of size n with corresponding
            licence plate labels.
        """
        output = raw_outputs[self.output_name]
        labels = []
        for i in range(output.shape[0]):
            labels.append(self._post_process_one(output[i, ...]))
        return labels
