"""Contains implementation of gender classifier."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..models_factory import TritonModelFactory
from ..transforms import get_transform_sequence
from .classifier import ClassifierTritonModel


@TritonModelFactory.register_model(model_type="classifier", arch_type="gender-v5")
class GenderV5ClassifierTritonModel(ClassifierTritonModel):
    """
    A specialized trition model class for gender classification task.

    Based on yolov5 model.

    Default preprocessing config is (0.485, 0.456, 0.406) as mean,
    (0.229, 0.224, 0.225) as std and crop_mode equal to 'top'.

    See docs of `ClassifierTritonModel` class for details.
    """

    def get_transform(self) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
        """Get transform for classifier.

        Returns
        -------
            Callable[[List[NDArray[np.float32]]], NDArray[np.float32]]
        """
        return get_transform_sequence(
            size=self.input_size,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_mode="top",
        )


@TritonModelFactory.register_model(model_type="classifier", arch_type="gender-v8")
class GenderV8ClassifierTritonModel(ClassifierTritonModel):
    """
    A specialized trition model class for gender classification task.

    Based on yolov8 model.

    Default preprocessing config is (0.0, 0.0, 0.0) as mean,
    (1.0, 1.0, 1.0) as std and crop_mode equal to 'top'.

    See docs of `ClassifierTritonModel` class for details.
    """

    def get_transform(self) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
        """Get transform for classifier.

        Returns
        -------
            Callable[[List[NDArray[np.float32]]], NDArray[np.float32]]
        """
        return get_transform_sequence(
            size=self.input_size,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            crop_mode="top",
        )
