"""Contains implementation of car model classifier."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..models_factory import TritonModelFactory
from ..transforms import get_transform_sequence
from .classifier import ClassifierTritonModel


@TritonModelFactory.register_model(model_type="classifier", arch_type="car")
class CarClassifierTritonModel(ClassifierTritonModel):
    """
    A specialized trition model class for car classification task.

    Default preprocessing config is (0.485, 0.456, 0.406) as mean,
    (0.229, 0.224, 0.225) as std and crop_mode equal to 'bottom'.

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
            crop_mode="bottom",
        )
