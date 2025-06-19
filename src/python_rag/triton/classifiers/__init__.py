from .age_classifier import AgeV5ClassifierTritonModel, AgeV8ClassifierTritonModel
from .car_classifier import CarClassifierTritonModel
from .classifier import ClassifierTritonModel
from .gender_classifer import GenderV5ClassifierTritonModel, GenderV8ClassifierTritonModel

__all__ = [
    "ClassifierTritonModel",
    "AgeV5ClassifierTritonModel",
    "AgeV8ClassifierTritonModel",
    "GenderV5ClassifierTritonModel",
    "GenderV8ClassifierTritonModel",
    "CarClassifierTritonModel",
]
