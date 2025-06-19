"""Contains imports of text recognizer models."""

from .lprnet_text_recongnizer import LPRNetTextRecognizerTritonModel
from .text_recognizer import TextRecognizerTritonModel

__all__ = ["TextRecognizerTritonModel", "LPRNetTextRecognizerTritonModel"]
