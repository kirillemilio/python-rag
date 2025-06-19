"""Contains implementation of clip image encoder model."""

from __future__ import annotations

from transformers import AutoImageProcessor

from ..models_factory import TritonModelFactory
from .base_image_encoder import BaseImageEncoderTritonModel


@TritonModelFactory.register_model(model_type="image-encoder", arch_type="clip-image")
class OpenCLIPImageEncoderTritonModel(BaseImageEncoderTritonModel):
    """Implements an image encoder for OpenCLIP models using Hugging Face interface.

    This class uses the pretrained model 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    to initialize an image processor compatible with CLIP-style image encoders.
    It provides preprocessing logic for images prior to Triton inference.

    Methods
    -------
    create_preprocessor()
        Load and return the pretrained Hugging Face image processor.
    """

    def create_preprocessor(self) -> AutoImageProcessor:
        """
        Create and return a Hugging Face image processor.

        Loads the image processor associated with the pretrained
        'laion/CLIP-ViT-H-14-laion2B-s32B-b79K' checkpoint.

        Returns
        -------
        AutoImageProcessor
            Hugging Face image processor instance for OpenCLIP.
        """
        return AutoImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
