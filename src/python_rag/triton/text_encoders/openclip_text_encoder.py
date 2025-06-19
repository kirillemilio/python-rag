"""Contains implementation of openclip text encoder triton model."""

from __future__ import annotations

from transformers import AutoTokenizer

from ..models_factory import TritonModelFactory
from .base_text_encoder import BaseTextEncoderTritonModel


@TritonModelFactory.register_model(model_type="text-encoder", arch_type="clip-text")
class OpenCLIPTextEncoderTritonModel(BaseTextEncoderTritonModel):
    """
    Implement a text encoder for OpenCLIP models from the LAION family.

    Loads the tokenizer for the 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    checkpoint. This tokenizer is compatible with Hugging Face Transformers
    and can be used with OpenCLIP inference in Triton.

    Methods
    -------
    create_tokenizer()
        Load the Hugging Face tokenizer for the OpenCLIP model.
    """

    def create_tokenizer(self):
        """
        Load the Hugging Face tokenizer for OpenCLIP (ViT-H-14).

        Returns
        -------
        AutoTokenizer
            Tokenizer for 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'.
        """
        return AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
