"""Contains implementation of mpnet text encoder triton model."""

from __future__ import annotations

from transformers import AutoTokenizer

from ..models_factory import TritonModelFactory
from .base_text_encoder import BaseTextEncoderTritonModel


@TritonModelFactory.register_model(model_type="text-encoder", arch_type="mpnet")
class MPNetTextEncoderTritonModel(BaseTextEncoderTritonModel):
    """
    Implement a text encoder using the MPNet transformer model.

    This class loads the pretrained 'all-mpnet-base-v2' model from the
    Sentence-Transformers library and provides a tokenizer compatible with
    the underlying Triton encoder interface.

    Methods
    -------
    create_tokenizer()
        Load and return the Hugging Face tokenizer for MPNet.
    """

    def create_tokenizer(self):
        """
        Load a Hugging Face tokenizer for the MPNet model.

        Returns
        -------
        AutoTokenizer
            Tokenizer for 'sentence-transformers/all-mpnet-base-v2'.
        """
        return AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
