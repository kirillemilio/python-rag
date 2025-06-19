"""Contains implementation of text encoder triton model."""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore
from numpy.typing import NDArray
from transformers import AutoTokenizer

from ..triton_model import BaseTritonModel


class BaseTextEncoderTritonModel(BaseTritonModel[str, NDArray[np.float32]]):
    """Defines a base class for text encoder models deployed on Triton.

    This class provides an interface for handling tokenized text input,
    managing Triton inference, and extracting embeddings from the model's
    outputs.

    Attributes
    ----------
    embedding_size : int
        Size of the embedding vectors produced by the model.
    tokenizer : AutoTokenizer
        Hugging Face tokenizer used for text preprocessing.

    Methods
    -------
    create_tokenizer()
        Create a tokenizer instance (to be implemented by subclass).
    preprocess(inputs)
        Convert text into model input tensors.
    postprocess(raw_outputs)
        Extract and return embeddings from Triton output.
    get_embedding_size()
        Return the dimensionality of the output embeddings.
    """

    embedding_size: int

    text_input_name: str
    mask_input_name: str
    hidden_output_name: str
    embeddings_output_name: str

    tokenizer: AutoTokenizer

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        text_input_name: str,
        mask_input_name: str,
        hidden_output_name: str,
        embeddings_output_name: str,
        client_timeout: float | None,
        embedding_size: int,
        model_version: str = "1",
        device_id: int = 0,
        cushm_inputs: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the base text encoder Triton model.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            Triton Inference Server client instance.
        model_name : str
            Name of the model as registered in the Triton server.
        text_input_name : str
            Name of the input field for token IDs.
        mask_input_name : str
            Name of the input field for attention masks.
        hidden_output_name : str
            Name of the model's hidden state output (optional).
        embeddings_output_name : str
            Name of the output containing final embeddings.
        datatype : str
            Data type of the inputs (e.g., "FP32").
        client_timeout : float or None
            Timeout in seconds for inference requests.
        embedding_size : int
            Size of the embedding vectors produced by the model.
        model_version : str, optional
            Version of the model to use. Default is "1".
        device_id : int, optional
            CUDA device ID for shared memory. Default is 0.
        cushm_inputs : list of str or None, optional
            Input names to use CUDA shared memory for. Default is None.
        **kwargs : dict
            Additional arguments passed to the base Triton model class.
        """
        kwargs.pop("datatype", None)
        super().__init__(
            client=client,
            model_name=model_name,
            inputs={text_input_name: (8,), mask_input_name: (8,)},
            outputs=[hidden_output_name, embeddings_output_name],
            cushm_inputs=cushm_inputs,
            datatype="INT64",
            client_timeout=client_timeout,
            model_version=model_version,
            device_id=device_id,
            allow_spatial_adjustment=True,
            **kwargs,
        )

        self.text_input_name = str(text_input_name)
        self.mask_input_name = str(mask_input_name)
        self.hidden_output_name = str(hidden_output_name)
        self.embeddings_output_name = str(embeddings_output_name)
        self.embedding_size = int(embedding_size)
        self.tokenizer = self.create_tokenizer()

    @abstractmethod
    def create_tokenizer(self) -> AutoTokenizer:
        """
        Create and return a Hugging Face tokenizer.

        This method must be implemented in a subclass to initialize a
        tokenizer appropriate for the deployed model.

        Returns
        -------
        AutoTokenizer
            Hugging Face tokenizer instance.
        """
        raise NotImplementedError()

    def preprocess(self, inputs: List[str]) -> Dict[str, NDArray[np.float32]]:
        """
        Convert raw input text into token IDs and attention masks.

        Parameters
        ----------
        inputs : list of str
            List of input strings to tokenize.

        Returns
        -------
        dict of str to NDArray[np.float32]
            Dictionary containing input tensors ready for inference.
        """
        res = self.tokenizer(inputs, return_tensors="pt", padding=True)  # type: ignore
        return {
            self.text_input_name: res["input_ids"].numpy().astype(np.int64),
            self.mask_input_name: res["attention_mask"].numpy().astype(np.int64),
        }

    def postprocess(self, raw_outputs: Dict[str, NDArray[np.float32]]) -> List[NDArray[np.float32]]:
        """
        Extract embeddings from Triton model output.

        Parameters
        ----------
        raw_outputs : dict of str to NDArray[np.float32]
            Raw output from Triton inference containing embeddings.

        Returns
        -------
        list of NDArray[np.float32]
            List of embedding vectors, one per input sequence.
        """
        x = raw_outputs[self.embeddings_output_name]
        return [x[i, ...] for i in range(x.shape[0])]

    def get_embedding_size(self) -> int:
        """
        Return the size of the output embedding vectors.

        Returns
        -------
        int
            Dimensionality of each output embedding.
        """
        return self.embedding_size
