"""Contains implementation of osnet roi align encoder."""

import tritonclient.grpc as grpcclient  # type: ignore

from ...dto import Size
from ..models_factory import TritonModelFactory
from .roialign_encoder import RoiAlignEncoderTritonModel


@TritonModelFactory.register_model(model_type="roialign-encoder", arch_type="osnet-roialign")
class OsnetEncoderTritonModel(RoiAlignEncoderTritonModel):
    """
    An encoder model using OSNet architecture for feature extraction from regions of interest (ROI).

    This model is tailored for scenarios where precise localization and feature
    extraction are critical, such as in person re-identification.
    """

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        image_input_name: str,
        bbox_input_name: str,
        input_size: Size,
        output_name: str,
        client_timeout: float | None,
        embedding_size: int = 512,
        model_version: str = "1",
        device_id: int = 0,
        use_cushm: bool = False,
        **kwargs,
    ):
        """Init osnet roi align encoder.

        An encoder model using OSNet architecture for feature extraction
        from regions of interest (ROI) defined within images. This model
        is tailored for scenarios where precise localization and feature
        extraction are critical, such as in person re-identification.

        Parameters
        ----------
        client : grpcclient.InferenceServerClient
            Triton Inference Server instance.
        model_name : str
            Name of the model on the Triton server.
        image_input_name : str
            The input name for image data.
        bbox_input_name : str
            The input name for bounding box data.
        input_size : Size
            The expected HxW dimensions of the input images.
        output_name : str
            The name of the output tensor for embeddings.
        client_timeout : float | None
            Timeout in seconds for the client to wait for a response.
        embedding_size : int, optional
            The size of the embedding vector, default is 512.
        model_version : str, optional
            The version of the model to use, defaults to "1".
        device_id : int, optional
            The GPU device ID for inference, defaults to 0.
        use_cushm : bool, optional
            Whether to use CUDA shared memory for input/output.
        **kwargs : dict
            Additional arguments to configure the Triton client.
        """
        super().__init__(
            client=client,
            model_name=model_name,
            image_input_name=image_input_name,
            bbox_input_name=bbox_input_name,
            input_size=input_size,
            output_name=output_name,
            client_timeout=client_timeout,
            embedding_size=embedding_size,
            model_version=model_version,
            device_id=device_id,
            use_cushm=use_cushm,
            **kwargs,
        )
