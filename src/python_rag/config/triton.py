"""Contains implementation of triton and triton models configs."""

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict
from ruamel.yaml import CommentedMap
from typing_extensions import TypedDict

from ..dto.size import Size


class TritonConfig(BaseModel):
    """
    Configuration for connecting to a Triton Inference Server.

    Attributes
    ----------
    host : str
        The hostname or IP address of the Triton server.
    port : int
        The port number on which Triton server is listening.
    client_timeout : Optional[float], default=None
        Timeout for the client requests in seconds.
    verbose : bool, default=False
        Enables verbose logging if set to True.
    ssl : bool, default=False
        Enable SSL/TLS support.
    root_certificates : Optional[str], default=None
        Path to the root CA certificates for SSL/TLS validation.
    private_key : Optional[str], default=None
        Path to the private key for SSL/TLS.
    certificte_chain : Optional[str], default=None
        Path to the certificate chain for SSL/TLS.
    """

    host: str
    port: int
    client_timeout: Optional[float] = None
    verbose: bool = False
    ssl: bool = False
    root_certificates: Optional[str] = None
    private_key: Optional[str] = None
    certificte_chain: Optional[str] = None

    def format_yaml(self) -> CommentedMap:
        """Format queue config into pretty yaml.

        Returns
        -------
        CommentedMap
            commented map containing queue config.
        """
        formatted_config = CommentedMap()
        defaults = {name: field.default for name, field in self.model_fields.items()}

        # Always include host and port
        formatted_config["host"] = self.host
        formatted_config["port"] = self.port

        # Optional fields
        if self.client_timeout != defaults["client_timeout"]:
            formatted_config["client_timeout"] = self.client_timeout
        if self.verbose != defaults["verbose"]:
            formatted_config["verbose"] = self.verbose
        if self.ssl != defaults["ssl"]:
            formatted_config["ssl"] = self.ssl

        # SSL/TLS specific fields (separated visually)
        if self.root_certificates != defaults["root_certificates"]:
            formatted_config.yaml_set_comment_before_after_key(
                "root_certificates", before="\n"
            )
            formatted_config["root_certificates"] = self.root_certificates
        if self.private_key != defaults["private_key"]:
            formatted_config["private_key"] = self.private_key
        if self.certificte_chain != defaults["certificte_chain"]:
            formatted_config["certificte_chain"] = self.certificte_chain

        return formatted_config
        return formatted_config


class NamedDefaultParameter(TypedDict):
    """
    A dictionary type hint that specifies a model parameter with a default value.

    Attributes
    ----------
    name : str
        The name of the parameter.
    default : float
        The default value of the parameter.
    """

    name: str
    default: float


class BaseModelConfig(BaseModel):
    """
    Base configuration class for all model types in Triton.

    Attributes
    ----------
    model_type : str
        Type of the model (e.g., 'detector', 'classifier').
    arch_type : str
        Architecture type of the model.
    name : str
        Unique name for the model.
    client_timeout : Optional[float], default=None
        Timeout for the client requests in seconds.
    compression_algorithm : Optional[Literal['deflate', 'gzip']], default=None
        Compression algorithm used in model processing.
    device_id : int, default=0
        Device ID to run the model on.
    use_cushm : bool, default=True
        Use CUDA shared memory for model inference.
    version : str, default=""
        Version of the model.
    """

    model_type: str
    arch_type: str

    name: str

    client_timeout: Optional[float] = None
    compression_algorithm: Optional[Literal["deflate", "gzip"]] = None
    device_id: int = 0
    use_cushm: bool = True
    version: str = ""

    model_config = ConfigDict(protected_namespaces=())


class ClassifierConfig(BaseModelConfig):
    """
    Configuration for classifier models.

    Attributes
    ----------
    model_type : Literal['classifier']
        Specific type indication for classifier models.
    input_name : str
        Name of the input tensor.
    outputs : List[str]
        List of output tensors.
    image_size : Size
        Size of the input image.
    outputs_map : Optional[Dict[str, str]], default=None
        Mapping of output tensor names to their types.
    """

    model_type: Literal["classifier"] = "classifier"
    input_name: str
    outputs: List[str]
    image_size: Size

    outputs_map: Optional[Dict[str, str]] = None


class TextEncoderConfig(BaseModelConfig):
    """Configuration for text encoder models.

    Attributes
    ----------
    model_type : Literal["text-encoder"]
        Specifid type indication for text encoder models.
    text_input_name : str
        Name of the text input tensor.
    mask_input_name : str
        Name of text attention mask input tensor.
    hidden_output_name : str
        Name of hidden output tensor.
    embeddings_output_name : str
        Name of embeddings output tensor.
    """

    model_type: Literal["text-encoder"] = "text-encoder"

    text_input_name: str
    mask_input_name: str

    hidden_output_name: str
    embeddings_output_name: str

    embedding_size: int


class ImageEncoderConfig(BaseModelConfig):
    """Configuration for image encoder models.

    Attributes
    ----------
    model_type : Literal["text-encoder"]
        Specifid type indication for text encoder models.
    image_input_name : str
        Name of the image input tensor.
    input_size : Size
        image input size.
    hidden_output_name : str
        Name of hidden output tensor.
    embeddings_output_name : str
        Name of embeddings output tensor.
    """

    model_type: Literal["image-encoder"] = "image-encoder"

    image_input_name: str
    input_size: Size
    hidden_output_name: str
    embeddings_output_name: str

    embedding_size: int


class CropEncoderConfig(BaseModelConfig):
    """
    Configuration for crop-encoder models.

    Attributes
    ----------
    model_type : Literal['crop-encoder']
        Specific type indication for crop-encoder models.
    image_input_name : str
        Name of the image input tensor.
    output_name : str
        Name of the output tensor.
    image_size : Size
        Size of the input image.
    embedding_size : int
        Size of the embedding produced by the model.
    """

    model_type: Literal["crop-encoder"] = "crop-encoder"

    image_input_name: str
    output_name: str

    image_size: Size
    embedding_size: int


class RoiAlignEncoderConfig(BaseModelConfig):
    """
    Configuration for ROI-Align encoder models.

    Attributes
    ----------
    model_type : Literal['roialign-encoder']
        Specific type indication for ROI-Align encoder models.
    image_input_name : str
        Name of the image input tensor.
    bbox_input_name : str
        Name of the bounding box input tensor.
    output_name : str
        Name of the output tensor.
    image_size : Size
        Size of the input image.
    embedding_size : int
        Size of the embedding produced by the model.
    """

    model_type: Literal["roialign-encoder"] = "roialign-encoder"

    image_input_name: str
    bbox_input_name: str
    output_name: str

    image_size: Size
    embedding_size: int


class DetectorConfig(BaseModelConfig):
    """
    Configuration for detector models.

    Attributes
    ----------
    model_type : Literal['detector']
        Specific type indication for detector models.
    input_name : str
        Name of the input tensor.
    output_name : str
        Name of the output tensor.
    input_size : Size
        Size of the input tensor.
    iou_threshold : NamedDefaultParameter
        Default IOU threshold for detection.
    conf_threshold : NamedDefaultParameter
        Default confidence threshold for detection.
    """

    model_type: Literal["detector"] = "detector"

    input_name: str
    output_name: str

    input_size: Size

    iou_threshold: NamedDefaultParameter = {"name": "iou_threshold", "default": 0.1}
    conf_threshold: NamedDefaultParameter = {"name": "conf_threshold", "default": 0.1}


class SemanticSegmentatorConfig(BaseModelConfig):
    """Configuration for semantic segmentator models.

    Attributes
    ----------
    model_type : Literal["semantic-segmentator"]
        Specific type indication for semantic segmentation models.
    input_name : str
        Nmae of the input tensor.
    output_name : str
        Name of the output tensor.
    input_size : Size
        Size of the input tensor.
    labels : List[str]
        list of segmentation labels.
    preprocess_mode : Literal["grey", "color"]
        preprocess mode. Can be "grey" or "color".
        Default is "grey".
    mean : Tuple[float, float, float]
        mean value for each channel for prerpocessing.
        Default is (0.0, 0.0, 0.0).
    std : Tuple[float, float, float]
        std value for each channel for preprocessing.
        Default is (1.0, 1.0, 1.0).
    """

    model_type: Literal["semantic-segmentator"]

    input_name: str
    output_name: str

    input_size: Size

    labels: List[str]

    preprocess_mode: Literal["grey", "color"] = "grey"

    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)


class PointsDetectorConfig(BaseModelConfig):
    """
    Configuration for points detetor models.

    Attributes
    ----------
    model_type : Literal["points-detector"]
        specify type indication points detector models.
    input_name : str
        Name of the input tensor.
    output_name : str
        Name of the output tensor.
    input_size : Size
        Size of the input image.
    iou_threshold : float
        iou threshold for nms.
    conf_threshold : float
        conf threshold for nms.
    """

    model_type: Literal["points-detector"] = "points-detector"

    input_name: str
    output_name: str

    input_size: Size

    iou_threshold: float = 1.0
    conf_threshold: float = 0.0


class TextRecognizerConfig(BaseModelConfig):
    """
    Configuration for text recognizer models.

    Attributes
    ----------
    model_type : Literal["text-recognizer"]
        must be `text-recognizer` for text
        recognition models.
    input_name : str
        input name for image input
        of text recognition model.
    output_name : str
        output name for the text
        recognition model.
    input_size : Size
        size of the image input
        of text recognition model.
    """

    model_type: Literal["text-recognizer"] = "text-recognizer"

    input_name: str
    output_name: str

    input_size: Size
