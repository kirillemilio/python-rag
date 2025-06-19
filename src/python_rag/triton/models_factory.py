"""Contains implementation of factory with registry for triton models."""

from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore

from ..config.triton import (
    BaseModelConfig,
    ClassifierConfig,
    CropEncoderConfig,
    DetectorConfig,
    ImageEncoderConfig,
    PointsDetectorConfig,
    RoiAlignEncoderConfig,
    SemanticSegmentatorConfig,
    TextEncoderConfig,
    TextRecognizerConfig,
    TritonConfig,
)
from .validation_status import EValidationStatus

if TYPE_CHECKING:
    from .classifiers import ClassifierTritonModel
    from .detectors import DetectorTritonModel
    from .encoders import CropEncoderTritonModel, RoiAlignEncoderTritonModel
    from .image_encoders import BaseImageEncoderTritonModel
    from .points_detectors import PointsDetectorTritonModel
    from .segmentators import SemanticSegmentatorTritonModel
    from .text_encoders import BaseTextEncoderTritonModel
    from .text_recognizers import TextRecognizerTritonModel
    from .triton_model import BaseTritonModel


T = TypeVar("T", bound="BaseTritonModel")


class TritonModelFactoryValidator(type):
    """Metaclass for factory create methods validation."""

    def __new__(mtcls, name: str, bases: Tuple[type], attrs: Dict[str, Any]) -> Any:  # noqa: N804
        """Add validaiton for factory class create methods."""
        found_models_types = set()
        for key, value in attrs.items():
            if key.startswith("create_") and key != "create_model" and callable(value):
                found_models_types.add(key.replace("create_", "").lower())
                signature = inspect.signature(value)
                params = signature.parameters
                if len(params) != 2 or "config_dict" not in params:
                    raise TypeError(
                        f"{key} as builder method must have exactly "
                        + "two params: self and 'config_dict'. "
                        + f" Current implementation requires: {params}"
                    )
                config_param = params["config_dict"]
                if config_param.annotation is config_param.empty:
                    raise TypeError(
                        f"{key} parameter `config_dict` has empty annotation."
                        + f" Must be of subtype of {BaseModelConfig}"
                    )
        instance = super(TritonModelFactoryValidator, mtcls).__new__(mtcls, name, bases, attrs)
        return instance


class TritonModelFactory(metaclass=TritonModelFactoryValidator):
    """A factory class to manage and instantiate specific Triton model instances based on type.

    This class supports the dynamic registration and creation of models like classifiers,
    detectors, and encoders, facilitating easy integration and management of diverse
    models in a Triton inference server setup.

    Attributes
    ----------
    _models : Dict[str, Type[BaseTritonModel]]
        A class variable that maps model names to their corresponding classes.
    triton_config : TritonConfig
        Configuration settings for the Triton server.
    """

    _models: ClassVar[Dict[str, Type[BaseTritonModel]]] = {}

    triton_config: TritonConfig
    client: grpcclient.InferenceServerClient

    def __init__(self, triton_config: TritonConfig):
        """Initialize the TritonModelFactory with configurations for Triton server and models.

        Parameters
        ----------
        triton_config : TritonConfig
            The configuration settings for the Triton inference server.
        """
        self.triton_config = triton_config
        self.client = grpcclient.InferenceServerClient(
            f"{self.triton_config.host}:{self.triton_config.port}",
            verbose=triton_config.verbose,
            ssl=triton_config.ssl,
            root_certificates=triton_config.root_certificates,
            private_key=triton_config.private_key,
            certificate_chain=triton_config.private_key,
        )

    @classmethod
    def from_config(cls, triton_config: TritonConfig) -> TritonModelFactory:
        """Create triton model factory from config.

        Parameters
        ----------
        triton_config : TritonConfig
            triton config.

        Returns
        -------
        TritonModelFactory
            model factory instance.
        """
        return cls(triton_config=triton_config)

    @classmethod
    def from_url(cls, url: str) -> TritonModelFactory:
        """Create triton model factory from url.

        Parameters
        ----------
        url : str
            url of triton server in format host:port.

        Returns
        -------
        TritonModelFactory
            triton models factory.
        """
        host, port = url.split(":")
        return cls(triton_config=TritonConfig(host=host, port=int(port)))

    def get_client(self) -> grpcclient.InferenceServerClient:
        """Get grpc triton inference server client.

        Returns
        -------
        grpcclient.InferenceServerClient
            triton inferece server client.
        """
        return self.client

    @classmethod
    def register_model(
        cls,
        model_type: Literal[
            "detector",
            "classifier",
            "text-encoder",
            "image-encoder",
            "crop-encoder",
            "roialign-encoder",
            "points-detector",
            "text-recognizer",
            "semantic-segmentator",
            "instance-segmentator",
        ],
        arch_type: Optional[str] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """Class method decorator for registering model classes under a specific type and name.

        Parameters
        ----------
        model_type : Literal["detector", "classifier",
                             "crop-encoder", "roialign-encoder",
                             "points-detector", "text-recognizer",
                             "semantic-segmentator", "instance-segmentator"]
            The type of model to register.
        arch_type : Optional[str]
            The name under which to register the model class.
            If None, uses the class name in lower case.

        Returns
        -------
        Callable[[Type[BaseTritonModel]], Type[BaseTritonModel]]
            A decorator that registers the model class into the appropriate dictionary.
        """

        def wrapper(model_cls: Type[T]) -> Type[T]:
            nonlocal arch_type
            arch_type = model_cls.__name__.lower() if arch_type is None else arch_type
            model_key = f"{model_type}${arch_type}"
            if model_key in cls._models:
                raise ValueError(
                    f"Model architecture '{arch_type}' is already"
                    + f" registered among '{model_type}' models"
                )
            cls._models[model_key] = model_cls
            return model_cls

        return wrapper

    @property
    def url(self) -> str:
        """Get url for triton.

        Returns
        -------
        str
            get triton url.
        """
        return f"{self.triton_config.host}:{self.triton_config.port}"

    def create_model(self, config_dict: Dict[str, Any], validate: bool = False) -> BaseTritonModel:
        """Create model from registry of arbitrary type.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            base config dict must be parsed into BaseModelConfig
            and target model type config otherwise ValidationError
            will be raised.
        validate : bool
            whether to run validation over model.

        Returns
        -------
        BaseTritonModel
            model.
        """
        config = BaseModelConfig.model_validate(config_dict)
        model_key = f"{config.model_type}${config.arch_type}"
        if model_key not in self._models:
            raise ValueError(
                f"Model type '{config.arch_type}' not registered under '{config.model_type}'"
            )
        create_method_name = f"create_{config.model_type}".replace("-", "_")
        create_method = getattr(self, create_method_name, None)
        if create_method is None:
            raise ValueError(
                f"No create method was found for '{config.model_type}'."
                + f" Consider adding method with name '{create_method_name}'"
            )
        instance: BaseTritonModel = getattr(self, create_method_name).__call__(
            config_dict=config_dict
        )
        if validate:
            status, message = instance.validate_model()
            if status != EValidationStatus.SUCCESS:
                raise ValueError(message)
        return instance

    def create_classifier(self, config_dict: Dict[str, Any]) -> ClassifierTritonModel:
        """Construct an instance of a classifier model based on provided configuration settings.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            The configuration dictionary for the classifier model
            which must be parsed into ClassifierConfig
            which includes all necessary details such as
            model type, input sizes, outputs, and additional settings.

        Returns
        -------
        ClassifierTritonModel
            An instance of a classifier model configured
            according to the provided settings.

        Raises
        ------
        ValueError
            If the classifier type specified in the configuration
            is not registered in the factory.
        ValidationError
            If config parsing from dictionary to ClassifierConfig
            failed.
        """
        from .classifiers import ClassifierTritonModel

        config = ClassifierConfig.model_validate(config_dict)
        model_key = f"{config.model_type}${config.arch_type}"
        if model_key not in self._models:
            raise ValueError(f"Classifier type with name `{config.arch_type}` not found")
        builder_cls: Type[ClassifierTritonModel] = cast(
            Type[ClassifierTritonModel], self._models[model_key]
        )

        client_timeout = self.triton_config.client_timeout
        if config.client_timeout is not None:
            client_timeout = config.client_timeout
        return builder_cls(
            client=self.client,
            model_name=config.name,
            input_name=config.input_name,
            input_size=config.image_size,
            outputs=config.outputs,
            client_timeout=client_timeout,
            compression_algorithm=config.compression_algorithm,
            outputs_map=config.outputs_map,
            model_version=config.version,
            device_id=config.device_id,
            use_cushm=config.use_cushm,
        )

    def create_detector(self, config_dict: Dict[str, Any]) -> DetectorTritonModel:
        """Construct an instance of a detector model based on provided configuration settings.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            The configuration dict for the detector model which includes
            model type, input sizes, outputs, and thresholds for detection.
            This dict must be parsed into DetectorConfig otherwise
            ValidationError will be raised.

        Returns
        -------
        DetectorTritonModel
            An instance of a detector model configured
            according to the provided settings.

        Raises
        ------
        ValueError
            If the detector type specified in the configuration
            is not registered in the factory.
        ValidationError
            If config parsing from dictionary to DetectorConfig
            failed.
        """
        from .detectors import DetectorTritonModel

        config = DetectorConfig.model_validate(config_dict)
        model_key = f"{config.model_type}${config.arch_type}"
        if model_key not in self._models:
            raise ValueError(f"Detector type with name `{config.arch_type}` not found")
        builder_cls: Type[DetectorTritonModel] = cast(
            Type[DetectorTritonModel], self._models[model_key]
        )

        client_timeout = self.triton_config.client_timeout
        if config.client_timeout is not None:
            client_timeout = config.client_timeout

        return builder_cls(
            client=self.client,
            model_name=config.name,
            input_name=config.input_name,
            input_size=config.input_size,
            output_name=config.output_name,
            client_timeout=client_timeout,
            compression_algorithm=config.compression_algorithm,
            conf_threshold_name=config.conf_threshold["name"],
            conf_threshold_default=config.conf_threshold["default"],
            iou_threshold_name=config.iou_threshold["name"],
            iou_threshold_default=config.conf_threshold["default"],
            model_version=config.version,
            use_cushm=config.use_cushm,
            device_id=config.device_id,
        )

    def create_text_recognizer(self, config_dict: Dict[str, Any]) -> TextRecognizerTritonModel:
        """
        Construct an instance of text recognizer model based on provided configuration.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            The configuration dict for the text recognizer model which includes
            model type, input name, output name, input size.
            This dict must be parsed into TextRecognizerConfig otherwise
            ValidationError will be raised.

        Returns
        -------
        TextRecognizerTritonModel
            An instance of a text recognizer triton
            model built according to provided settings.

        Raises
        ------
        ValueError
            If the text recognizer type specified in the configuration
            is not registered in the factory.
        ValidaitonError
            If config parsing for dictionary to TextRecognizerConfig
            failed.
        """
        from .text_recognizers import TextRecognizerTritonModel

        config = TextRecognizerConfig.model_validate(config_dict)
        model_key = f"{config.model_type}${config.arch_type}"
        if model_key not in self._models:
            raise ValueError(f"TextRecongnizesr type with name `{config.arch_type}` not found")
        builder_cls: Type[TextRecognizerTritonModel] = cast(
            Type[TextRecognizerTritonModel], self._models[model_key]
        )

        client_timeout = self.triton_config.client_timeout
        if config.client_timeout is not None:
            client_timeout = config.client_timeout

        return builder_cls(
            client=self.client,
            model_name=config.name,
            input_name=config.input_name,
            input_size=config.input_size,
            output_name=config.output_name,
            client_timeout=client_timeout,
            compression_algorithm=config.compression_algorithm,
            model_version=config.version,
            use_cushm=config.use_cushm,
            device_id=config.device_id,
        )

    def create_points_detector(self, config_dict: Dict[str, Any]) -> PointsDetectorTritonModel:
        """Construct an instance of a points detector model based on provided configuration.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            The configuration dict for the points detector model which includes
            model type, input sizes, outputs, and thresholds for detection.
            This dict must be parsed into PointsDetectorConfig otherwise
            ValidationError will be raised.

        Returns
        -------
        PointsDetectorTritonModel
            An instance of a detector model configured
            according to the provided settings.

        Raises
        ------
        ValueError
            If the points detector type specified in the configuration
            is not registered in the factory.
        ValidationError
            If config parsing from dictionary to PointsDetectorConfig
            failed.
        """
        from .points_detectors import PointsDetectorTritonModel

        config = PointsDetectorConfig.model_validate(config_dict)
        model_key = f"{config.model_type}${config.arch_type}"
        if model_key not in self._models:
            raise ValueError(f"Points detector type with name `{config.arch_type}` not found")
        builder_cls: Type[PointsDetectorTritonModel] = cast(
            Type[PointsDetectorTritonModel],
            self._models[model_key],
        )

        client_timeout = self.triton_config.client_timeout
        if config.client_timeout is not None:
            client_timeout = config.client_timeout

        return builder_cls(
            client=self.client,
            model_name=config.name,
            input_name=config.input_name,
            input_size=config.input_size,
            output_name=config.output_name,
            client_timeout=client_timeout,
            compression_algorithm=config.compression_algorithm,
            iou_threshold=config.iou_threshold,
            conf_threshold=config.conf_threshold,
            model_version=config.version,
            use_cushm=config.use_cushm,
            device_id=config.device_id,
        )

    def create_text_encoder(self, config_dict: Dict[str, Any]) -> BaseTextEncoderTritonModel:
        """
        Construct an instance of a text encoder model from configuration.

        This method parses a configuration dictionary into a validated
        TextEncoderConfig and uses it to instantiate a registered text
        encoder model from the factory.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing the configuration for the text encoder model.
            It must include fields like model type, architecture, input/output
            names, embedding size, and optional client settings.

        Returns
        -------
        BaseTextEncoderTritonModel
            An instance of a text encoder model configured
            according to the provided settings.

        Raises
        ------
        ValueError
            If the specified text encoder architecture is not registered
            in the factory.
        ValidationError
            If the configuration dictionary cannot be parsed into
            a valid TextEncoderConfig.
        """
        from .text_encoders import BaseTextEncoderTritonModel

        config = TextEncoderConfig.model_validate(config_dict)
        model_key = f"{config.model_type}${config.arch_type}"
        if model_key not in self._models:
            raise ValueError(f"Text encoder type with name `{config.arch_type} not found")
        builder_cls: Type[BaseTextEncoderTritonModel] = cast(
            Type[BaseTextEncoderTritonModel], self._models[model_key]
        )

        client_timeout = self.triton_config.client_timeout
        if config.client_timeout is not None:
            client_timeout = config.client_timeout

        return builder_cls(
            client=self.client,
            model_name=config.name,
            text_input_name=config.text_input_name,
            mask_input_name=config.mask_input_name,
            hidden_output_name=config.hidden_output_name,
            embeddings_output_name=config.embeddings_output_name,
            client_timeout=client_timeout,
            compression_algorithm=config.compression_algorithm,
            model_version=config.version,
            embedding_size=config.embedding_size,
            datatype="FP32",
        )

    def create_image_encoder(self, config_dict: Dict[str, Any]) -> BaseImageEncoderTritonModel:
        """
        Construct an instance of an image encoder model from configuration.

        This method parses a configuration dictionary into a validated
        ImageEncoderConfig and uses it to instantiate a registered image
        encoder model from the factory.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing the configuration for the image encoder model.
            It must include fields like model type, architecture, input size,
            input/output names, and embedding dimension.

        Returns
        -------
        BaseImageEncoderTritonModel
            An instance of an image encoder model configured
            according to the provided settings.

        Raises
        ------
        ValueError
            If the specified image encoder architecture is not registered
            in the factory.
        ValidationError
            If the configuration dictionary cannot be parsed into
            a valid ImageEncoderConfig.
        """
        from .image_encoders import BaseImageEncoderTritonModel

        config = ImageEncoderConfig.model_validate(config_dict)
        model_key = f"{config.model_type}${config.arch_type}"
        if model_key not in self._models:
            raise ValueError(f"Image encoder type with name `{config.arch_type} not found")
        builder_cls: Type[BaseImageEncoderTritonModel] = cast(
            Type[BaseImageEncoderTritonModel], self._models[model_key]
        )

        client_timeout = self.triton_config.client_timeout
        if config.client_timeout is not None:
            client_timeout = config.client_timeout

        return builder_cls(
            client=self.client,
            model_name=config.name,
            image_input_name=config.image_input_name,
            input_size=config.input_size,
            hidden_output_name=config.hidden_output_name,
            embeddings_output_name=config.embeddings_output_name,
            client_timeout=client_timeout,
            compression_algorithm=config.compression_algorithm,
            model_version=config.version,
            embedding_size=config.embedding_size,
            datatype="FP32",
        )

    def create_crop_encoder(self, config_dict: Dict[str, Any]) -> CropEncoderTritonModel:
        """Construct an instance of a crop encoder model based on provided configuration settings.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            The configuration dict for the crop encoder model
            which includes model type, input sizes, output name,
            and other necessary details. This dictionary
            must be parsed into CropEncoderConfig
            otherwise ValidationError will be raised.

        Returns
        -------
        CropEncoderTritonModel
            An instance of a crop encoder model configured
            according to the provided settings.

        Raises
        ------
        ValueError
            If the crop encoder type specified in the configuration
            is not registered in the factory.
        ValidationError
            If config parsing from dictionary to CropEncoderConfig
            failed.
        """
        from .encoders import CropEncoderTritonModel

        config = CropEncoderConfig.model_validate(config_dict)
        model_key = f"{config.model_type}${config.arch_type}"
        if model_key not in self._models:
            raise ValueError(f"Crop encoder type with name `{config.arch_type}` not found")

        builder_cls: Type[CropEncoderTritonModel] = cast(
            Type[CropEncoderTritonModel], self._models[model_key]
        )

        client_timeout = self.triton_config.client_timeout
        if config.client_timeout is not None:
            client_timeout = config.client_timeout

        return builder_cls(
            client=self.client,
            model_name=config.name,
            image_input_name=config.image_input_name,
            input_size=config.image_size,
            output_name=config.output_name,
            client_timeout=client_timeout,
            compression_algorithm=config.compression_algorithm,
            model_version=config.version,
            embedding_size=config.embedding_size,
        )

    def create_roialign_encoder(self, config_dict: Dict[str, Any]) -> RoiAlignEncoderTritonModel:
        """
        Construct an instance of a roi align encoder model based on provided configuration settings.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            The configuration dict for the roi align encoder model
            which includes model type, input sizes, output name,
            and additional settings related to ROI alignment.
            This dicionary must be parsed into RoiAlignEncoderConfig
            otherwise ValidationError will be raised.

        Returns
        -------
        RoiAlignEncoderTritonModel
            An instance of a roi align encoder model configured
            according to the provided settings.

        Raises
        ------
        ValueError
            If the roi align encoder type specified in the configuration
            is not registered in the factory.
        ValidationError
            If config parsing from dictionary to RoiAlignEncoderConfig
            failed.
        """
        from .encoders import RoiAlignEncoderTritonModel

        config = RoiAlignEncoderConfig.model_validate(config_dict)
        model_key = f"{config.model_type}${config.arch_type}"
        if model_key not in self._models:
            raise ValueError(f"Roi Align encoder type with name `{config.arch_type}` not found")

        builder_cls: Type[RoiAlignEncoderTritonModel] = cast(
            Type[RoiAlignEncoderTritonModel],
            self._models[model_key],
        )

        client_timeout = self.triton_config.client_timeout
        if config.client_timeout is not None:
            client_timeout = config.client_timeout
        return builder_cls(
            client=self.client,
            model_name=config.name,
            image_input_name=config.image_input_name,
            bbox_input_name=config.bbox_input_name,
            input_size=config.image_size,
            output_name=config.output_name,
            client_timeout=client_timeout,
            compression_algorithm=config.compression_algorithm,
            model_version=config.version,
            device_id=config.device_id,
            use_cushm=config.use_cushm,
            embedding_size=config.embedding_size,
        )

    def create_semantic_segmentator(
        self, config_dict: Dict[str, Any]
    ) -> SemanticSegmentatorTritonModel:
        """Construct an instance of semantic segmentator model based on provided config.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            The configuration dict for the semantic segmentator model
            which includes model type, input size, output name
            and additional settings related to semantic segmentator model.
            This dictionary must be parsed into SemanticSegmentatorConfi
            otherwise ValidationError will be raised.

        Returns
        -------
        SemanticSegmentatorTritonModel
            An instance of semantic segmentator triton model configured
            according to the provided settings.

        Raises
        ------
        ValueError
            If semantic segmentator model with given architecture
            specified in configuration is not registered in the factory.
        ValidationError
            If config parsing from dictionary to SemanticSegmentatorConfig
            failed.
        """
        from .segmentators import SemanticSegmentatorTritonModel

        config = SemanticSegmentatorConfig.model_validate(config_dict)
        model_key = f"{config.model_type}${config.arch_type}"
        if model_key not in self._models:
            raise ValueError(f"Semantic segmentator model with name `{config.arch_type}` not found")

        builder_cls: Type[SemanticSegmentatorTritonModel] = cast(
            Type[SemanticSegmentatorTritonModel], self._models[model_key]
        )

        client_timeout = self.triton_config.client_timeout
        if config.client_timeout is not None:
            client_timeout = config.client_timeout
        return builder_cls(
            client=self.client,
            model_name=config.name,
            input_name=config.input_name,
            output_name=config.output_name,
            input_size=config.input_size,
            labels=config.labels,
            mean=np.array(config.mean, dtype=np.float32),
            std=np.array(config.std, dtype=np.float32),
            client_timeout=client_timeout,
            preprocess_mode=config.preprocess_mode,
            compression_algorithm=config.compression_algorithm,
            model_version=config.version,
            device_id=config.device_id,
            use_cushm=config.use_cushm,
        )
