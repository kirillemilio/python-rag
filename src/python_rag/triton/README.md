# Triton Model Subsystem

This repository contains the implementation of a dynamic model loading subsystem for Triton Inference Server. It is designed to simplify the integration and management of different types of models such as classifiers, detectors, and encoders.

## Features

- **Dynamic Model Registration**: Easily register new model types and instances without modifying the core code.
- **Extensible and Configurable**: Support for various model configurations and easy extension to accommodate new model types.
- **Uniform Factory Interface**: Utilizes a factory pattern with a consistent interface for creating model instances, promoting code reuse and maintainability.

## Model Types

- **Classifiers**: Models that classify input data into various categories.
- **Detectors**: Models that detect objects within the input data.
- **Encoders**: Models that encode input data into a different format or representation, typically for feature extraction purposes.

## How to Register a New Model

To register a new model, use the `@TritonModelFactory.register_model` decorator. Specify the `model_type` and an optional `arch_type`. If `arch_type` is not provided, the class name in lower case will be used by default.

### Example

```python
from triton_model_subsystem.models_factory import TritonModelFactory
from triton_model_subsystem.triton_model import BaseTritonModel

@TritonModelFactory.register_model(model_type="detector", arch_type="custom_detector")
class CustomDetector(BaseTritonModel):
    def preprocess(self, inputs):
        # Implementation for preprocessing
        pass

    def postprocess(self, outputs):
        # Implementation for postprocessing
        pass
```


### Extending the Triton Models Factory
If you need to introduce a new type of model that doesn't fit into the existing categories, you can extend the factory as follows:

- *Extend the Base Factory:*
Create a new factory class that inherits from the base factory and add new model storage and registration logic.

```python
class ExtendedTritonModelFactory(BaseTritonModelFactory):
    _new_model_type: ClassVar[Dict[str, Type[BaseTritonModel]]] = {}

    @classmethod
    def register_model(cls, model_type: str, arch_type: Optional[str] = None) -> Callable:
        decorator = super().register_model(model_type, arch_type)
        if model_type == 'new_model_type':
            def new_model_decorator(model_cls: Type[BaseTritonModel]) -> Type[BaseTritonModel]:
                cls._new_model_type[arch_type or model_cls.__name__.lower()] = model_cls
                return model_cls
            return new_model_decorator
        return decorator

```

### Classifier Configuration Example
```python
from multitrack.lib.config.triton import ClassifierConfig

config = ClassifierConfig(
    model_type='classifier',
    arch_type='custom_classifier',
    name='example_classifier',
    input_name='input_0',
    image_size=(224, 224),
    outputs=['output_0']
)
```

### *Building Models*
To build an instance of any registered model, use the factory’s build methods which handle configuration and instantiation automatically:

```python
config = ClassifierConfig(
    name='my_custom_classifier',
    classifier_type='my_custom_classifier',
    input_name='input',
    image_size=Size(w=224, h=224),
    outputs=['output'],
    client_timeout=1000
)

model_instance = TritonModelFactory.create_classifier(config)
```