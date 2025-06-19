"""Contains implementation of models pool class."""

from __future__ import annotations

from typing import Any, Dict, List, Type, TypeVar

from tabulate import tabulate  # type: ignore

from .models_factory import TritonModelFactory
from .triton_model import BaseTritonModel

T = TypeVar("T", bound=BaseTritonModel)


class TritonModelsPool:
    """
    A pool to manage and access multiple models from Triton inference server.

    Attributes
    ----------
    models : Dict[str, BaseTritonModel]
        A dictionary mapping model names to their instances.
    model_factory : TritonModelFactory
        The factory used for creating model instances.

    Methods
    -------
    __init__(models, model_factory)
        Initializes the pool with a set of models and a factory for model creation.
    from_config(config, model_factory)
        Creates an instance of TritonModelsPool from a configuration object.
    get_factory()
        Returns the model factory associated with this pool.
    get_model(model_name, model_type)
        Retrieves a model by name and ensures it is of the specified type.
    get_validation_report()
        Generates a validation report for all models in the pool.
    show_report()
        Displays a formatted validation report.

    Examples
    --------
    >>> model_configs = [
                {
                    "model_type": "detector",
                    "arch_type": "yolov7",
                    "name": "det-gf-shop-v7",
                    "input_name": "images",
                    "output_name": "output",
                    "input_size": {"h": 320, "w": 640},
                    "iou_threshold": {"default": 0.1, "name": "iou_threshold"},
                    "conf_threshold": {"default": 0.1, "name": "conf_threshold"},
                },
                # Additional models...
        ])
    >>> triton_config = TritonConfig(host="0.0.0.0", port=8001, client_timeout=1)
    >>> model_factory = TritonModelFactory(triton_config)
    >>> models_pool = TritonModelsPool.from_config(models_configs, model_factory=model_factory)
    >>> print(models_pool.get_validation_report())
    """

    models: Dict[str, BaseTritonModel]
    model_factory: TritonModelFactory

    def __init__(self, models: Dict[str, BaseTritonModel], model_factory: TritonModelFactory):
        """Initialize the models pool with a dictionary of models and a model factory.

        Parameters
        ----------
        models : Dict[str, BaseTritonModel]
            A dictionary containing model names as keys and model instances as values.
        model_factory : TritonModelFactory
            The factory used for creating models.
        """
        self.models = dict(models)
        self.model_factory = model_factory

    @classmethod
    def from_config(
        cls, config: List[Dict[str, Any]], model_factory: TritonModelFactory
    ) -> TritonModelsPool:
        """Class method to initialize the models pool from a configuration object.

        Parameters
        ----------
        config : List[Dict[str, Any]]
            Configuration object specifying models to be included in the pool.
        model_factory : TritonModelFactory
            The factory to be used for creating the models.

        Returns
        -------
        TritonModelsPool
            An instance of TritonModelsPool populated with models specified in the config.
        """
        models: Dict[str, BaseTritonModel] = {}
        for model_config_dict in config:
            model = model_factory.create_model(config_dict=model_config_dict)
            models[model.model_name] = model
        return cls(models=models, model_factory=model_factory)

    def get_factory(self) -> TritonModelFactory:
        """Return the model factory associated with this pool.

        Returns
        -------
        TritonModelFactory
            The factory used for creating models in this pool.
        """
        return self.model_factory

    def get_model(self, model_name: str, model_type: Type[T]) -> T:
        """Retrieve a model by name and ensures it is of the specified type.

        Parameters
        ----------
        model_name : str
            The name of the model to retrieve.
        model_type : Type[T]
            The expected type of the model.

        Returns
        -------
        T
            The model instance of the specified type.

        Raises
        ------
        KeyError
            If the model name is not found in the pool.
        TypeError
            If the retrieved model is not of the expected type.
        """
        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise KeyError(
                f"Model with name `{model_name}` not found in pool. "
                f"Available models are: {available_models}"
            )
        model = self.models[model_name]
        if not isinstance(model, model_type):
            raise TypeError(
                f"Model with name `{model_name}` has type `{type(model).__name__}` "
                f"which is different from target type `{model_type.__name__}`."
            )

        return model

    def get_validation_report(self) -> List[Dict[str, str]]:
        """Generate a validation report for all models in the pool.

        Returns
        -------
        List[Dict[str, str]]
            A list of dictionaries containing the model name, validation status, and message.
        """
        report = []
        for model_name, model in self.models.items():
            status, message = model.validate_model()
            report.append({"model_name": model_name, "status": status.name, "message": message})
        return report

    def show_report(self) -> str:
        """Display a formatted validation report using tabulate for better readability.

        Returns
        -------
        str
            A string representing the formatted table of the validation report.
        """
        report = self.get_validation_report()

        green = "\033[92m"  # Green text
        red = "\033[91m"  # Red text
        reset = "\033[0m"  # Reset to default terminal color

        colored_data = []
        for row in report:
            colored_row = row.copy()
            status = colored_row["status"]
            if status == "SUCCESS":
                colored_row["status"] = f"{green}✔{reset}"
            else:
                colored_row["status"] = f"{red}✖{reset}"
            colored_row["message"] = row["message"]
            colored_data.append(colored_row)

        headers = {"model_name": "Model Name", "status": "Status", "message": "Message"}
        table = tabulate(colored_data, headers, tablefmt="fancy_grid")
        return table

    def cleanup(self) -> None:
        """Cleanup models resources."""
        for _, model in self.models.items():
            model.cleanup()
