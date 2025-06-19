"""Contains implementation of memory manager interface for triton models."""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray


class IMemoryManager(ABC):
    """Memory manager interface for triton models."""

    @abstractmethod
    def init(self, triton_inputs: Dict[str, grpcclient.InferInput]) -> None:
        """Init memory manager for provided triton model inputs.

        Parameters
        ----------
        triton_inputs : Dict[str, grpcclient.InferInput]
            mapping from input name to InferInput.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_inputs(
        self,
        triton_inputs: Dict[str, grpcclient.InferInput],
        data: Dict[str, NDArray[np.float32]],
    ) -> None:
        """Abstract method for setting inputs of triton model.

        Parameters
        ----------
        triton_inputs : Dict[str, grpcclient.InferInput]
            mapping from triton model's input name
            to InferInput object that will be passed
            into triton client's infer method directly.
        data : Dict[str, NDArray[np.float32]]
             mapping from triton model's input name
             to numpy array that will passed as input.
        """
        raise NotImplementedError()

    @abstractmethod
    def cleanup(self) -> None:
        """Abstract method for resources cleanup."""
        raise NotImplementedError()
