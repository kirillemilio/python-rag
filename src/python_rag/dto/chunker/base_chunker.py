"""Contains implementation of base chunker class."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from .chunker_interface import IChunker


class BaseChunker(IChunker):
    """Base chunker class implementation.
    
    Base chunker extends IChunker with
    from_config method that enables construction
    of chunker from raw configuration dictionary
    and being registered in chunker factory. 
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config_dict: dict[str, Any]) -> IChunker:
        """Create chunker from configuration dictionary.
        
        Parameters
        ----------
        config_dict : dict[str, Any] 
            configuration dictionary.
        
        Returns
        -------
        IChunker
            constructed chunker
        """
        raise NotImplementedError()

