"""Contains Pydantic configuration model for Qdrant connection."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field
from qdrant_client.http.models import Distance


class QdrantCollectionConfig(BaseModel):
    """
    Configuration model for defining and creating a Qdrant vector collection.

    Attributes
    ----------
    name : str
        Unique name of the collection.

    vector_size : int
        Dimensionality of the vector embeddings.

    distance : Literal["cosine", "euclidean", "dot"]
        Distance metric used to compute similarity. Defaults to "cosine".

    hnsw_m : int
        Number of bi-directional links created for each new element in the HNSW graph.
        Higher means better recall. Defaults to 16.

    ef_construct : int
        Controls recall/accuracy during index construction.
        Larger values increase indexing time but improve search accuracy. Defaults to 100.

    on_disk : bool
        Whether to store vectors on disk (slower access, less RAM usage). Defaults to True.

    full_scan_threshold : int
        full scan threshold to apply.

    compression : Litera["x4", "x8", "x16", "x32", "x64"] | None
        compression rate for pq quantization. Default is None meaning that
        no compression will be applied.
    """

    name: str = Field(..., description="Collection name")
    vector_size: int = Field(..., description="Embedding dimensionality")
    distance: Literal["cosine", "euclidean", "dot"] = Field(default="cosine")
    hnsw_m: int = Field(default=16)
    ef_construct: int = Field(default=100)
    on_disk: bool = Field(default=True)
    full_scan_threshold: int = 10_000
    compression: Literal["x4", "x8", "x16", "x32", "x64"] | None = None

    def get_distance_enum(self) -> Distance:
        """
        Convert string distance metric into Qdrant Distance enum.

        Returns
        -------
        Distance
            Enum value compatible with Qdrant API.
        """
        return {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }[self.distance]


class QdrantConfig(BaseModel):
    """
    Configuration for connecting to a Qdrant instance.

    Attributes
    ----------
    host : str
        Hostname or IP address of the Qdrant server.
        Defaults to 'localhost' for local development.

    use_grpc : bool
        Whether to use gRPC instead of HTTP for communication.
        Defaults to False (HTTP).

    use_secure : bool
        Whether to use secure http protocol (https) for communication.
        Defaults to False.

    http_port : int
        Port to use if `use_grpc` is False. Defaults to 6333.

    grpc_port : int
        Port to use if `use_grpc` is True. Defaults to 6334.

    timeout : int | None
        timeout for connection. Default is None
        meaning that default qdrant timeout is used.

    api_key : Optional[str]
        API key for Qdrant Cloud. Required if connecting to a protected cloud instance.

    prefer_cloud : bool
        If True, configures host for Qdrant Cloud deployment pattern.
        Defaults to False.
    """

    host: str = Field(default="localhost")
    use_grpc: bool = Field(default=False)
    use_secure: bool = Field(default=True)
    http_port: int = Field(default=6333)
    grpc_port: int = Field(default=6334)

    timeout: int | None = None

    api_key: Optional[str] = Field(default=None)
    prefer_cloud: bool = Field(default=False)

    @property
    def url(self) -> str:
        """
        Construct connection URL based on transport protocol.

        Returns
        -------
        str
            Full connection string (e.g., http://host:port or grpc://host:port)
        """
        scheme = "grpc" if self.use_grpc else "http"
        port = self.grpc_port if self.use_grpc else self.http_port
        return f"{scheme}://{self.host}:{port}"
