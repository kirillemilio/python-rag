"""
Contains Pydantic configuration model for PostgreSQL connection.

Supports both local and cloud environments with optional SSL.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, PostgresDsn, model_validator


class PostgresConfig(BaseModel):
    """
    Configuration for connecting to a PostgreSQL instance.

    Attributes
    ----------
    host : str
        Hostname or IP address of the PostgreSQL server.
        Defaults to 'localhost'.

    port : int
        Port number for PostgreSQL. Defaults to 5432.

    user : str
        Username to authenticate with. Defaults to 'postgres'.

    password : Optional[str]
        Password for the PostgreSQL user. Required for remote/cloud DBs.

    database : str
        Database name to connect to. Defaults to 'postgres'.

    use_ssl : bool
        Whether to enable SSL. Defaults to False.

    dsn : Optional[str]
        Full connection URI, auto-computed if not explicitly provided.
    """

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    user: str = Field(default="postgres")
    password: Optional[str] = None
    database: str = Field(default="postgres")
    use_ssl: bool = Field(default=False)
    application_name: Optional[str] = None

    @property
    def dsn(self) -> str:
        """Get full dsn address.

        Returns
        -------
        str
            full dsn address.
        """
        userinfo = f"{self.user}:{self.password}" if self.password else self.user
        sslmode = "require" if self.use_ssl else "prefer"
        query = f"?sslmode={sslmode}"
        return f"postgresql://{userinfo}@{self.host}:{self.port}/{self.database}{query}"

    @model_validator(mode="after")
    def validate_dsn(self):
        """Validate DSN format if manually provided."""
        PostgresDsn.validate(self.dsn)
        return self
