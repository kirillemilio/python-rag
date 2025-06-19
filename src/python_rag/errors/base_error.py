"""Contains implementation of base exception class."""

from typing import ClassVar

from prometheus_client import Counter

from ..monitoring import MetricsHolder


class BaseError(Exception):
    """Base class for all exceptions.

    Attributes
    ----------
    _pr_error_counter : ClassVar[MetricsHolder[Counter]]
        metrics counter holder for errors.
    """

    _pr_error_counter: ClassVar[MetricsHolder[Counter]] = MetricsHolder.get_default_counter()

    def __init__(self, subsystem: str, method: str, message: str = ""):
        self._pr_error_counter.with_method(subsystem=subsystem, method=method).inc()
        super().__init__(message)
