"""Contains implementation of metrics holder."""

from __future__ import annotations

import logging
from typing import ClassVar, Dict, Generic, TypeVar

from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client.metrics import MetricWrapperBase

from ..config import EnvSettings

LOGGER = logging.getLogger("python-rag")


T = TypeVar("T", bound=MetricWrapperBase)


class MetricsHolder(Generic[T]):
    """Holds Prometheus metrics with environment and service labels.

    Attributes
    ----------
    env : str
        The environment label.
    service : str
        The service label.
    metric : T
        The Prometheus metric instance.

    Methods
    -------
    get_hist(env: str, service: str) -> MetricsHolder[Histogram]:
        Create a histogram metric holder.
    get_summary(env: str, service: str) -> MetricsHolder[Summary]:
        Create a summary metric holder.
    get_counter(env: str, service: str) -> MetricsHolder[Counter]:
        Create a counter metric holder.
    get_gauge(env: str, service: str) -> MetricsHolder[Gauge]:
        Create a gauge metric holder.
    get_default_hist() -> MetricsHolder[Histogram]:
        Get a default histogram metric holder.
    get_default_summary() -> MetricsHolder[Summary]:
        Get a default summary metric holder.
    get_default_counter() -> MetricsHolder[Counter]:
        Get a default counter metric holder.
    get_default_gauge() -> MetricsHolder[Gauge]:
        Get a default gauge metric holder.
    with_method(subsystem: str, method: str) -> T:
        Create a metric with specified labels.
    """

    env: str
    service: str
    metric: T

    _hists: ClassVar[Dict[str, Histogram]] = {}
    _summaries: ClassVar[Dict[str, Summary]] = {}
    _counters: ClassVar[Dict[str, Counter]] = {}
    _gauges: ClassVar[Dict[str, Gauge]] = {}

    def __init__(self, metric: T, env: str, service: str):
        """
        Initialize MetricsHolder with metric, env, and service.

        Parameters
        ----------
        metric : T
            The Prometheus metric instance.
        env : str
            The environment label.
        service : str
            The service label.
        """
        self.env = env
        self.service = service
        self.metric = metric

    @classmethod
    def get_hist(cls, env: str, service: str) -> MetricsHolder[Histogram]:
        """Create a histogram metric holder.

        Parameters
        ----------
        env : str
            The environment label.
        service : str
            The service label.

        Returns
        -------
        MetricsHolder[Histogram]
            A holder for the histogram metric.
        """
        if "rag_hist" not in cls._hists:
            hist = Histogram(
                name="rag_hist",
                labelnames=["env", "service", "subsystem", "method"],
                documentation="python rag histogram",
            )
            cls._hists["rag_hist"] = hist
        return MetricsHolder(cls._hists["rag_hist"], env=env, service=service)

    @classmethod
    def get_summary(cls, env: str, service: str) -> MetricsHolder[Summary]:
        """Create a summary metric holder.

        Parameters
        ----------
        env : str
            The environment label.
        service : str
            The service label.

        Returns
        -------
        MetricsHolder[Summary]
            A holder for the summary metric.
        """
        if "rag_summary" not in cls._summaries:
            summary = Summary(
                name="rag_sum",
                labelnames=["env", "service", "subsystem", "method"],
                documentation="python rag summary",
            )
            cls._summaries["rag_summary"] = summary
        return MetricsHolder(cls._summaries["rag_summary"], env=env, service=service)

    @classmethod
    def get_counter(cls, env: str, service: str) -> MetricsHolder[Counter]:
        """Create a counter metric holder.

        Parameters
        ----------
        env : str
            The environment label.
        service : str
            The service label.

        Returns
        -------
        MetricsHolder[Counter]
            A holder for the counter metric.
        """
        if "rag_counter" not in cls._counters:
            counter = Counter(
                name="rag_counter",
                labelnames=["env", "service", "subsystem", "method"],
                documentation="python rag counter",
            )
            cls._counters["rag_counter"] = counter
        return MetricsHolder(cls._counters["rag_counter"], env=env, service=service)

    @classmethod
    def get_gauge(cls, env: str, service: str) -> MetricsHolder[Gauge]:
        """Create a gauge metric holder.

        Parameters
        ----------
        env : str
            The environment label.
        service : str
            The service label.

        Returns
        -------
        MetricsHolder[Gauge]
            A holder for the gauge metric.
        """
        if "rag_gauge" not in cls._gauges:
            gauge = Gauge(
                name="rag_gauge",
                labelnames=["env", "service", "subsystem", "method"],
                documentation="python rag gauge",
            )
            cls._gauges["rag_gauge"] = gauge
        return MetricsHolder(cls._gauges["rag_gauge"], env=env, service=service)

    @classmethod
    def get_default_hist(cls) -> MetricsHolder[Histogram]:
        """Get a default histogram metric holder.

        Returns
        -------
        MetricsHolder[Histogram]
            A holder for the default histogram metric.
        """
        settings = EnvSettings()
        return cls.get_hist(env=settings.service_env, service=settings.service_name)

    @classmethod
    def get_default_summary(cls) -> MetricsHolder[Summary]:
        """Get a default summary metric holder.

        Returns
        -------
        MetricsHolder[Summary]
            A holder for the default summary metric.
        """
        settings = EnvSettings()
        return cls.get_summary(env=settings.service_env, service=settings.service_name)

    @classmethod
    def get_default_counter(cls) -> MetricsHolder[Counter]:
        """Get a default counter metric holder.

        Returns
        -------
        MetricsHolder[Counter]
            A holder for the default counter metric.
        """
        settings = EnvSettings()
        return cls.get_counter(env=settings.service_env, service=settings.service_name)

    @classmethod
    def get_default_gauge(cls) -> MetricsHolder[Gauge]:
        """Get a default gauge metric holder.

        Returns
        -------
        MetricsHolder[Gauge]
            A holder for the default gauge metric.
        """
        settings = EnvSettings()
        return cls.get_gauge(env=settings.service_env, service=settings.service_name)

    def with_method(self, subsystem: str, method: str) -> T:
        """Create a metric with specified labels.

        Parameters
        ----------
        subsystem : str
            The subsystem label.
        method : str
            The method label.

        Returns
        -------
        T
            The labeled metric.
        """
        return self.metric.labels(self.env, self.service, subsystem, method)
