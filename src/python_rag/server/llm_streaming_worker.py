"""Contains implementation of llm streaming worker service."""

from __future__ import annotations

import argparse
import logging

from ..config.streaming_worker_config import StreamingWorkerConfig
from ..streaming.streaming_worker import StreamingWorkerBuilder
from .utils import ConfigLoader

logging.basicConfig(level=logging.DEBUG)


def run_streaming_worker(config_path: str) -> None:
    """Run streaming worker using configuration path.

    Parameters
    ----------
    config_path : str
        configuration file path.
    """
    config = ConfigLoader.load_config(config_path, config_class=StreamingWorkerConfig)
    streaming_worker = (
        StreamingWorkerBuilder()
        .with_llm_config(config.llm)
        .with_streaming_backend_config(config.streaming_backend)
        .with_redis_config(config.redis)
        .get_streaming_worker(
            worker_type=config.worker_type, sampling_temperature=config.sampling_temperature
        )
    )
    streaming_worker.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run llm streaming worker')
    parser.add_argument('--config', type=str, help='configuration file path')
    args = parser.parse_args()
    run_streaming_worker(config_path=args.config)
