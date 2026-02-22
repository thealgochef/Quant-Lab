"""Structured logging setup for the Alpha Signal Research Lab."""

from __future__ import annotations

import logging
import logging.config

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the entire system."""
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": LOG_FORMAT,
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "alpha_lab": {
                "level": level,
                "handlers": ["console"],
                "propagate": False,
            },
            "alpha_lab.bus": {
                "level": level,
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
    }
    logging.config.dictConfig(config)


def get_agent_logger(agent_id: str) -> logging.Logger:
    """Get a logger bound to a specific agent ID."""
    return logging.getLogger(f"alpha_lab.agent.{agent_id}")
