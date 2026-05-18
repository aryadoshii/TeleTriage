"""
Logging setup.

Using structlog for structured (JSON) logs because:
  - Every log line becomes queryable post-hoc (grep / jq / DataDog / etc.)
  - Cross-cutting context (request_id, user_id) propagates automatically
  - In dev, we render pretty console output; in prod, JSON.

The choice between dev-pretty and JSON is driven by TELETRIAGE_ENV.
"""
from __future__ import annotations

import logging
import sys

import structlog

from backend.config import get_config


def setup_logging() -> None:
    """Configure logging for the whole app. Call once at startup."""
    cfg = get_config()
    log_level = cfg.observability.log_level.upper()
    is_dev = cfg.secrets.teletriage_env == "development"

    # Standard-library logging -> structlog pipeline
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if is_dev:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
