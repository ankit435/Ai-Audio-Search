"""Structured (JSON) logging configuration — composition-root concern.

Configured once, at process startup (see `src/api/main.py`). Application
services obtain a logger via `logging.getLogger(__name__)` and log
dict-shaped `extra` payloads through `log_event`, which get rendered as
one JSON object per line by `JsonFormatter`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

_RESERVED_LOG_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_ATTRS:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Idempotent logging setup. `level` defaults to env `LOG_LEVEL` or INFO."""
    resolved_level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()

    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Avoid duplicate handlers if called more than once (e.g. in tests).
    if any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def log_event(logger: logging.Logger, level: int, event: str, **context: Any) -> None:
    """Log one structured event with arbitrary key/value context fields."""
    logger.log(level, event, extra=context)
