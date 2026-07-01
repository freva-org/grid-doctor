"""Shared helpers for compact structured logging in the ERA5-Land workflow."""

import logging


def log_stage(logger: logging.Logger, stage: str, **fields: object) -> None:
    """Emit a compact structured log line for one workflow stage.

    Parameters
    ----------
    logger:
        Logger instance that should receive the message.
    stage:
        Stable stage identifier used by the colored formatter.
    **fields:
        Additional structured key/value pairs appended to the message.
    """

    tokens = [f"stage={stage}"]
    tokens.extend(f"{key}={value}" for key, value in fields.items())
    logger.info(" ".join(tokens))
