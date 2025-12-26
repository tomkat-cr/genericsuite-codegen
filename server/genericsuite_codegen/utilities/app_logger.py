"""
Logging utilities
"""

from typing import Any, Union
import os
import sys
import logging
import datetime

DEBUG = True

app_logs: Union[logging.Logger, None] = None


def is_local_service() -> bool:
    """Check if the service is running locally"""
    return os.getenv("APP_ENV", "development") == "development"


def log_config(
    log_file: str = None,
    name: str = None,
    debug: bool = DEBUG
) -> logging.Logger:
    """Logging configuration"""
    logger_options = os.getenv("LOGGER_OPTIONS", "")
    logger = logging.getLogger(name if name else "")
    logger.propagate = False
    if debug:
        logger.setLevel(logging.DEBUG)
        if "silent" not in logger_options:
            print("Logger configured in DEBUG mode")
    else:
        logger.setLevel(logging.INFO)
        if "silent" not in logger_options:
            print("Logger configured in INFO mode")
    handler = logging.StreamHandler(sys.stdout)
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        ("%(name)s-" if name else "")
        + "%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    # Reduce noise from external libraries in production
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("selectors").setLevel(logging.WARNING)
    logging.getLogger("gitpython").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)

    return logger


def set_app_logs(
    log_file: str = None,
    name: str = None,
    debug: bool = DEBUG
) -> None:
    global app_logs
    app_logs = log_config(log_file, name, debug)


def _get_logger() -> logging.Logger:
    """Gets the application logger, initializing it if necessary."""
    global app_logs
    if not app_logs:
        set_app_logs()
    return app_logs


def db_stamp() -> str:
    db_engine = os.environ.get("APP_DB_ENGINE", "MONGODB")
    if db_engine == "DYNAMODB":
        response = f"{db_engine}|" + \
            f"{os.environ.get('DYNAMDB_PREFIX', 'No-Prefix')}"
    else:
        response = f"{db_engine}|{os.environ.get('APP_DB_NAME')}"
    if is_local_service():
        response += "|LOCAL"
    else:
        response += "|CLOUD"
    return response


def formatted_message(message: Any) -> str:
    """Returns a formatted message with database name and date/time"""
    return (
        f"[{db_stamp()}]"
        + f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        + f" | {message}"
    )


def log_debug(message: Any) -> str:
    """Register a Debug log"""
    fmt_msg = formatted_message(message)
    _get_logger().debug("%s", fmt_msg)
    return fmt_msg


def log_info(message: Any) -> str:
    """Register an Info log"""
    fmt_msg = formatted_message(message)
    _get_logger().info("%s", fmt_msg)
    return fmt_msg


def log_warning(message: Any) -> str:
    """Register a Warning log"""
    fmt_msg = formatted_message(message)
    _get_logger().warning("%s", fmt_msg)
    return fmt_msg


def log_error(message: Any) -> str:
    """Register an Error log"""
    fmt_msg = formatted_message(message)
    _get_logger().error("%s", fmt_msg)
    return fmt_msg
