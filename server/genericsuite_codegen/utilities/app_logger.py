"""
Logging utilities
"""

from typing import Any, Union
import sys
import logging
import datetime
import traceback

from genericsuite_codegen.utilities.env_vars import get_envvar

DEBUG = False

app_logs: Union[logging.Logger, None] = None


def is_local_service() -> bool:
    """Check if the service is running locally"""
    return get_envvar("APP_STAGE", "dev") == "dev"


def log_config(
    log_file: str = None,
    name: str = None,
    debug: bool = DEBUG
) -> logging.Logger:
    """Logging configuration"""
    app_logger_options = get_envvar("APP_LOGGER_OPTIONS", "")
    logger = logging.getLogger(name if name else "")
    logger.propagate = False
    if debug:
        logger.setLevel(logging.DEBUG)
        if "silent" not in app_logger_options:
            print("Logger configured in DEBUG mode")
    else:
        logger.setLevel(logging.INFO)
        if "silent" not in app_logger_options:
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
    logging.getLogger("pypdf").setLevel(logging.WARNING)
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
    db_engine = get_envvar("APP_DB_ENGINE", "MONGODB")
    if db_engine == "DYNAMODB":
        response = f"{db_engine}|" + \
            f"{get_envvar('DYNAMDB_PREFIX', 'No-Prefix')}"
    else:
        response = f"{db_engine}|{get_envvar('APP_DB_NAME')}"
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


def log_info(message: Any, exc_info=False) -> str:
    """Register an Info log"""
    fmt_msg = formatted_message(message)
    log = _get_logger()
    log.info("%s", fmt_msg)
    if exc_info:
        # Log the latest exception traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        log.info(f"Exception type: {exc_type}")
        log.info(f"Exception value: {exc_value}")
        log.info("Traceback details:")
        traceback.print_tb(exc_traceback)
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
