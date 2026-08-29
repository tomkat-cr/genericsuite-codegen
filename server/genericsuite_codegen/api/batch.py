"""
Batch server for GenericSuite CodeGen.

This module sets up the main FastAPI application with CORS, middleware,
and all API endpoints for the GenericSuite CodeGen RAG system.
"""
from typing import Union
import sys
import asyncio
from pathlib import Path

from genericsuite_codegen.api.types import (
    StandardGsResponse,
    StandardGsErrorResponse,
)
from genericsuite_codegen.api.main import success_wrapper

from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    set_app_logs,
)
from genericsuite_codegen.utilities.env_vars import get_envvar


DEBUG = get_envvar("BATCH_SERVER_DEBUG", "0") == "1"


current_dir = Path(__file__).parent
log_dir = get_envvar("SERVER_LOGS_DIR", f"{current_dir}/../..")


def result_wrapper(
    result: Union[StandardGsResponse, StandardGsErrorResponse],
) -> StandardGsResponse:
    """
    Wrap the result in a error or success response.

    Args:
        result: The result to wrap. It must have the following attributes:
            - error: bool
            - error_message: str
            - status_code: int
            - result: any

    Returns:
        The wrapped result.
    """
    if result.error:
        raise Exception(
            result.error_message +
            " | Status code: " +
            str(result.status_code)
        )
    return success_wrapper(result.result)


async def execute_route(option: str) -> None:
    """
    Execute a route based on the option.

    Args:
        option: The option to execute.
    """
    from .endpoint_methods import get_endpoint_methods
    methods = get_endpoint_methods()

    if option == "update-knowledge-base":
        """
        Trigger knowledge base update.

        Args:
            request: Update request parameters.

        Returns:
            Dict[str, str]: Update initiation response.
        """
        return result_wrapper(
            await methods.update_knowledge_base()
        )
    else:
        raise Exception("Invalid option: " + option)


def run_server():
    """
    Run the batch server
    """
    if len(sys.argv) < 2:
        raise Exception("No option provided")
    option = sys.argv[1]
    set_app_logs(
        name="batchserver",
        log_file=f"{log_dir}/batch_server.log",
        debug=DEBUG
    )
    _ = DEBUG and log_debug(f"Running batch server with option: '{option}'")
    asyncio.run(execute_route(option))
    _ = DEBUG and log_debug("Batch server finished")


if __name__ == "__main__":
    run_server()
