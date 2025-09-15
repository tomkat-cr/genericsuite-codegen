#!/usr/bin/env python3
"""
Startup script for the GenericSuite CodeGen MCP Server.

This script provides a convenient way to start the MCP server with
proper environment setup and error handling.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

from genericsuite_codegen.mcp_server import create_mcp_server, MCPConfig

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add the server directory to Python path for imports
server_dir = current_dir.parent / "server"
sys.path.insert(0, str(server_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mcp_server.log')
    ]
)

logger = logging.getLogger(__name__)


def load_environment():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv

        # Look for .env file in current directory or parent directories
        env_file = current_dir / ".env"
        if not env_file.exists():
            env_file = current_dir.parent / ".env"

        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
        else:
            logger.warning(
                "No .env file found, using system environment variables")

    except ImportError:
        logger.warning(
            "python-dotenv not available, using system environment variables")


def validate_environment():
    """Validate required environment variables."""
    required_vars = []
    optional_vars = {
        "MCP_SERVER_HOST": "0.0.0.0",
        "MCP_SERVER_PORT": "8070",
        "MCP_API_KEY": None,
        "MCP_DEBUG": "0",
        "MCP_TRANSPORT": "http"
    }

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False

    # Log optional variables
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        logger.info(f"{var}: {value}")

    return True


def get_mcp_config():
    """Get MCP server configuration from environment variables."""
    return MCPConfig(
        server_name=os.getenv("MCP_SERVER_NAME", "genericsuite-codegen"),
        server_version=os.getenv("MCP_SERVER_VERSION", "1.0.0"),
        api_key=os.getenv("MCP_API_KEY"),
        host=os.getenv("MCP_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_SERVER_PORT", "8070")),
        debug=os.getenv("MCP_DEBUG", "0") == "1",
        transport=os.getenv("MCP_TRANSPORT", "http")
    )


def report_mcp_config(config: MCPConfig):
    """Report MCP server configuration."""
    logger.info("Server configuration:")
    logger.info(f"  Name: {config.server_name}")
    logger.info(f"  Version: {config.server_version}")
    logger.info(f"  Host: {config.host}")
    logger.info(f"  Port: {config.port}")
    logger.info(f"  Debug: {config.debug}")
    logger.info(f"  API Key: {'Set' if config.api_key else 'Not set'}")
    logger.info(f"  Transport: {config.transport}")


def main():
    """Main entry point for the MCP server startup script."""
    try:
        logger.info("=" * 60)
        logger.info("GenericSuite CodeGen MCP Server Startup")
        logger.info("=" * 60)

        # Load environment
        load_environment()

        # Validate environment
        if not validate_environment():
            sys.exit(1)

        # Create MCP server configuration
        config = get_mcp_config()
        report_mcp_config(config)

        # Create and start MCP server
        logger.info("Creating MCP server...")
        mcp_server = create_mcp_server(config)

        logger.info("Starting MCP server...")
        mcp_server.run()

    except KeyboardInterrupt:
        logger.info("\nMCP server stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}", exc_info=True)
        sys.exit(1)


async def main_async():
    """Async main entry point for the MCP server."""
    try:
        logger.info("=" * 60)
        logger.info("GenericSuite CodeGen MCP Server Startup (Async)")
        logger.info("=" * 60)

        # Load environment
        load_environment()

        # Validate environment
        if not validate_environment():
            sys.exit(1)

        # Create MCP server configuration
        config = get_mcp_config()
        report_mcp_config(config)

        # Create and start MCP server
        logger.info("Creating MCP server...")
        mcp_server = create_mcp_server(config)

        logger.info("Starting MCP server (async)...")
        await mcp_server.run_async()

    except KeyboardInterrupt:
        logger.info("\nMCP server stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Check if we should run in async mode
    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        asyncio.run(main_async())
    else:
        print("Running MCP server in normal mode !!!")
        main()
