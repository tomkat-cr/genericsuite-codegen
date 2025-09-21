#!/usr/bin/env python3
"""
Startup script for the GenericSuite CodeGen MCP Server.

This script provides a convenient way to start the MCP server with
proper environment setup and error handling.
"""

import sys
import asyncio
from pathlib import Path

from genericsuite_codegen.mcp_server import (
    create_mcp_server,
    load_environment,
    validate_environment,
    get_mcp_config,
    report_mcp_config,
    print_output,
    logger,
)

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add the server directory to Python path for imports
server_dir = current_dir.parent / "server"
sys.path.insert(0, str(server_dir))


def main():
    """Main entry point for the MCP server startup script."""
    try:
        logger.info("=" * 60)
        logger.info("GenericSuite CodeGen MCP Server Startup")
        logger.info("=" * 60)

        # Load environment
        load_environment(current_dir)

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
        load_environment(current_dir)

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
        print_output("Running MCP server in normal mode !!!")
        main()
