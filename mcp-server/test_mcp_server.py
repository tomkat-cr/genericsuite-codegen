#!/usr/bin/env python3
"""
Test script for the GenericSuite CodeGen MCP Server.

This script tests the MCP server functionality without requiring
a full MCP client setup.
"""

from genericsuite_codegen.mcp_server import create_mcp_server, MCPConfig
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add the server directory to Python path for imports
server_dir = current_dir.parent / "server"
sys.path.insert(0, str(server_dir))

# Ensure we can import from the mcp-server directory
sys.path.insert(0, str(current_dir))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_mcp_server():
    """Test the MCP server functionality."""
    try:
        logger.info("Testing GenericSuite CodeGen MCP Server...")

        # Create test configuration
        config = MCPConfig(
            server_name="test-genericsuite-codegen",
            server_version="1.0.0-test",
            host="127.0.0.1",
            port=8071,  # Use different port for testing
            debug=True
        )

        # Create MCP server
        mcp_server = create_mcp_server(config)

        logger.info("MCP server created successfully")

        # Test server initialization
        if hasattr(mcp_server, 'mcp'):
            logger.info("FastMCP instance initialized")

        if mcp_server.agent:
            logger.info("AI agent available")

            # Test agent health check
            health = await mcp_server.agent.health_check()
            logger.info(f"Agent health: {health['status']}")
        else:
            logger.warning("AI agent not available")

        if mcp_server.kb_tool:
            logger.info("Knowledge base tool available")
        else:
            logger.warning("Knowledge base tool not available")

        logger.info("MCP server test completed successfully")
        return True

    except Exception as e:
        logger.error(f"MCP server test failed: {e}")
        return False


async def test_mcp_tools():
    """Test individual MCP tools."""
    try:
        logger.info("Testing MCP tools...")

        # Create MCP server
        config = MCPConfig(debug=True)
        mcp_server = create_mcp_server(config)

        # Test knowledge base search (if available)
        if mcp_server.kb_tool:
            try:
                from genericsuite_codegen.mcp_server \
                    import KnowledgeBaseSearchRequest

                _ = KnowledgeBaseSearchRequest(
                    query="GenericSuite table configuration",
                    limit=3
                )

                # This would normally be called through MCP protocol
                # For testing, we'll just verify the tool exists
                logger.info("Knowledge base search tool available")

            except Exception as e:
                logger.warning(f"Knowledge base search test failed: {e}")

        # Test agent query (if available)
        if mcp_server.agent:
            try:
                from genericsuite_codegen.agent.agent import QueryRequest

                query_request = QueryRequest(
                    query="What is GenericSuite?",
                    task_type="general"
                )

                # Test query
                response = await mcp_server.agent.query(query_request)
                logger.info(
                    f"Agent query successful: {len(response.content)} chars")

            except Exception as e:
                logger.warning(f"Agent query test failed: {e}")

        logger.info("MCP tools test completed")
        return True

    except Exception as e:
        logger.error(f"MCP tools test failed: {e}")
        return False


def test_configuration():
    """Test MCP server configuration."""
    try:
        logger.info("Testing MCP server configuration...")

        # Test default configuration
        default_config = MCPConfig()
        logger.info(f"Default config - Name: {default_config.server_name}")
        logger.info(f"Default config - Port: {default_config.port}")

        # Test environment-based configuration
        os.environ["MCP_SERVER_NAME"] = "test-server"
        os.environ["MCP_SERVER_PORT"] = "9999"

        env_config = MCPConfig(
            server_name=os.getenv("MCP_SERVER_NAME", "default"),
            port=int(os.getenv("MCP_SERVER_PORT", "8070"))
        )

        logger.info(f"Env config - Name: {env_config.server_name}")
        logger.info(f"Env config - Port: {env_config.port}")

        # Clean up environment
        del os.environ["MCP_SERVER_NAME"]
        del os.environ["MCP_SERVER_PORT"]

        logger.info("Configuration test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("GenericSuite CodeGen MCP Server Test Suite")
    logger.info("=" * 60)

    tests = [
        ("Configuration", test_configuration),
        ("MCP Server", test_mcp_server),
        ("MCP Tools", test_mcp_tools)
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            results.append((test_name, result))
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name} test: {status}")

        except Exception as e:
            logger.error(f"{test_name} test error: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary:")
    logger.info("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("All tests passed! ✅")
        return 0
    else:
        logger.error("Some tests failed! ❌")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)
