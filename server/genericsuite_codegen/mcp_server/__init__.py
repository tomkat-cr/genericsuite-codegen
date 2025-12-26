"""
FastMCP Server for GenericSuite CodeGen.

This module provides MCP (Model Context Protocol) server integration
for the GenericSuite CodeGen system, exposing AI agent capabilities
as standardized MCP tools and resources.
"""

from .server import (
    create_mcp_server,
    MCPConfig,
    load_environment,
    validate_environment,
    get_mcp_config,
    report_mcp_config,
    print_output,
)

__all__ = [
    "create_mcp_server",
    "MCPConfig",
    "load_environment",
    "validate_environment",
    "get_mcp_config",
    "report_mcp_config",
    "print_output",
]
