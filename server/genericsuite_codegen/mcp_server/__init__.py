"""
FastMCP Server for GenericSuite CodeGen.

This module provides MCP (Model Context Protocol) server integration
for the GenericSuite CodeGen system, exposing AI agent capabilities
as standardized MCP tools and resources.
"""

from .server import create_mcp_server, MCPConfig

__all__ = ["create_mcp_server", "MCPConfig"]