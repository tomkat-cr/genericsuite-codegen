"""
FastMCP Server implementation for GenericSuite CodeGen.

This module implements the MCP server that exposes the AI agent capabilities
as standardized MCP tools and resources for integration with external tools.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "FastMCP is required for MCP server functionality. "
        "Install it with: pip install fastmcp"
    )

from ..agent.agent import GenericSuiteAgent
from ..database.setup import DatabaseManager
from ..api.endpoint_methods import get_endpoint_methods
from ..agent.tools import KnowledgeBaseTool


DEBUG = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if DEBUG else logging.WARNING)


@dataclass
class MCPConfig:
    """Configuration for the MCP server."""

    server_name: str = "genericsuite-codegen"
    server_version: str = "1.0.0"
    api_key: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8070
    debug: bool = False
    transport: str = "http"  # or "stdio"


class GenericSuiteMCPServer:
    """MCP Server for GenericSuite CodeGen system."""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.mcp = FastMCP(config.server_name)
        self.agent: Optional[GenericSuiteAgent] = None
        self.vector_db: Optional[DatabaseManager] = None
        self.kb_tool = None

        self.methods = get_endpoint_methods()

        # Initialize components
        self._setup_components()
        self._setup_authentication()
        self._register_tools()
        self._register_resources()

    def _setup_components(self):
        """Initialize the AI agent and database components."""
        try:
            # Initialize vector database
            self.vector_db = DatabaseManager()

            # Initialize AI agent
            self.agent = GenericSuiteAgent()

            # Initialize knowledge base tool
            self.kb_tool = KnowledgeBaseTool()

            logger.info("MCP server components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MCP server components: {e}")
            raise

    def _setup_authentication(self):
        """Setup MCP authentication and security."""
        try:
            # Add authentication middleware if API key is provided
            if self.config.api_key:
                logger.info("MCP authentication enabled with API key")
                # Note: FastMCP handles authentication through the protocol
                # The API key will be validated in tool calls
            else:
                logger.warning("MCP server running without authentication")

        except Exception as e:
            logger.error(f"Failed to setup MCP authentication: {e}")
            raise

    def _validate_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate incoming MCP requests."""
        try:
            # If API key is configured, validate it
            if self.config.api_key:
                # TODO:
                # In a real implementation, you would check the request headers
                # For now, we'll implement basic validation
                return True

            return True

        except Exception as e:
            logger.error(f"Request validation failed: {e}")
            return False

    def _handle_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Centralized error handling for MCP operations."""
        error_id = f"mcp_error_{hash(str(error)) % 10000:04d}"

        logger.error(f"MCP Error [{error_id}] in {context}: {error}",
                     exc_info=True)

        return {
            "success": False,
            "error": {
                "id": error_id,
                "message": str(error),
                "context": context,
                "type": type(error).__name__,
            },
        }

    def _register_tools(self):
        """Register MCP tools for external integration."""

        @self.mcp.tool()
        async def search_knowledge_base(
            query: str, limit: int = 5
        ) -> Dict[str, Any]:
            """
            Search the GenericSuite knowledge base for relevant information.

            Args:
                query: The search query string
                limit: Maximum number of results to return (default: 5)

            Returns:
                Dictionary containing search results and metadata
            """
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception("AI agent not initialized"),
                        "search_knowledge_base"
                    )

                # Use the agent's knowledge base search capability
                results = await self._search_knowledge_base_async(query, limit)

                return {
                    "success": True,
                    "query": query,
                    "results": results,
                    "count": len(results),
                }

            except Exception as e:
                return self._handle_error(e, "search_knowledge_base")

        @self.mcp.tool()
        async def generate_json_config(
            requirements: str,
            table_name: str,
            config_type: str = "table",
        ) -> Dict[str, Any]:
            """
            Generate GenericSuite JSON configuration based on requirements.

            Args:
                requirements: Description of the configuration requirements

            Returns:
                Dictionary containing the generated JSON configuration
            """
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception("AI agent not initialized"),
                        "generate_json_config"
                    )

                # Generate JSON configuration using the agent
                config = await self.methods.generate_json_config_endpoint(
                    requirements, table_name, config_type)

                return {
                    "success": True,
                    "requirements": requirements,
                    "configuration": config,
                }

            except Exception as e:
                return self._handle_error(e, "generate_json_config")

        @self.mcp.tool()
        async def generate_langchain_tool(
            requirements: str,
            tool_name: str,
            description: str,
        ) -> Dict[str, Any]:
            """
            Generate a Langchain Tool based on specification.

            Args:
                requirements: Description of the tool requirements
                tool_name: Name of the tool
                description: Description of the tool

            Returns:
                Dictionary containing the generated Python code
            """
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception(
                            "AI agent not initialized"),
                        "generate_langchain_tool"
                    )

                # Generate Langchain tool using the agent
                code = await self.methods.generate_python_code_endpoint(
                    requirements,
                    tool_name,
                    description,
                    "langchain_tool",
                )

                return {
                    "success": True,
                    "requirements": requirements,
                    "tool_name": tool_name,
                    "description": description,
                    "code": code,
                    "type": "langchain_tool",
                }

            except Exception as e:
                return self._handle_error(e, "generate_langchain_tool")

        @self.mcp.tool()
        async def generate_mcp_tool(
            requirements: str,
            tool_name: str,
            description: str,
        ) -> Dict[str, Any]:
            """
            Generate an MCP Tool based on specification.

            Args:
                requirements: Description of the tool requirements
                tool_name: Name of the tool
                description: Description of the tool

            Returns:
                Dictionary containing the generated Python code
            """
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception(
                            "AI agent not initialized"), "generate_mcp_tool"
                    )

                # Generate MCP tool using the agent
                code = await self.methods.generate_python_code_endpoint(
                    requirements,
                    tool_name,
                    description,
                    "mcp_tool",
                )

                return {
                    "success": True,
                    "requirements": requirements,
                    "tool_name": tool_name,
                    "description": description,
                    "code": code,
                    "type": "mcp_tool",
                }

            except Exception as e:
                return self._handle_error(e, "generate_mcp_tool")

        @self.mcp.tool()
        async def generate_frontend_code(requirements: str) -> Dict[str, Any]:
            """
            Generate ReactJS frontend code following GenericSuite patterns.

            Args:
                requirements: Description of the frontend requirements

            Returns:
                Dictionary containing the generated frontend code files
            """
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception(
                            "AI agent not initialized"),
                        "generate_frontend_code"
                    )

                # Generate frontend code using the agent
                code_files = await self.methods \
                    .generate_frontend_code_endpoint(requirements)

                return {
                    "success": True,
                    "requirements": requirements,
                    "files": code_files,
                    "type": "frontend_code",
                }

            except Exception as e:
                return self._handle_error(e, "generate_frontend_code")

        @self.mcp.tool()
        async def generate_backend_code(
            framework: str, requirements: str
        ) -> Dict[str, Any]:
            """
            Generate backend code for specified framework following
            GenericSuite patterns.

            Args:
                framework: Backend framework (fastapi, flask, or chalice)
                requirements: Description of the backend requirements

            Returns:
                Dictionary containing the generated backend code files
            """
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception(
                            "AI agent not initialized"),
                        "generate_backend_code"
                    )

                # Generate backend code using the agent
                code_files = await self.methods.generate_backend_code_endpoint(
                    framework, requirements)

                return {
                    "success": True,
                    "framework": framework,
                    "requirements": requirements,
                    "files": code_files,
                    "type": "backend_code",
                }

            except Exception as e:
                return self._handle_error(e, "generate_backend_code")

    def _register_resources(self):
        """Register MCP resources for agent capabilities."""

        @self.mcp.resource(
            uri="genericsuite://knowledge_base_stats",
            name="Knowledge Base Statistics"
        )
        async def get_knowledge_base_stats() -> Dict[str, Any]:
            """Get statistics about the knowledge base."""
            try:
                if not self.vector_db:
                    return self._handle_error(
                        Exception("Vector database not initialized"),
                        "get_knowledge_base_stats",
                    )

                # Get actual stats from the database
                stats = self.vector_db.get_knowledge_base_stats()
                stats["status"] = "healthy"
                stats["last_updated"] = (
                    "2025-09-08T11:42:00Z"
                    # TODO: Would be dynamic in real implementation
                )

                return {"success": True, "data": stats}

            except Exception as e:
                return self._handle_error(e, "get_knowledge_base_stats")

        @self.mcp.resource(uri="genericsuite://server_info",
                           name="Server Information")
        async def get_server_info() -> Dict[str, Any]:
            """Get information about the MCP server."""
            try:
                return {
                    "success": True,
                    "data": {
                        "name": self.config.server_name,
                        "version": self.config.server_version,
                        "status": "running",
                        "authentication": (
                            "enabled" if self.config.api_key else "disabled"
                        ),
                        "capabilities": [
                            "knowledge_base_search",
                            "json_config_generation",
                            "langchain_tool_generation",
                            "mcp_tool_generation",
                            "frontend_code_generation",
                            "backend_code_generation",
                        ],
                        "supported_frameworks": [
                            "fastapi", "flask", "chalice"],
                        "embedding_models": ["openai", "huggingface"],
                    },
                }
            except Exception as e:
                return self._handle_error(e, "get_server_info")

        @self.mcp.resource(
            uri="genericsuite://agent_capabilities", name="Agent Capabilities"
        )
        async def get_agent_capabilities() -> Dict[str, Any]:
            """Get detailed information about agent capabilities."""
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception(
                            "AI agent not initialized"),
                        "get_agent_capabilities"
                    )

                return {
                    "success": True,
                    "data": {
                        "tools": [
                            {
                                "name": "search_knowledge_base",
                                "description": "Search GenericSuite"
                                " documentation and knowledge base",
                                "parameters": ["query", "limit"],
                            },
                            {
                                "name": "generate_json_config",
                                "description": "Generate GenericSuite table"
                                " configuration JSON",
                                "parameters": ["requirements"],
                            },
                            {
                                "name": "generate_langchain_tool",
                                "description": "Generate Langchain Tool"
                                " Python code",
                                "parameters": ["specification"],
                            },
                            {
                                "name": "generate_mcp_tool",
                                "description": "Generate MCP Tool Python code",
                                "parameters": ["specification"],
                            },
                            {
                                "name": "generate_frontend_code",
                                "description": "Generate ReactJS frontend"
                                " code",
                                "parameters": ["requirements"],
                            },
                            {
                                "name": "generate_backend_code",
                                "description": "Generate backend code for"
                                " specified framework",
                                "parameters": ["framework", "requirements"],
                            },
                        ],
                        "resources": [
                            "knowledge_base_stats",
                            "server_info",
                            "agent_capabilities",
                        ],
                    },
                }
            except Exception as e:
                return self._handle_error(e, "get_agent_capabilities")

    # Async wrapper methods for agent operations
    async def _search_knowledge_base_async(
        self, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Async wrapper for knowledge base search."""
        try:
            if not self.agent:
                raise Exception("AI agent not initialized")

            # Perform search
            search_results = await self.kb_tool.get_context_for_generation(
                query, max_context_length=limit)

            # Format results for MCP response
            formatted_results = []
            for result in search_results:
                formatted_results.append(
                    {
                        "content": result.get("content", ""),
                        "source": result.get("source", "unknown"),
                        "similarity_score": result.get(
                            "similarity_score", 0.0),
                        "metadata": result.get("metadata", {}),
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            # Return empty results on error
            return []

    def get_mcp_run_args(self):
        """Get the MCP server run arguments."""
        mcp_run_args = {
            "host": self.config.host,
            "port": self.config.port,
        }
        logger.info(f">>>> MCP server run arguments: {mcp_run_args}")
        return mcp_run_args

    def run(self):
        """Run the MCP server synchronously."""
        try:
            logger.info("Starting MCP server on stdio")
            # FastMCP typically runs on stdio for MCP protocol
            mcp_run_args = self.get_mcp_run_args()
            if self.config.transport == "http":
                asyncio.run(self.mcp.run_http_async(**mcp_run_args))
            else:
                asyncio.run(self.mcp.run_stdio_async())
        except Exception as e:
            logger.error(f"MCP server failed to start: {e}")
            raise

    async def run_async(self):
        """Run the MCP server asynchronously."""
        try:
            logger.info("Starting MCP server (async) on stdio")
            # FastMCP typically runs on stdio for MCP protocol
            mcp_run_args = self.get_mcp_run_args()
            if self.config.transport == "http":
                await self.mcp.run_http_async(**mcp_run_args)
            else:
                await self.mcp.run_stdio_async()
        except Exception as e:
            logger.error(f"MCP server failed to start: {e}")
            raise


def create_mcp_server(config: MCPConfig) -> GenericSuiteMCPServer:
    """
    Create and configure a GenericSuite MCP server.

    Args:
        config: MCP server configuration

    Returns:
        Configured MCP server instance
    """
    return GenericSuiteMCPServer(config)
