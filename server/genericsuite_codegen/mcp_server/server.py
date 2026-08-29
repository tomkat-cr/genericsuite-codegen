"""
FastMCP Server implementation for GenericSuite CodeGen.

This module implements the MCP server that exposes the AI agent capabilities
as standardized MCP tools and resources for integration with external tools.
"""

import json
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "FastMCP is required for MCP server functionality. "
        "Install it with: pip install fastmcp"
    )
from fastmcp.server.dependencies import get_http_headers

from genericsuite_codegen.agent.agent import GenericSuiteAgent
from genericsuite_codegen.agent.tools import KnowledgeBaseTool
from genericsuite_codegen.api.endpoint_methods import get_endpoint_methods
from genericsuite_codegen.database.setup import DatabaseManager
from genericsuite_codegen.utilities import local_path_to_url
from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_warning,
    log_error,
)
from genericsuite_codegen.utilities.env_vars import get_envvar
from genericsuite_codegen.utilities.utilities import \
    DEFAULT_USER_ID, MSG_ERROR_INVALID_KEY


DEBUG = True

DEFAULT_MCP_TRANSPORT = "http"


@dataclass
class MCPConfig:
    """Configuration for the MCP server."""

    server_name: str = "genericsuite-codegen"
    server_version: str = "1.0.0"
    api_key: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8070
    debug: bool = False
    transport: str = DEFAULT_MCP_TRANSPORT  # "http" or "stdio"
    # TODO: Use default user for now (in real app, this would come from auth)
    user_id: str = DEFAULT_USER_ID


class GenericSuiteMCPServer:
    """MCP Server for GenericSuite CodeGen system."""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.mcp = FastMCP(config.server_name)
        self.agent: Optional[GenericSuiteAgent] = None
        self.vector_db: Optional[DatabaseManager] = None
        self.kb_tool = None
        self.user_id = config.user_id

        self.methods = get_endpoint_methods()

        # Initialize components
        self._setup_components()
        self._setup_authentication()
        self._register_tools()
        self._register_resources()

    def verify_api_key(self):
        """
        Verify the API key in the Authorization header
        """
        access_token = get_access_token().replace("Bearer ", "")
        if not access_token:
            log_error("No access token found")
            return False
        if not self.config.api_key:
            log_error("No configured API key found")
            return False
        _ = DEBUG and log_debug(
            f"Verifying API key: {access_token} == {self.config.api_key}")
        return access_token == self.config.api_key

    def _setup_components(self):
        """Initialize the AI agent and database components."""
        try:
            # Initialize vector database
            self.vector_db = DatabaseManager()

            # Initialize AI agent
            self.agent = GenericSuiteAgent()

            # Initialize knowledge base tool
            self.kb_tool = KnowledgeBaseTool()

            _ = DEBUG and log_debug(
                "MCP server components initialized successfully")

        except Exception as e:
            log_error(
                f"Failed to initialize MCP server components: {e}",
                exc_info=True)
            raise

    def _setup_authentication(self):
        """Setup MCP authentication and security."""
        try:
            # Add authentication middleware if API key is provided
            if self.config.api_key:
                _ = DEBUG and log_debug(
                    "MCP authentication enabled with API key")
                # Note: FastMCP handles authentication through the protocol
                # The API key will be validated in tool calls
            else:
                log_warning(
                    "MCP server running without authentication")

        except Exception as e:
            log_error(
                f"Failed to setup MCP authentication: {e}",
                exc_info=True)
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
            log_error(f"Request validation failed: {e}", exc_info=True)
            return False

    def _handle_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Centralized error handling for MCP operations."""
        error_id = f"mcp_error_{hash(str(error)) % 10000:04d}"

        log_error(
            f"MCP Error [{error_id}] in {context}: {error}")

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
            query: str,
            limit: int = 5
        ) -> Dict[str, Any]:
            """
            Search the GenericSuite knowledge base for relevant information.

            Args:
                query: The search query string
                limit: Maximum number of results to return (default: 5)

            Returns:
                Dictionary containing search results and metadata
            """
            if not self.verify_api_key():
                return self._handle_error(
                    Exception(MSG_ERROR_INVALID_KEY),
                    "search_knowledge_base"
                )
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception("AI agent not initialized"),
                        "search_knowledge_base"
                    )

                # Use the agent's knowledge base search capability
                results = await self._search_knowledge_base_async(query, limit)

                _ = DEBUG and log_debug(
                    f">>> Search knowledge base results: {results}")

                final_result = {
                    "success": results["success"],
                    "query": results["query"],
                    "results": results["results"],
                    "sources": results["sources"],
                    "context": results["context"],
                    "count": results["count"],
                    "error": results["error"],
                }

                _ = DEBUG and log_debug(
                    f">>> Search knowledge base final result: {final_result}")

                return final_result

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
            if not self.verify_api_key():
                return self._handle_error(
                    Exception(MSG_ERROR_INVALID_KEY),
                    "generate_json_config"
                )
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception("AI agent not initialized"),
                        "generate_json_config"
                    )

                # Generate JSON configuration using the agent
                config = await self.methods.generate_json_config_endpoint(
                    requirements=requirements,
                    table_name=table_name,
                    user_id=self.user_id,
                    config_type=config_type,
                )

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
            if not self.verify_api_key():
                return self._handle_error(
                    Exception(MSG_ERROR_INVALID_KEY),
                    "generate_langchain_tool"
                )
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception(
                            "AI agent not initialized"),
                        "generate_langchain_tool"
                    )

                # Generate Langchain tool using the agent
                code = await self.methods.generate_python_code_endpoint(
                    requirements=requirements,
                    tool_name=tool_name,
                    description=description,
                    user_id=self.user_id,
                    code_type="langchain_tool",
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
            if not self.verify_api_key():
                return self._handle_error(
                    Exception(MSG_ERROR_INVALID_KEY),
                    "generate_mcp_tool"
                )
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception(
                            "AI agent not initialized"), "generate_mcp_tool"
                    )

                # Generate MCP tool using the agent
                code = await self.methods.generate_python_code_endpoint(
                    requirements=requirements,
                    tool_name=tool_name,
                    description=description,
                    user_id=self.user_id,
                    code_type="mcp_tool",
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
        async def generate_frontend_code(requirements: str
                                         ) -> Dict[str, Any]:
            """
            Generate ReactJS frontend code following GenericSuite patterns.

            Args:
                requirements: Description of the frontend requirements

            Returns:
                Dictionary containing the generated frontend code files
            """
            if not self.verify_api_key():
                return self._handle_error(
                    Exception(MSG_ERROR_INVALID_KEY),
                    "generate_frontend_code"
                )
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
            if not self.verify_api_key():
                return self._handle_error(
                    Exception(MSG_ERROR_INVALID_KEY),
                    "generate_backend_code"
                )
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
            if not self.verify_api_key():
                return self._handle_error(
                    Exception(MSG_ERROR_INVALID_KEY),
                    "get_knowledge_base_stats"
                )
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
            if not self.verify_api_key():
                return self._handle_error(
                    Exception(MSG_ERROR_INVALID_KEY),
                    "get_server_info"
                )
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
            if not self.verify_api_key():
                return self._handle_error(
                    Exception(MSG_ERROR_INVALID_KEY),
                    "get_agent_capabilities"
                )
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
        self,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Async wrapper for knowledge base search."""
        final_result = {
            "success": True,
            "query": query,
            "results": [],
            "count": 0,
            "error": None,
        }
        try:
            if not self.agent:
                raise Exception("AI agent not initialized")

            final_context, search_results, raw_results = \
                self.kb_tool.get_context_for_generation(
                    query, limit=limit)

            _ = DEBUG and log_debug(
                ">>> _search_knowledge_base_async"
                f"\n | final_context: {final_context}"
                f"\n | search_results: {search_results}"
                f"\n | raw_results: {raw_results}"
            )

            # Format results for MCP response
            formatted_results = []
            # for result in search_results.results:
            for result in raw_results:
                formatted_results.append({
                    "content": local_path_to_url(result.content, False),
                    "source": local_path_to_url(result.document_path, True),
                    "similarity_score": result.similarity_score,
                    "metadata": result.metadata,
                })

            final_result["results"] = formatted_results
            final_result["context"] = final_context
            final_result["sources"] = search_results
            final_result["count"] = len(raw_results)
            return final_result

        except Exception as e:
            # Get error source and line number if possible
            log_error(
                f"Knowledge base search failed [SKBA-010]: {e}",
                exc_info=True
            )
            # Return empty results on error
            final_result["success"] = False
            final_result["error"] = str(e)
            raise
            return final_result

    def get_mcp_run_args(self):
        """Get the MCP server run arguments."""
        mcp_run_args = {
            "host": self.config.host,
            "port": self.config.port,
        }
        _ = DEBUG and log_debug(
            f">>>> MCP server run arguments: {mcp_run_args}")
        return mcp_run_args

    def run(self):
        """Run the MCP server synchronously."""
        try:
            _ = DEBUG and log_debug(
                f"Starting MCP server on {self.config.transport}")
            # FastMCP typically runs on stdio for MCP protocol
            mcp_run_args = self.get_mcp_run_args()
            if self.config.transport == "http":
                asyncio.run(self.mcp.run_http_async(**mcp_run_args))
            else:
                asyncio.run(self.mcp.run_stdio_async())
        except Exception as e:
            log_error(f"MCP server failed to start: {e}", exc_info=True)
            raise

    async def run_async(self):
        """Run the MCP server asynchronously."""
        try:
            _ = DEBUG and log_debug(
                f"Starting MCP server (async) on {self.config.transport}")
            # FastMCP typically runs on stdio for MCP protocol
            mcp_run_args = self.get_mcp_run_args()
            if self.config.transport == "http":
                await self.mcp.run_http_async(**mcp_run_args)
            else:
                await self.mcp.run_stdio_async()
        except Exception as e:
            log_error(f"MCP server failed to start: {e}", exc_info=True)
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


def load_environment(current_dir: str):
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv

        # Look for .env file in current directory or parent directories
        env_file = current_dir / ".env"
        if not env_file.exists():
            env_file = current_dir.parent / ".env"

        if env_file.exists():
            load_dotenv(env_file)
            _ = DEBUG and log_debug(f"Loaded environment from {env_file}")
        else:
            log_warning(
                "No .env file found, using system environment variables")

    except ImportError:
        log_warning(
            "python-dotenv not available, using system environment variables")


def validate_environment():
    """Validate required environment variables."""
    required_vars = []
    optional_vars = {
        "MCP_SERVER_HOST": "0.0.0.0",
        "MCP_SERVER_PORT": "8072",
        "MCP_API_KEY": None,
        "MCP_DEBUG": "0",
        "MCP_TRANSPORT": DEFAULT_MCP_TRANSPORT  # "http" or "stdio"
    }

    missing_vars = []
    for var in required_vars:
        if not get_envvar(var):
            missing_vars.append(var)

    if missing_vars:
        log_error(f"Missing required environment variables: {missing_vars}")
        return False

    # Log optional variables
    for var, default in optional_vars.items():
        value = get_envvar(var, default)
        _ = DEBUG and log_debug(f"{var}: {value}")

    return True


def get_mcp_config() -> MCPConfig:
    """Get MCP server configuration from environment variables."""
    return MCPConfig(
        server_name=get_envvar("MCP_SERVER_NAME", "genericsuite-codegen"),
        server_version=get_envvar("MCP_SERVER_VERSION", "1.0.0"),
        api_key=get_envvar("MCP_API_KEY"),
        host=get_envvar("MCP_SERVER_HOST", "0.0.0.0"),
        port=int(get_envvar("MCP_SERVER_PORT", "8072")),
        debug=get_envvar("MCP_DEBUG", "0") == "1",
        transport=get_envvar("MCP_TRANSPORT", DEFAULT_MCP_TRANSPORT),
    )


def report_mcp_config(config: MCPConfig):
    """Report MCP server configuration."""
    if DEBUG:
        log_debug("Server configuration:")
        log_debug(f"  Name: {config.server_name}")
        log_debug(f"  Version: {config.server_version}")
        log_debug(f"  Host: {config.host}")
        log_debug(f"  Port: {config.port}")
        log_debug(f"  Debug: {config.debug}")
        log_debug(f"  API Key: {'Set' if config.api_key else 'Not set'}")
        log_debug(f"  Transport: {config.transport}")


def print_output(message: str):
    """Print output to the terminal."""
    mcp_transport = get_envvar("MCP_TRANSPORT", DEFAULT_MCP_TRANSPORT)
    if mcp_transport == "http":
        print(message)
    else:
        json_message = json.dumps({
            "message": message})
        print(json_message)


def get_access_token():
    """
    Get the access token
    """
    headers = get_http_headers()
    return headers.get("authorization")
