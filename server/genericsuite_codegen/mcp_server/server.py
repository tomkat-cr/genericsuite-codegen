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

logger = logging.getLogger(__name__)


@dataclass
class MCPConfig:
    """Configuration for the MCP server."""

    server_name: str = "genericsuite-codegen"
    server_version: str = "1.0.0"
    api_key: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8070
    debug: bool = False


class GenericSuiteMCPServer:
    """MCP Server for GenericSuite CodeGen system."""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.mcp = FastMCP(config.server_name)
        self.agent: Optional[GenericSuiteAgent] = None
        self.vector_db: Optional[DatabaseManager] = None

        # Initialize components
        self._setup_components()
        self._register_tools()
        self._register_resources()

    def _setup_components(self):
        """Initialize the AI agent and database components."""
        try:
            # Initialize vector database
            self.vector_db = DatabaseManager()

            # Initialize AI agent
            self.agent = GenericSuiteAgent()

            logger.info("MCP server components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MCP server components: {e}")
            raise

    def _register_tools(self):
        """Register MCP tools for external integration."""

        @self.mcp.tool()
        async def search_knowledge_base(query: str, limit: int = 5) -> Dict[str, Any]:
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
                    return {"error": "AI agent not initialized"}

                # Use the agent's knowledge base search capability
                results = await self._search_knowledge_base_async(query, limit)

                return {
                    "success": True,
                    "query": query,
                    "results": results,
                    "count": len(results),
                }

            except Exception as e:
                logger.error(f"Knowledge base search failed: {e}")
                return {"success": False, "error": str(e), "query": query}

        @self.mcp.tool()
        async def generate_json_config(requirements: str) -> Dict[str, Any]:
            """
            Generate GenericSuite JSON configuration based on requirements.

            Args:
                requirements: Description of the configuration requirements

            Returns:
                Dictionary containing the generated JSON configuration
            """
            try:
                if not self.agent:
                    return {"error": "AI agent not initialized"}

                # Generate JSON configuration using the agent
                config = await self._generate_json_config_async(requirements)

                return {
                    "success": True,
                    "requirements": requirements,
                    "configuration": config,
                }

            except Exception as e:
                logger.error(f"JSON config generation failed: {e}")
                return {"success": False, "error": str(e), "requirements": requirements}

        @self.mcp.tool()
        async def generate_langchain_tool(specification: str) -> Dict[str, Any]:
            """
            Generate a Langchain Tool based on specification.

            Args:
                specification: Description of the tool requirements

            Returns:
                Dictionary containing the generated Python code
            """
            try:
                if not self.agent:
                    return {"error": "AI agent not initialized"}

                # Generate Langchain tool using the agent
                code = await self._generate_langchain_tool_async(specification)

                return {
                    "success": True,
                    "specification": specification,
                    "code": code,
                    "type": "langchain_tool",
                }

            except Exception as e:
                logger.error(f"Langchain tool generation failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "specification": specification,
                }

        @self.mcp.tool()
        async def generate_mcp_tool(specification: str) -> Dict[str, Any]:
            """
            Generate an MCP Tool based on specification.

            Args:
                specification: Description of the tool requirements

            Returns:
                Dictionary containing the generated Python code
            """
            try:
                if not self.agent:
                    return {"error": "AI agent not initialized"}

                # Generate MCP tool using the agent
                code = await self._generate_mcp_tool_async(specification)

                return {
                    "success": True,
                    "specification": specification,
                    "code": code,
                    "type": "mcp_tool",
                }

            except Exception as e:
                logger.error(f"MCP tool generation failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "specification": specification,
                }

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
                    return {"error": "AI agent not initialized"}

                # Generate frontend code using the agent
                code_files = await self._generate_frontend_code_async(requirements)

                return {
                    "success": True,
                    "requirements": requirements,
                    "files": code_files,
                    "type": "frontend_code",
                }

            except Exception as e:
                logger.error(f"Frontend code generation failed: {e}")
                return {"success": False, "error": str(e), "requirements": requirements}

        @self.mcp.tool()
        async def generate_backend_code(
            framework: str, requirements: str
        ) -> Dict[str, Any]:
            """
            Generate backend code for specified framework following GenericSuite patterns.

            Args:
                framework: Backend framework (fastapi, flask, or chalice)
                requirements: Description of the backend requirements

            Returns:
                Dictionary containing the generated backend code files
            """
            try:
                if not self.agent:
                    return {"error": "AI agent not initialized"}

                # Validate framework
                valid_frameworks = ["fastapi", "flask", "chalice"]
                if framework.lower() not in valid_frameworks:
                    return {
                        "success": False,
                        "error": f"Invalid framework. Must be one of: {valid_frameworks}",
                        "framework": framework,
                    }

                # Generate backend code using the agent
                code_files = await self._generate_backend_code_async(
                    framework, requirements
                )

                return {
                    "success": True,
                    "framework": framework,
                    "requirements": requirements,
                    "files": code_files,
                    "type": "backend_code",
                }

            except Exception as e:
                logger.error(f"Backend code generation failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "framework": framework,
                    "requirements": requirements,
                }

    def _register_resources(self):
        """Register MCP resources for agent capabilities."""

        @self.mcp.resource(
            uri="genericsuite://knowledge_base_stats", name="Knowledge Base Statistics"
        )
        async def get_knowledge_base_stats() -> Dict[str, Any]:
            """Get statistics about the knowledge base."""
            try:
                if not self.vector_db:
                    return {"error": "Vector database not initialized"}

                # For now, return placeholder stats since we need to implement the actual method
                stats = {
                    "document_count": 0,  # self.vector_db.get_document_count(),
                    "status": "healthy",
                }

                return stats

            except Exception as e:
                logger.error(f"Failed to get knowledge base stats: {e}")
                return {"error": str(e)}

        @self.mcp.resource(uri="genericsuite://server_info", name="Server Information")
        async def get_server_info() -> Dict[str, Any]:
            """Get information about the MCP server."""
            return {
                "name": self.config.server_name,
                "version": self.config.server_version,
                "status": "running",
                "capabilities": [
                    "knowledge_base_search",
                    "json_config_generation",
                    "langchain_tool_generation",
                    "mcp_tool_generation",
                    "frontend_code_generation",
                    "backend_code_generation",
                ],
            }

    # Async wrapper methods for agent operations
    async def _search_knowledge_base_async(
        self, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Async wrapper for knowledge base search."""
        # This would typically use the agent's search functionality
        # For now, return a placeholder implementation
        return [
            {
                "content": f"Search result for: {query}",
                "source": "placeholder",
                "similarity_score": 0.9,
            }
        ]

    async def _generate_json_config_async(self, requirements: str) -> Dict[str, Any]:
        """Async wrapper for JSON config generation."""
        # Placeholder implementation
        return {
            "table_name": "example_table",
            "fields": [
                {"name": "id", "type": "string", "required": True},
                {"name": "name", "type": "string", "required": True},
            ],
        }

    async def _generate_langchain_tool_async(self, specification: str) -> str:
        """Async wrapper for Langchain tool generation."""
        # Placeholder implementation
        return f"""
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class ExampleToolInput(BaseModel):
    query: str = Field(description="Input query for the tool")

class ExampleTool(BaseTool):
    name = "example_tool"
    description = "Tool generated based on: {specification}"
    args_schema: Type[BaseModel] = ExampleToolInput

    def _run(self, query: str) -> str:
        # Implementation based on specification
        return f"Processed: {{query}}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
"""

    async def _generate_mcp_tool_async(self, specification: str) -> str:
        """Async wrapper for MCP tool generation."""
        # Placeholder implementation
        return f"""
from fastmcp import FastMCP
from typing import Dict, Any

mcp = FastMCP("example_tool")

@mcp.tool()
async def example_tool(query: str) -> Dict[str, Any]:
    \"\"\"
    Tool generated based on: {specification}
    \"\"\"
    return {{
        "result": f"Processed: {{query}}",
        "specification": "{specification}"
    }}
"""

    async def _generate_frontend_code_async(self, requirements: str) -> Dict[str, str]:
        """Async wrapper for frontend code generation."""
        # Placeholder implementation
        return {
            "App.tsx": f"// Frontend code generated based on: {requirements}\nexport default function App() {{ return <div>Hello World</div>; }}",
            "package.json": '{"name": "generated-app", "version": "1.0.0"}',
        }

    async def _generate_backend_code_async(
        self, framework: str, requirements: str
    ) -> Dict[str, str]:
        """Async wrapper for backend code generation."""
        # Placeholder implementation
        if framework.lower() == "fastapi":
            return {
                "main.py": f"# FastAPI backend generated based on: {requirements}\nfrom fastapi import FastAPI\napp = FastAPI()",
                "requirements.txt": "fastapi\nuvicorn",
            }
        elif framework.lower() == "flask":
            return {
                "app.py": f"# Flask backend generated based on: {requirements}\nfrom flask import Flask\napp = Flask(__name__)",
                "requirements.txt": "flask",
            }
        else:  # chalice
            return {
                "app.py": f"# Chalice backend generated based on: {requirements}\nfrom chalice import Chalice\napp = Chalice(app_name='generated-app')",
                "requirements.txt": "chalice",
            }

    def run(self):
        """Run the MCP server synchronously."""
        try:
            logger.info(f"Starting MCP server on stdio")
            # FastMCP typically runs on stdio for MCP protocol
            asyncio.run(self.mcp.run_stdio_async())
        except Exception as e:
            logger.error(f"MCP server failed to start: {e}")
            raise

    async def run_async(self):
        """Run the MCP server asynchronously."""
        try:
            logger.info(f"Starting MCP server (async) on stdio")
            # FastMCP typically runs on stdio for MCP protocol
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
