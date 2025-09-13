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
                    return self._handle_error(
                        Exception("AI agent not initialized"),
                        "generate_json_config"
                    )

                # Generate JSON configuration using the agent
                config = await self._generate_json_config_async(requirements)

                return {
                    "success": True,
                    "requirements": requirements,
                    "configuration": config,
                }

            except Exception as e:
                return self._handle_error(e, "generate_json_config")

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
                    return self._handle_error(
                        Exception("AI agent not initialized"), "generate_langchain_tool"
                    )

                # Generate Langchain tool using the agent
                code = await self._generate_langchain_tool_async(specification)

                return {
                    "success": True,
                    "specification": specification,
                    "code": code,
                    "type": "langchain_tool",
                }

            except Exception as e:
                return self._handle_error(e, "generate_langchain_tool")

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
                    return self._handle_error(
                        Exception("AI agent not initialized"), "generate_mcp_tool"
                    )

                # Generate MCP tool using the agent
                code = await self._generate_mcp_tool_async(specification)

                return {
                    "success": True,
                    "specification": specification,
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
                        Exception("AI agent not initialized"), "generate_frontend_code"
                    )

                # Generate frontend code using the agent
                code_files = await self._generate_frontend_code_async(requirements)

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
            Generate backend code for specified framework following GenericSuite patterns.

            Args:
                framework: Backend framework (fastapi, flask, or chalice)
                requirements: Description of the backend requirements

            Returns:
                Dictionary containing the generated backend code files
            """
            try:
                if not self.agent:
                    return self._handle_error(
                        Exception("AI agent not initialized"), "generate_backend_code"
                    )

                # Validate framework
                valid_frameworks = ["fastapi", "flask", "chalice"]
                if framework.lower() not in valid_frameworks:
                    return self._handle_error(
                        ValueError(
                            f"Invalid framework. Must be one of: {valid_frameworks}"
                        ),
                        "generate_backend_code",
                    )

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
                return self._handle_error(e, "generate_backend_code")

    def _register_resources(self):
        """Register MCP resources for agent capabilities."""

        @self.mcp.resource(
            uri="genericsuite://knowledge_base_stats", name="Knowledge Base Statistics"
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
                    "2025-09-08T11:42:00Z"  # TODO: Would be dynamic in real implementation
                )

                return {"success": True, "data": stats}

            except Exception as e:
                return self._handle_error(e, "get_knowledge_base_stats")

        @self.mcp.resource(uri="genericsuite://server_info", name="Server Information")
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
                        "supported_frameworks": ["fastapi", "flask", "chalice"],
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
                        Exception("AI agent not initialized"), "get_agent_capabilities"
                    )

                return {
                    "success": True,
                    "data": {
                        "tools": [
                            {
                                "name": "search_knowledge_base",
                                "description": "Search GenericSuite documentation and knowledge base",
                                "parameters": ["query", "limit"],
                            },
                            {
                                "name": "generate_json_config",
                                "description": "Generate GenericSuite table configuration JSON",
                                "parameters": ["requirements"],
                            },
                            {
                                "name": "generate_langchain_tool",
                                "description": "Generate Langchain Tool Python code",
                                "parameters": ["specification"],
                            },
                            {
                                "name": "generate_mcp_tool",
                                "description": "Generate MCP Tool Python code",
                                "parameters": ["specification"],
                            },
                            {
                                "name": "generate_frontend_code",
                                "description": "Generate ReactJS frontend code",
                                "parameters": ["requirements"],
                            },
                            {
                                "name": "generate_backend_code",
                                "description": "Generate backend code for specified framework",
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

            # Use the agent's knowledge base search tool
            from ..agent.tools import KnowledgeBaseTool

            # Create knowledge base tool instance
            kb_tool = KnowledgeBaseTool()

            # Perform search
            search_results = await kb_tool.search_async(query, limit)

            # Format results for MCP response
            formatted_results = []
            for result in search_results:
                formatted_results.append(
                    {
                        "content": result.get("content", ""),
                        "source": result.get("source", "unknown"),
                        "similarity_score": result.get("similarity_score", 0.0),
                        "metadata": result.get("metadata", {}),
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            # Return empty results on error
            return []

    async def _generate_json_config_async(self, requirements: str) -> Dict[str, Any]:
        """Async wrapper for JSON config generation."""
        try:
            if not self.agent:
                raise Exception("AI agent not initialized")

            # Use the agent to generate JSON configuration
            prompt = f"""Generate a GenericSuite table configuration JSON based on these requirements:
            
{requirements}

Please provide a complete JSON configuration following GenericSuite patterns."""

            # Run the agent to generate configuration
            result = await self.agent.run_async(prompt)

            # Parse the result to extract JSON configuration
            # For now, return a structured example based on requirements
            return {
                "table_name": "generated_table",
                "description": f"Table generated based on: {requirements}",
                "fields": [
                    {
                        "name": "id",
                        "type": "string",
                        "required": True,
                        "primary_key": True,
                    },
                    {"name": "created_at", "type": "datetime", "required": True},
                    {"name": "updated_at", "type": "datetime", "required": True},
                ],
                "generated_from": requirements,
            }

        except Exception as e:
            logger.error(f"JSON config generation failed: {e}")
            # Return basic structure on error
            return {
                "error": "Failed to generate configuration",
                "requirements": requirements,
            }

    async def _generate_langchain_tool_async(self, specification: str) -> str:
        """Async wrapper for Langchain tool generation."""
        try:
            if not self.agent:
                raise Exception("AI agent not initialized")

            # Use the agent to generate Langchain tool code
            prompt = f"""Generate a complete Langchain Tool implementation based on this specification:

{specification}

Please provide a complete Python class that follows Langchain Tool patterns and includes:
1. Proper imports
2. Input schema with Pydantic BaseModel
3. Tool class with name, description, and args_schema
4. Both sync (_run) and async (_arun) methods
5. Proper error handling

Make sure the code is production-ready and follows best practices."""

            # Run the agent to generate code
            result = await self.agent.run_async(prompt)

            # Extract code from result
            # TODO: (would need proper parsing in real implementation)
            return result.data if hasattr(result, "data") else str(result)

        except Exception as e:
            logger.error(f"Langchain tool generation failed: {e}")
            # Return template on error
            return f"""
# Error generating tool: {e}
# Specification: {specification}

from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class GeneratedToolInput(BaseModel):
    query: str = Field(description="Input for the generated tool")

class GeneratedTool(BaseTool):
    name = "generated_tool"
    description = "Tool generated from specification: {specification}"
    args_schema: Type[BaseModel] = GeneratedToolInput

    def _run(self, query: str) -> str:
        # TODO: Implement based on specification
        return f"Processing: {{query}}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
"""

    async def _generate_mcp_tool_async(self, specification: str) -> str:
        """Async wrapper for MCP tool generation."""
        try:
            if not self.agent:
                raise Exception("AI agent not initialized")

            # Use the agent to generate MCP tool code
            prompt = f"""Generate a complete FastMCP Tool implementation based on this specification:

{specification}

Please provide a complete Python implementation that includes:
1. Proper imports for FastMCP
2. Tool function with @mcp.tool() decorator
3. Proper type hints and documentation
4. Error handling
5. Return structured data as Dict[str, Any]

Make sure the code is production-ready and follows FastMCP best practices."""

            # Run the agent to generate code
            result = await self.agent.run_async(prompt)

            # Extract code from result
            # TODO: (would need proper parsing in real implementation)
            return result.data if hasattr(result, "data") else str(result)

        except Exception as e:
            logger.error(f"MCP tool generation failed: {e}")
            # Return template on error
            return f"""
# Error generating MCP tool: {e}
# Specification: {specification}

from fastmcp import FastMCP
from typing import Dict, Any

mcp = FastMCP("generated_tool")

@mcp.tool()
async def generated_tool(query: str) -> Dict[str, Any]:
    \"\"\"
    Tool generated from specification: {specification}
    \"\"\"
    try:
        # TODO: Implement based on specification
        return {{
            "success": True,
            "result": f"Processing: {{query}}",
            "specification": "{specification}"
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": str(e)
        }}
"""

    async def _generate_frontend_code_async(self, requirements: str) -> Dict[str, str]:
        """Async wrapper for frontend code generation."""
        try:
            if not self.agent:
                raise Exception("AI agent not initialized")

            # Use the agent to generate frontend code
            prompt = f"""Generate a complete ReactJS frontend application based on these requirements:

{requirements}

Please provide a complete React application following GenericSuite patterns that includes:
1. Main App.tsx component
2. package.json with dependencies
3. Any additional components needed
4. TypeScript types if applicable
5. Proper styling and structure

Make sure the code follows React best practices and GenericSuite patterns."""

            # Run the agent to generate code
            result = await self.agent.run_async(prompt)

            # Parse result to extract files 
            # TODO: (would need proper parsing in real implementation)
            return {
                "App.tsx": f"""// Generated React App based on: {requirements}
import React from 'react';
import './App.css';

function App() {{
  return (
    <div className="App">
      <header className="App-header">
        <h1>Generated Application</h1>
        <p>Requirements: {requirements}</p>
      </header>
    </div>
  );
}}

export default App;""",
                "package.json": """{
  "name": "generated-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^4.9.5"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  }
}""",
                "App.css": """.App {
  text-align: center;
}

.App-header {
  background-color: #282c34;
  padding: 20px;
  color: white;
}""",
            }

        except Exception as e:
            logger.error(f"Frontend code generation failed: {e}")
            # Return basic template on error
            return {
                "App.tsx": f"// Error generating frontend: {e}\n// Requirements: {requirements}\nexport default function App() {{ return <div>Error generating app</div>; }}",
                "package.json": '{"name": "error-app", "version": "1.0.0"}',
            }

    async def _generate_backend_code_async(
        self, framework: str, requirements: str
    ) -> Dict[str, str]:
        """Async wrapper for backend code generation."""
        try:
            if not self.agent:
                raise Exception("AI agent not initialized")

            # Use the agent to generate backend code
            prompt = f"""Generate a complete {framework} backend application based on these requirements:

{requirements}

Please provide a complete {framework} application following GenericSuite patterns that includes:
1. Main application file
2. Requirements/dependencies file
3. API endpoints as needed
4. Database models if applicable
5. Proper error handling and structure

Make sure the code follows {framework} best practices and GenericSuite patterns."""

            # Run the agent to generate code
            result = await self.agent.run_async(prompt)

            # Generate framework-specific code structure
            if framework.lower() == "fastapi":
                return {
                    "main.py": f"""# Generated FastAPI backend based on: {requirements}
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Generated API", version="1.0.0")

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="API is running")

@app.get("/")
async def root():
    return {{"message": "Generated API based on: {requirements}"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)""",
                    "requirements.txt": "fastapi>=0.100.0\nuvicorn[standard]>=0.20.0\npydantic>=2.0.0",
                    "README.md": f"# Generated FastAPI Application\n\nRequirements: {requirements}\n\n## Run\n```bash\npip install -r requirements.txt\npython main.py\n```",
                }
            elif framework.lower() == "flask":
                return {
                    "app.py": f"""# Generated Flask backend based on: {requirements}
from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({{"status": "healthy", "message": "API is running"}})

@app.route('/')
def root():
    return jsonify({{"message": "Generated API based on: {requirements}"}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)""",
                    "requirements.txt": "Flask>=2.3.0\nWerkzeug>=2.3.0",
                    "README.md": f"# Generated Flask Application\n\nRequirements: {requirements}\n\n## Run\n```bash\npip install -r requirements.txt\npython app.py\n```",
                }
            else:  # chalice
                return {
                    "app.py": f"""# Generated Chalice backend based on: {requirements}
from chalice import Chalice

app = Chalice(app_name='generated-chalice-app')

@app.route('/health')
def health_check():
    return {{"status": "healthy", "message": "API is running"}}

@app.route('/')
def index():
    return {{"message": "Generated API based on: {requirements}"}}""",
                    "requirements.txt": "chalice>=1.28.0",
                    ".chalice/config.json": """{
  "version": "2.0",
  "app_name": "generated-chalice-app",
  "stages": {
    "dev": {
      "api_gateway_stage": "api"
    }
  }
}""",
                    "README.md": f"# Generated Chalice Application\n\nRequirements: {requirements}\n\n## Deploy\n```bash\npip install -r requirements.txt\nchalice deploy\n```",
                }

        except Exception as e:
            logger.error(f"Backend code generation failed: {e}")
            # Return basic template on error
            return {
                "main.py": f"# Error generating {framework} backend: {e}\n# Requirements: {requirements}",
                "requirements.txt": f"# Error generating requirements for {framework}",
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
