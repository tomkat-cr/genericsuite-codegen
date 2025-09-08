# GenericSuite CodeGen MCP Server Usage Guide

## Overview

The GenericSuite CodeGen MCP Server exposes AI-powered code generation and documentation search capabilities through the Model Context Protocol (MCP). This allows external applications to integrate with the GenericSuite CodeGen system.

## Quick Start

### 1. Start the MCP Server

```bash
# Using Poetry (recommended)
cd mcp-server
poetry install
poetry run python start_mcp_server.py

# Or using Make
make start

# Or using the module
poetry run python -m genericsuite_codegen
```

### 2. Configure Environment

Create a `.env` file in the mcp-server directory:

```bash
# MCP Server Configuration
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8070
MCP_API_KEY=your_api_key_here
MCP_REQUIRE_AUTH=0
MCP_DEBUG=0

# AI Agent Configuration (if available)
OPENAI_API_KEY=your_openai_key
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
```

## Available MCP Tools

### 1. search_knowledge_base

Search the GenericSuite documentation and examples using vector similarity.

**Request Format:**
```json
{
  "query": "How to create a table configuration",
  "limit": 5,
  "file_type_filter": "json"
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "Document content...",
      "source": "path/to/document.md",
      "similarity_score": 0.95,
      "file_type": "md",
      "metadata": {}
    }
  ],
  "total_results": 5,
  "query": "How to create a table configuration",
  "context_summary": "Retrieved 5 relevant documents...",
  "sources": ["path/to/document.md"]
}
```

### 2. generate_json_config

Generate JSON configurations for GenericSuite tables, forms, and menus.

**Request Format:**
```json
{
  "requirements": "Create a user management table with name, email, role fields",
  "config_type": "table",
  "table_name": "users"
}
```

**Response:**
```json
{
  "configuration": {
    "table_name": "users",
    "table_config": {
      "fields": {
        "name": {"type": "string", "required": true},
        "email": {"type": "email", "required": true},
        "role": {"type": "select", "options": ["admin", "user"]}
      }
    }
  },
  "config_type": "table",
  "requirements": "Create a user management table...",
  "sources": ["path/to/reference.md"],
  "generated_at": "2024-01-01T12:00:00Z"
}
```

### 3. generate_python_code

Generate Python code for tools, Langchain integrations, and MCP servers.

**Request Format:**
```json
{
  "requirements": "Create a Langchain tool for user authentication",
  "code_type": "langchain"
}
```

**Response:**
```json
{
  "code": "from langchain.tools import BaseTool...",
  "code_type": "langchain",
  "requirements": "Create a Langchain tool...",
  "sources": ["path/to/example.py"],
  "generated_at": "2024-01-01T12:00:00Z"
}
```

### 4. generate_frontend_code

Generate ReactJS frontend components and applications.

**Request Format:**
```json
{
  "requirements": "Create a user profile form with validation"
}
```

**Response:**
```json
{
  "files": [
    {
      "filename": "UserProfileForm.jsx",
      "content": "import React from 'react'...",
      "language": "jsx",
      "size": 1024
    }
  ],
  "requirements": "Create a user profile form...",
  "sources": ["path/to/component.jsx"],
  "generated_at": "2024-01-01T12:00:00Z"
}
```

### 5. generate_backend_code

Generate backend API code for FastAPI, Flask, or Chalice.

**Request Format:**
```json
{
  "requirements": "Create CRUD endpoints for user management",
  "framework": "fastapi"
}
```

**Response:**
```json
{
  "files": [
    {
      "filename": "user_api.py",
      "content": "from fastapi import FastAPI...",
      "language": "python",
      "size": 2048
    }
  ],
  "framework": "fastapi",
  "requirements": "Create CRUD endpoints...",
  "sources": ["path/to/api.py"],
  "generated_at": "2024-01-01T12:00:00Z"
}
```

### 6. query_agent

Direct query interface to the AI agent for general questions.

**Request Format:**
```json
{
  "query": "What is GenericSuite?",
  "task_type": "general"
}
```

**Response:**
```json
{
  "response": "GenericSuite is a comprehensive framework...",
  "task_type": "general",
  "sources": ["path/to/docs.md"],
  "model_used": "gpt-4",
  "timestamp": "2024-01-01T12:00:00Z",
  "token_usage": {
    "prompt_tokens": 100,
    "completion_tokens": 200,
    "total_tokens": 300
  }
}
```

### 7. get_knowledge_base_stats

Get statistics about the knowledge base and system status.

**Request Format:**
```json
{}
```

**Response:**
```json
{
  "knowledge_base_documents": 1500,
  "conversations": 250,
  "file_type_distribution": [
    {"file_type": "md", "count": 800},
    {"file_type": "py", "count": 400},
    {"file_type": "json", "count": 300}
  ],
  "last_updated": "2024-01-01T12:00:00Z"
}
```

## MCP Resources

### genericsuite://capabilities

Get information about server capabilities and status.

### genericsuite://examples

Get usage examples for all MCP tools.

### genericsuite://health

Get health status of server components.

## Integration Examples

### Using with MCP Client

```python
import asyncio
from mcp_client import MCPClient

async def main():
    client = MCPClient("http://localhost:8070")
    
    # Search knowledge base
    result = await client.call_tool("search_knowledge_base", {
        "query": "table configuration",
        "limit": 3
    })
    
    print(f"Found {result['total_results']} results")
    
    # Generate JSON config
    config = await client.call_tool("generate_json_config", {
        "requirements": "User table with name and email",
        "config_type": "table"
    })
    
    print(f"Generated config: {config['configuration']}")

asyncio.run(main())
```

### Using with Kiro MCP Configuration

Add to your `.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "genericsuite-codegen": {
      "command": "poetry",
      "args": ["run", "python", "start_mcp_server.py"],
      "cwd": "/path/to/mcp-server",
      "env": {
        "MCP_SERVER_PORT": "8070",
        "MCP_DEBUG": "0"
      },
      "disabled": false,
      "autoApprove": [
        "search_knowledge_base",
        "get_knowledge_base_stats"
      ]
    }
  }
}
```

## Error Handling

All tools return error information when something goes wrong:

```json
{
  "error": "Tool execution failed: Connection timeout",
  "error_type": "ConnectionError",
  "tool": "search_knowledge_base"
}
```

Common error scenarios:
- **Agent not available**: AI agent components not initialized
- **Knowledge base unavailable**: Database connection issues
- **Authentication failed**: Invalid API key (when auth enabled)
- **Rate limit exceeded**: Too many requests per hour
- **Invalid request**: Missing required parameters

## Authentication

When authentication is enabled (`MCP_REQUIRE_AUTH=1`):

1. Set `MCP_API_KEY` in environment
2. Include API key in requests:
   - Header: `X-API-Key: your_api_key`
   - Or: `Authorization: Bearer your_api_key`

## Rate Limiting

Default rate limit: 100 requests per hour per client.

Response headers include:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## Troubleshooting

### Server won't start
- Check port availability: `lsof -i :8070`
- Verify dependencies: `poetry install`
- Check logs in `mcp_server.log`

### Tools return errors
- Verify environment variables are set
- Check database connectivity
- Ensure OpenAI API key is valid

### Performance issues
- Increase rate limits if needed
- Check database performance
- Monitor memory usage

## Development

### Running Tests

```bash
# Basic functionality test
poetry run python simple_test.py

# Full test suite (requires server components)
poetry run python test_mcp_server.py

# Using Make
make test
```

### Code Formatting

```bash
make format  # Format code
make lint    # Check code quality
```

### Adding New Tools

1. Add tool function in `mcp_server.py`
2. Register with `@self.mcp.tool(name="tool_name")`
3. Add error handling with `@handle_errors`
4. Update documentation

For more information, see the main README.md file.