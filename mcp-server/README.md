# GenericSuite CodeGen MCP Server

This is the MCP (Model Context Protocol) server for the GenericSuite CodeGen RAG AI system. It exposes the AI agent capabilities as standardized MCP tools and resources for integration with external applications.

## Features

- **Knowledge Base Search**: Vector similarity search through GenericSuite documentation
- **JSON Configuration Generation**: Generate table, form, and menu configurations
- **Python Code Generation**: Create Langchain tools, MCP tools, and API code
- **Frontend Code Generation**: Generate ReactJS components and applications
- **Backend Code Generation**: Create FastAPI, Flask, or Chalice backend code
- **Agent Query Interface**: Direct access to the AI agent for general queries

## Installation

```bash
# Install dependencies
poetry install

# Or using pip
pip install -r requirements.txt
```

## Configuration

Create a `.env` file or set environment variables:

```bash
make init-app-environment
```

Configure environment variablesL

```bash
# MCP Server Configuration
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8070
MCP_API_KEY=your_api_key_here
MCP_DEBUG=0

# AI Agent Configuration
LLM_API_KEY=your_openai_key
LLM_PROVIDER=openai
LLM_MODEL=gpt-4

# Database Configuration
MONGODB_URI=mongodb://localhost:27017/genericsuite_codegen
```

## Usage

### Start the MCP Server

```bash
# Install development dependencies
make install

# Start the server (with Docker Compose)
make run
```

### Test the Server

```bash
# Run tests
poetry run python test_mcp_server.py

# Or using Make
make test
```

## MCP Tools

The server exposes the following MCP tools:

### search_knowledge_base
Search the GenericSuite documentation and examples.

```json
{
  "query": "How to create a table configuration",
  "limit": 5,
  "file_type_filter": "json"
}
```

### generate_json_config
Generate JSON configurations for GenericSuite.

```json
{
  "requirements": "Create a user management table with name, email, role fields",
  "config_type": "table",
  "table_name": "users"
}
```

### generate_python_code
Generate Python code for tools and applications.

```json
{
  "requirements": "Create a Langchain tool for user authentication",
  "code_type": "langchain"
}
```

### generate_frontend_code
Generate ReactJS frontend components.

```json
{
  "requirements": "Create a user profile form with validation"
}
```

### generate_backend_code
Generate backend API code.

```json
{
  "requirements": "Create CRUD endpoints for user management",
  "framework": "fastapi"
}
```

### query_agent
Direct query interface to the AI agent.

```json
{
  "query": "What is GenericSuite?",
  "task_type": "general"
}
```

## MCP Resources

The server provides these MCP resources:

- `genericsuite://capabilities` - Server capabilities and status
- `genericsuite://examples` - Usage examples for all tools
- `genericsuite://health` - Health status of server components

## Development

```bash
# Install development dependencies
make install

# Start the server
make dev
```

## Integration

To use this MCP server with MCP-compatible applications:

1. Start the server on your desired host/port
2. Configure your MCP client to connect to the server
3. Use the exposed tools and resources in your application

Example MCP client configuration:

```json
{
    "mcpServers": {
        "genericsuite-codegen": {
            "url": "http://localhost:8070/mcp",
            "headers": {
                "MCP_API_KEY": "ag-api-key-..."
            }
        }
    }
}
```

There are configuration files for different MCP clients in the `mcp-server` directory.

- [claude_desktop_http_config.json](./claude_desktop_http_config.json) for Claude Desktop with HTTP transport
- [claude_desktop_stdio_config.json](./claude_desktop_stdio_config.json) for Claude Desktop with STDIO transport
- [vscode_mcp_http_config.json](./vscode_mcp_http_config.json) for VS Code with HTTP transport
- [vscode_mcp_stdio_config.json](./vscode_mcp_stdio_config.json) for VS Code with STDIO transport
- [kiro_stdio_config.json](./kiro_stdio_config.json) for Kiro with STDIO transport (Kiro is not supported with HTTP transport)

## Troubleshooting

### Common Issues

1. **Agent not available**: Ensure the main server components are properly installed
2. **Database connection failed**: Check MongoDB connection and credentials
3. **API key errors**: Verify OpenAI API key is set correctly
4. **Port conflicts**: Change MCP_SERVER_PORT if 8070 is in use

### Logs

Server logs are written to `mcp_server.log` and console output.

### Health Check

Check server health at runtime:
```bash
# Test server functionality
poetry run python test_mcp_server.py
```