# GenericSuite CodeGen MCP Server

This is the MCP (Model Context Protocol) server for the GenericSuite CodeGen RAG AI system. It exposes the AI agent capabilities as standardized MCP tools and resources for integration with external applications.

## Features

- **Knowledge Base Search**: Vector similarity search through GenericSuite documentation
- **Context-Aware Generation**: Automatic search for relevant examples based on code generation type using configurable search templates
- **JSON Configuration Generation**: Generate table, form, and menu configurations with GenericSuite patterns
- **Python Code Generation**: Create LangChain tools, MCP tools, and API code following GenericSuite conventions
- **Frontend Code Generation**: Generate ReactJS components and applications (including AI-enhanced)
- **Backend Code Generation**: Create FastAPI, Flask, or Chalice backend code (including AI-enhanced)
- **Agent Query Interface**: Direct access to the AI agent for general queries with enhanced context retrieval
- **Document Retrieval Tool**: Secure agent tool for accessing complete GenericSuite documents from local storage with path validation
- **Enhanced Search Types**: Comprehensive type definitions for dual search operations, context-aware generation, and document retrieval workflows

## Installation

```bash
# Install dependencies
poetry install

# Or using pip
pip install -r requirements.txt
```

## Quick Setup

If you're starting with an empty MCP configuration, see the [MCP Setup Guide](./MCP_SETUP_GUIDE.md) for step-by-step instructions.

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
poetry run python run_mcp_server_test.py

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
Generate ReactJS frontend components with automatic contextual search for GenericSuite frontend patterns.

```json
{
  "requirements": "Create a user profile form with validation",
  "include_ai": false
}
```

For AI-enhanced frontend code, set `include_ai` to `true` to search for AI integration examples.

### generate_backend_code
Generate backend API code with automatic contextual search for GenericSuite backend patterns.

```json
{
  "requirements": "Create CRUD endpoints for user management",
  "framework": "fastapi",
  "include_ai": false
}
```

For AI-enhanced backend code, set `include_ai` to `true` to search for AI integration examples.

### query_agent
Direct query interface to the AI agent.

```json
{
  "query": "What is GenericSuite?",
  "task_type": "general"
}
```

### retrieve_document_from_local_storage
Retrieve complete documents from local storage (available as Agent tool).

```json
{
  "document_path": "local_repo_files/genericsuite-basecamp/docs/Configuration-Guide/Generic-CRUD-Editor-Configuration.md"
}
```

This tool is automatically available to the Agent during code generation workflows and allows access to complete GenericSuite documentation and examples stored in the `local_repo_files` directory.

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

### MCP Client Configuration

#### Kiro Integration

For Kiro, add the server configuration to your `.kiro/settings/mcp.json` file:

```json
{
  "mcpServers": {
    "genericsuite-codegen": {
      "command": "sh",
      "args": ["/absolute/path/to/genericsuite-codegen/mcp-server/run_mcp_server.sh"],
      "env": {
        "MCP_API_KEY": "ag-api-key-...",
        "MCP_TRANSPORT": "stdio"
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

**Note**: Replace `/absolute/path/to/genericsuite-codegen` with the actual path to your project directory.

#### HTTP-based Integration

For HTTP-based MCP clients:

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

### Configuration Templates

There are configuration files for different MCP clients in the `mcp-server` directory:

- [claude_desktop_http_config.json](./claude_desktop_http_config.json) for Claude Desktop with HTTP transport
- [claude_desktop_stdio_config.json](./claude_desktop_stdio_config.json) for Claude Desktop with STDIO transport
- [vscode_mcp_http_config.json](./vscode_mcp_http_config.json) for VS Code with HTTP transport
- [vscode_mcp_stdio_config.json](./vscode_mcp_stdio_config.json) for VS Code with STDIO transport
- [kiro_stdio_config.json](./kiro_stdio_config.json) for Kiro with STDIO transport (Kiro is not supported with HTTP transport)

### Setting Up MCP Configuration

If your `.kiro/settings/mcp.json` file is empty or missing servers, you can:

1. **Copy from template**: Use one of the provided configuration templates as a starting point
2. **Manual setup**: Add the server configuration manually using the examples above
3. **Auto-configure**: Use the Kiro command palette to search for 'MCP' and configure servers

Remember to:
- Replace placeholder paths with actual absolute paths
- Set appropriate API keys if authentication is enabled
- Adjust `autoApprove` settings based on your security preferences

## Documentation

For detailed information about enhanced search capabilities:

- **[Enhanced Search Examples](../server/docs/ENHANCED_SEARCH_EXAMPLES.md)**: Examples of dual search results and document retrieval
- **[Configuration Guide](../server/docs/CONFIGURATION_GUIDE.md)**: Configuration options and template customization
- **[API Documentation](../server/docs/API_DOCUMENTATION.md)**: Complete API reference
- **[Troubleshooting Guide](../server/docs/TROUBLESHOOTING_GUIDE.md)**: Troubleshooting guide for enhanced search issues

## Troubleshooting

### Common Issues

1. **Agent not available**: Ensure the main server components are properly installed
2. **Database connection failed**: Check MongoDB connection and credentials
3. **API key errors**: Verify OpenAI API key is set correctly
4. **Port conflicts**: Change MCP_SERVER_PORT if 8070 is in use
5. **Enhanced search not working**: See the [Troubleshooting Guide](../server/docs/TROUBLESHOOTING_GUIDE.md) for detailed diagnostics

### Logs

Server logs are written to `mcp_server.log` and console output.

### Health Check

Check server health at runtime:
```bash
# Test server functionality
poetry run python run_mcp_server_test.py
```