# GenericSuite CodeGen MCP Server

This module implements a comprehensive FastMCP server that exposes the GenericSuite CodeGen AI agent capabilities as standardized MCP (Model Context Protocol) tools and resources.

## Features Implemented

### ✅ MCP Tools (6 tools)

1. **search_knowledge_base**
   - Search the GenericSuite documentation and knowledge base
   - Parameters: `query` (string), `limit` (int, default: 5)
   - Returns: Structured search results with content, sources, and similarity scores

2. **generate_json_config**
   - Generate GenericSuite table configuration JSON based on requirements
   - Parameters: `requirements` (string)
   - Returns: Complete JSON configuration following GenericSuite patterns

3. **generate_langchain_tool**
   - Generate Langchain Tool Python code based on specification
   - Parameters: `specification` (string)
   - Returns: Complete Python code for a Langchain Tool

4. **generate_mcp_tool**
   - Generate MCP Tool Python code based on specification
   - Parameters: `specification` (string)
   - Returns: Complete Python code for an MCP Tool

5. **generate_frontend_code**
   - Generate ReactJS frontend code following GenericSuite patterns
   - Parameters: `requirements` (string)
   - Returns: Multiple files (App.tsx, package.json, CSS, etc.)

6. **generate_backend_code**
   - Generate backend code for specified framework
   - Parameters: `framework` (fastapi|flask|chalice), `requirements` (string)
   - Returns: Complete backend application files

### ✅ MCP Resources (3 resources)

1. **genericsuite://knowledge_base_stats**
   - Provides statistics about the knowledge base
   - Returns: Document count, status, last updated timestamp

2. **genericsuite://server_info**
   - Provides information about the MCP server
   - Returns: Server name, version, status, capabilities, supported frameworks

3. **genericsuite://agent_capabilities**
   - Provides detailed information about available tools and resources
   - Returns: Complete list of tools with parameters and descriptions

### ✅ Authentication & Security

- **API Key Authentication**: Configurable API key-based authentication
- **Request Validation**: Validates incoming MCP requests
- **Security Logging**: Logs authentication attempts and security events
- **Environment Configuration**: Secure configuration through environment variables

### ✅ Error Handling

- **Centralized Error Handling**: `_handle_error()` method for consistent error responses
- **Error Correlation IDs**: Unique error IDs for tracking and debugging
- **Structured Error Responses**: Consistent error format with context and type information
- **Comprehensive Logging**: Detailed error logging with stack traces
- **Graceful Degradation**: Fallback responses when components fail

### ✅ Real Agent Integration

- **Knowledge Base Search**: Integrates with actual KnowledgeBaseTool for searches
- **AI Agent Integration**: Uses real GenericSuiteAgent for code generation
- **Database Integration**: Connects to actual MongoDB vector database
- **Embedding Models**: Supports both OpenAI and HuggingFace embeddings

## Architecture

```
MCP Client (Kiro, etc.)
    ↓ (MCP Protocol)
FastMCP Server
    ↓
GenericSuiteMCPServer
    ├── Authentication Layer
    ├── Error Handling Layer
    ├── MCP Tools (6)
    ├── MCP Resources (3)
    └── Agent Integration
        ├── GenericSuiteAgent (Pydantic AI)
        ├── KnowledgeBaseTool
        ├── DatabaseManager (MongoDB)
        └── Embedding Providers
```

## Configuration

The MCP server is configured through the `MCPConfig` dataclass:

```python
@dataclass
class MCPConfig:
    server_name: str = "genericsuite-codegen"
    server_version: str = "1.0.0"
    api_key: Optional[str] = None  # Enable authentication
    host: str = "0.0.0.0"
    port: int = 8070
    debug: bool = False
```

## Usage

### Starting the Server

```bash
cd mcp-server
python start_mcp_server.py
```

### Environment Variables

- `MCP_SERVER_NAME`: Server name (default: "genericsuite-codegen")
- `MCP_SERVER_VERSION`: Server version (default: "1.0.0")
- `MCP_API_KEY`: API key for authentication (optional)
- `MCP_SERVER_HOST`: Host address (default: "0.0.0.0")
- `MCP_SERVER_PORT`: Port number (default: 8070)
- `MCP_DEBUG`: Debug mode (default: "0")

### Integration with Kiro

The MCP server can be integrated with Kiro IDE by adding it to the MCP configuration:

```json
{
  "mcpServers": {
    "genericsuite-codegen": {
      "command": "python",
      "args": ["/path/to/mcp-server/start_mcp_server.py"],
      "env": {
        "MCP_API_KEY": "your-api-key-here"
      },
      "disabled": false,
      "autoApprove": [
        "search_knowledge_base",
        "generate_json_config"
      ]
    }
  }
}
```

## Error Handling Examples

All tools return structured error responses:

```json
{
  "success": false,
  "error": {
    "id": "mcp_error_1234",
    "message": "AI agent not initialized",
    "context": "search_knowledge_base",
    "type": "Exception"
  }
}
```

## Security Features

1. **API Key Validation**: Optional API key authentication
2. **Request Validation**: Validates all incoming requests
3. **Error Sanitization**: Prevents sensitive information leakage
4. **Audit Logging**: Comprehensive logging for security monitoring
5. **Input Sanitization**: Validates and sanitizes all user inputs

## Performance Features

1. **Async Operations**: All operations are asynchronous for better performance
2. **Connection Pooling**: Database connection pooling for efficiency
3. **Caching**: Embedding model caching to avoid reloading
4. **Error Recovery**: Graceful error recovery and fallback mechanisms

## Compliance

- **MCP Protocol**: Fully compliant with MCP specification
- **FastMCP Framework**: Uses FastMCP for standardized implementation
- **Type Safety**: Full type hints and Pydantic validation
- **Documentation**: Comprehensive docstrings and API documentation

## Testing

The server can be tested by:

1. **Import Testing**: `python -c "from genericsuite_codegen.mcp_server import create_mcp_server, MCPConfig; print('Success')"`
2. **Server Creation**: Create server instance with configuration
3. **Tool Testing**: Test individual tools through MCP protocol
4. **Integration Testing**: Test with actual MCP clients like Kiro

## Monitoring

The server provides comprehensive logging:

- Component initialization
- Authentication events
- Tool execution
- Error tracking
- Performance metrics

All logs include correlation IDs for tracking requests across the system.