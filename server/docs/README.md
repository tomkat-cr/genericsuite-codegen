# GenericSuite CodeGen Documentation

This directory contains comprehensive documentation for the GenericSuite CodeGen enhanced vector search capabilities.

## Documentation Overview

### Core Documentation

- **[API Documentation](API_DOCUMENTATION.md)**: Complete API reference with enhanced search capabilities
  - Dual vector search endpoints
  - Document retrieval tool usage
  - Context-aware generation examples
  - Error handling and response formats

- **[Enhanced Search Examples](ENHANCED_SEARCH_EXAMPLES.md)**: Practical examples and use cases
  - Dual search result examples
  - Document retrieval workflows
  - Context-aware generation scenarios
  - Error handling examples
  - Integration patterns

- **[Configuration Guide](CONFIGURATION_GUIDE.md)**: Configuration options and customization
  - Search template configuration
  - Document retrieval settings
  - Environment variables
  - Performance tuning
  - Security configuration

- **[Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)**: Diagnostic and resolution procedures
  - Common issues and solutions
  - Error code reference
  - Performance troubleshooting
  - Debug mode usage
  - Recovery procedures

## Quick Start

### Basic Usage

1. **Enable Enhanced Search:**
```bash
echo "ENHANCED_SEARCH_ENABLED=true" >> .env
```

2. **Test Dual Search:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "table configuration example",
    "enable_contextual_search": true,
    "code_context": {"code_type": "json"}
  }'
```

3. **Generate Code with Context:**
```bash
curl -X POST http://localhost:8000/generate/json-config \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": "Create a user management table",
    "table_name": "users",
    "config_type": "table"
  }'
```

### Key Features

- **Dual Vector Search**: Combines user queries with contextual GenericSuite rules
- **Document Retrieval Tool**: Secure access to complete documents from local storage
- **Context-Aware Generation**: Automatic detection of code generation type
- **Configurable Templates**: Customizable search templates for different code types
- **Error Recovery**: Graceful fallback to basic search when enhanced features fail

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced Agent Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Dual Vector   │  │   Context       │  │   Document      │ │
│  │   Search        │  │   Determination │  │   Retrieval     │ │
│  │   Engine        │  │   Service       │  │   Tool          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Existing Agent Core                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Knowledge     │  │   MongoDB       │  │   Local File    │ │
│  │   Base Tool     │  │   Vector DB     │  │   Storage       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Code Generation Types

The system automatically performs contextual searches based on the type of code being generated:

| Code Type | Contextual Search Query | File Filter |
|-----------|------------------------|-------------|
| JSON Configuration | "examples of how to create a JSON table configuration files in Genericsuite" | json |
| LangChain Tools | "examples of how to create a Python Langchain Tool in Genericsuite" | py |
| MCP Server Tools | "examples of how to create a MCP server tool in Genericsuite" | py |
| Frontend Code | "examples of how to create frontend code in Genericsuite" | jsx |
| Backend Code | "examples of how to create backend code in Genericsuite" | py |
| AI-Enhanced Frontend | "examples of how to create frontend with AI code in Genericsuite" | jsx |
| AI-Enhanced Backend | "examples of how to create backend with AI code in Genericsuite" | py |

## Implementation Status

### ✅ Completed Components

- **Enhanced Search Types**: Comprehensive data models and type definitions
- **Document Retrieval Tool**: Secure document access with path validation
- **Search Template Manager**: Configurable templates with file-based configuration
- **Context Determination Service**: Automatic context detection from user queries
- **Enhanced Vector Search Engine**: Dual search with result merging
- **Agent Integration**: Tool registration and configuration
- **Error Handling**: Comprehensive error handling with specific error codes
- **Configuration System**: Flexible configuration with validation
- **Testing Suite**: Unit and integration tests for all components

### 📋 Documentation Components

- **API Documentation**: Complete endpoint reference with examples
- **Configuration Guide**: Detailed configuration options and customization
- **Examples Documentation**: Practical usage examples and integration patterns
- **Troubleshooting Guide**: Diagnostic procedures and common issue resolution

## Integration Examples

### MCP Server Integration

```json
{
  "method": "tools/call",
  "params": {
    "name": "generate_json_config",
    "arguments": {
      "requirements": "Create a product catalog table",
      "table_name": "products",
      "config_type": "table"
    }
  }
}
```

### API Integration

```python
import requests

response = requests.post("http://localhost:8000/generate/json-config", json={
    "requirements": "Create a user management table with authentication",
    "table_name": "users",
    "config_type": "table"
})

result = response.json()
print(f"Generated configuration: {result['files'][0]['content']}")
```

### Agent Tool Usage

The document retrieval tool is automatically available to the AI agent:

```python
# Agent automatically uses document retrieval during code generation
agent_response = await agent.query(
    "Create a table configuration following GenericSuite patterns"
)
# Agent will automatically retrieve relevant documents from local_repo_files
```

## Security Considerations

- **Path Validation**: All document paths are validated to prevent directory traversal
- **File Type Filtering**: Only allowed file types can be retrieved
- **Size Limits**: Configurable limits on document size and retrieval frequency
- **Error Sanitization**: Error messages don't expose sensitive file system information

## Performance Features

- **Parallel Processing**: Dual searches can be performed in parallel
- **Caching**: Search results and document content are cached
- **Batch Operations**: Support for retrieving multiple documents efficiently
- **Resource Management**: Configurable limits on memory and processing resources

## Monitoring and Observability

- **Health Checks**: Comprehensive health monitoring for all components
- **Metrics**: Performance metrics for search operations and document retrieval
- **Logging**: Detailed logging with correlation IDs for debugging
- **Debug Mode**: Enhanced logging for troubleshooting

## Getting Help

### Common Issues

1. **Enhanced search not working**: Check [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md#dual-search-not-working)
2. **Document retrieval failing**: See [Document Retrieval Issues](TROUBLESHOOTING_GUIDE.md#document-retrieval-failures)
3. **Poor search quality**: Review [Context Determination Issues](TROUBLESHOOTING_GUIDE.md#context-determination-issues)
4. **Performance problems**: Check [Performance Issues](TROUBLESHOOTING_GUIDE.md#performance-issues)

### Debug Information

When reporting issues, include:

- Error logs from the server
- Configuration files (without sensitive data)
- System information and versions
- Steps to reproduce the issue

### Support Resources

- **Documentation**: Complete guides in this directory
- **Examples**: Practical usage examples in [Enhanced Search Examples](ENHANCED_SEARCH_EXAMPLES.md)
- **Configuration**: Detailed setup in [Configuration Guide](CONFIGURATION_GUIDE.md)
- **Troubleshooting**: Issue resolution in [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)

## Contributing

When contributing to the enhanced search functionality:

1. **Follow Patterns**: Use existing patterns for new components
2. **Add Tests**: Include unit and integration tests
3. **Update Documentation**: Keep documentation current with changes
4. **Security Review**: Ensure security best practices are followed
5. **Performance Testing**: Verify performance impact of changes