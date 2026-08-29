# GenericSuite CodeGen API Documentation

## Overview

The GenericSuite CodeGen API provides enhanced vector search capabilities with dual search functionality that combines user queries with contextual GenericSuite rules. This ensures generated code consistently follows established patterns and conventions.

## Enhanced Search Features

### Dual Vector Search

The API automatically performs two types of searches for optimal code generation:

1. **User Query Search**: Direct search based on user requirements
2. **Contextual Rules Search**: Automatic search for relevant GenericSuite patterns based on code generation type

### Context-Aware Generation

The system automatically determines the appropriate contextual search based on the code generation context:

- **JSON Configuration**: Searches for GenericSuite table configuration patterns
- **LangChain Tools**: Searches for LangChain tool implementation patterns  
- **MCP Server Tools**: Searches for MCP tool implementation patterns
- **Frontend Code**: Searches for React/frontend implementation patterns
- **Backend Code**: Searches for FastAPI/backend implementation patterns
- **AI-Enhanced Code**: Searches for AI integration patterns

## API Endpoints

### Core Agent Endpoints

#### POST /query
Query the AI agent with enhanced search capabilities.

**Request Body:**
```json
{
  "query": "Create a user management table configuration",
  "task_type": "json_config",
  "conversation_id": "optional-conversation-id",
  "context": {
    "code_type": "json",
    "framework": null,
    "confidence": 0.9
  }
}
```

**Response:**
```json
{
  "response": "Generated response with contextual examples",
  "conversation_id": "conversation-uuid",
  "search_results": [
    {
      "content": "Relevant content snippet",
      "source": "document-path",
      "score": 0.95,
      "metadata": {
        "file_type": "json",
        "section": "table-configuration"
      }
    }
  ],
  "context_used": {
    "code_type": "json",
    "contextual_search": "examples of how to create a JSON table configuration files in Genericsuite",
    "documents_retrieved": ["path/to/config-example.json"]
  }
}
```

#### POST /query/stream
Stream AI agent query response with enhanced search.

**Request Body:** Same as `/query`

**Response:** Server-sent events stream with enhanced context information.

### Code Generation Endpoints

#### POST /generate/json-config
Generate JSON configuration with contextual GenericSuite patterns.

**Request Body:**
```json
{
  "requirements": "Create a user management table with name, email, role fields",
  "table_name": "users",
  "config_type": "table"
}
```

**Enhanced Search Behavior:**
- Automatically searches for: "examples of how to create a JSON table configuration files in Genericsuite"
- Retrieves complete configuration examples from local storage
- Prioritizes GenericSuite conventions over conflicting user requirements

**Response:**
```json
{
  "files": [
    {
      "filename": "users_table_config.json",
      "content": "{\n  \"table_name\": \"users\",\n  \"fields\": [...]\n}",
      "file_type": "json",
      "description": "User management table configuration following GenericSuite patterns"
    }
  ],
  "context_used": {
    "contextual_search": "examples of how to create a JSON table configuration files in Genericsuite",
    "documents_retrieved": [
      "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Configuration-Guide/Generic-CRUD-Editor-Configuration.md"
    ],
    "patterns_applied": ["table_structure", "field_validation", "ui_configuration"]
  }
}
```

#### POST /generate/python-code
Generate Python code with contextual GenericSuite patterns.

**Request Body:**
```json
{
  "requirements": "Create a Langchain tool for user authentication",
  "tool_name": "UserAuthTool",
  "description": "Tool for authenticating users",
  "type": "langchain"
}
```

**Enhanced Search Behavior:**
- For `type: "langchain"`: Searches for "examples of how to create a Python Langchain Tool in Genericsuite"
- For `type: "mcp"`: Searches for "examples of how to create a MCP server tool in Genericsuite"
- Retrieves complete tool implementations from local storage

#### POST /generate/frontend-code
Generate React frontend code with contextual patterns.

**Request Body:**
```json
{
  "requirements": "Create a user profile form with validation",
  "include_ai": false
}
```

**Enhanced Search Behavior:**
- Standard frontend: Searches for "examples of how to create frontend code in Genericsuite"
- AI-enhanced (`include_ai: true`): Searches for "examples of how to create frontend with AI code in Genericsuite"

#### POST /generate/backend-code
Generate backend code with contextual patterns.

**Request Body:**
```json
{
  "requirements": "Create CRUD endpoints for user management",
  "framework": "fastapi",
  "include_ai": false
}
```

**Enhanced Search Behavior:**
- Standard backend: Searches for "examples of how to create backend code in Genericsuite"
- AI-enhanced (`include_ai: true`): Searches for "examples of how to create backend with AI code in Genericsuite"

### Knowledge Base Endpoints

#### POST /search
Enhanced vector search with dual query capability.

**Request Body:**
```json
{
  "query": "table configuration examples",
  "limit": 10,
  "file_type_filter": "json",
  "enable_contextual_search": true,
  "code_context": {
    "code_type": "json",
    "framework": null
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "Configuration example content",
      "source": "document-path",
      "score": 0.95,
      "metadata": {
        "file_type": "json",
        "section": "configuration"
      }
    }
  ],
  "dual_search_results": {
    "user_results": [...],
    "context_results": [...],
    "merged_results": [...],
    "context_used": {
      "code_type": "json",
      "contextual_search": "examples of how to create a JSON table configuration files in Genericsuite"
    }
  },
  "documents_retrieved": [
    {
      "path": "local_repo_files/example.json",
      "content": "Complete document content",
      "metadata": {
        "size": 1024,
        "last_modified": "2024-01-01T00:00:00Z"
      }
    }
  ]
}
```

## Document Retrieval Tool

The API includes a comprehensive document retrieval tool that provides secure access to complete documents from local storage.

### Features

- **Secure Path Validation**: Prevents directory traversal attacks
- **File Type Detection**: Automatic detection and handling of different file types
- **Encoding Detection**: Smart encoding detection with fallback handling
- **Batch Operations**: Support for retrieving multiple documents
- **Rich Metadata**: Complete file information including size, modification time, and encoding

### Usage in Agent Context

The document retrieval tool is automatically available to the AI agent during code generation workflows. When the agent needs complete document content, it can retrieve files from the `local_repo_files` directory.

**Example Agent Tool Call:**
```json
{
  "tool": "retrieve_document_from_local_storage",
  "parameters": {
    "document_path": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Configuration-Guide/Generic-CRUD-Editor-Configuration.md"
  }
}
```

**Tool Response:**
```json
{
  "path": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Configuration-Guide/Generic-CRUD-Editor-Configuration.md",
  "content": "Complete document content...",
  "file_type": "md",
  "size": 15420,
  "last_modified": "2024-01-01T00:00:00Z",
  "encoding": "utf-8",
  "metadata": {
    "title": "Generic CRUD Editor Configuration",
    "section": "Configuration Guide"
  }
}
```

## Error Handling

### Enhanced Search Errors

The API includes comprehensive error handling for enhanced search operations:

```json
{
  "error_code": "DUAL_SEARCH_ERROR",
  "message": "Contextual search failed, continuing with user query results",
  "details": {
    "user_search_status": "success",
    "contextual_search_status": "failed",
    "fallback_applied": true
  },
  "correlation_id": "req-12345"
}
```

### Document Retrieval Errors

```json
{
  "error_code": "DOCUMENT_RETRIEVAL_ERROR",
  "message": "Document not found in local storage",
  "details": {
    "error_type": "FILE_NOT_FOUND",
    "requested_path": "invalid/path.md",
    "available_alternatives": ["similar/path.md"]
  }
}
```

### Error Codes

- `VALIDATION_ERROR`: Request validation failed
- `DUAL_SEARCH_ERROR`: Enhanced search operation failed
- `CONTEXT_DETERMINATION_ERROR`: Context determination failed
- `DOCUMENT_RETRIEVAL_ERROR`: Document retrieval failed
- `TEMPLATE_LOAD_ERROR`: Search template loading failed
- `SEARCH_MERGE_ERROR`: Search result merging failed

## Configuration

### Search Templates

Search templates can be configured via `server/genericsuite_codegen/config/search_templates.json`:

```json
{
  "templates": {
    "json": {
      "template": "examples of how to create a JSON table configuration files in Genericsuite",
      "file_type_filter": "json",
      "priority": 1
    },
    "langchain": {
      "template": "examples of how to create a Python Langchain Tool in Genericsuite",
      "file_type_filter": "py",
      "priority": 1
    },
    "mcp": {
      "template": "examples of how to create a MCP server tool in Genericsuite",
      "file_type_filter": "py",
      "priority": 1
    },
    "frontend": {
      "template": "examples of how to create frontend code in Genericsuite",
      "file_type_filter": "jsx",
      "priority": 1
    },
    "frontend_ai": {
      "template": "examples of how to create frontend with AI code in Genericsuite",
      "file_type_filter": "jsx",
      "priority": 1
    },
    "backend": {
      "template": "examples of how to create backend code in Genericsuite",
      "file_type_filter": "py",
      "priority": 1
    },
    "backend_ai": {
      "template": "examples of how to create backend with AI code in Genericsuite",
      "file_type_filter": "py",
      "priority": 1
    }
  }
}
```

### Environment Variables

```bash
# Enhanced Search Configuration
ENHANCED_SEARCH_ENABLED=true
ENHANCED_SEARCH_MAX_CONTEXT_LENGTH=10000
ENHANCED_SEARCH_FALLBACK_ENABLED=true

# Document Retrieval Configuration
LOCAL_REPO_PATH=local_repo_files
DOCUMENT_RETRIEVAL_MAX_SIZE=10485760  # 10MB
DOCUMENT_RETRIEVAL_TIMEOUT=30

# Search Template Configuration
SEARCH_TEMPLATES_CONFIG_PATH=server/genericsuite_codegen/config/search_templates.json
SEARCH_TEMPLATES_RELOAD_ENABLED=true
```

## Performance Considerations

### Caching

- Search results are cached for frequently accessed queries
- Document content is cached for recently retrieved files
- Template configurations are cached and reloaded only when changed

### Optimization

- Dual searches are performed in parallel when possible
- File type filters are applied to reduce search scope
- Document retrieval uses streaming for large files
- Batch operations are optimized for multiple document requests

## Security

### Path Validation

All document retrieval operations include comprehensive path validation:

- Prevents directory traversal attacks (`../` sequences)
- Restricts access to the `local_repo_files` directory
- Validates file extensions and types
- Logs all access attempts for audit purposes

### API Security

- Request validation using Pydantic models
- Rate limiting for search operations
- Correlation IDs for request tracking
- Comprehensive error logging without exposing sensitive information

## Monitoring and Observability

### Metrics

The API tracks the following metrics:

- Dual search success/failure rates
- Document retrieval performance
- Context determination accuracy
- Template loading success rates
- API response times

### Logging

Comprehensive logging includes:

- Request/response correlation IDs
- Enhanced search operation details
- Document retrieval attempts and results
- Error conditions with context
- Performance metrics

### Health Checks

Health check endpoints provide status for:

- Enhanced search components
- Document retrieval tool
- Search template manager
- Database connectivity
- AI agent availability

## Migration Guide

### From Basic Search to Enhanced Search

Enhanced search is automatically enabled and backward compatible. Existing API calls will benefit from enhanced search without changes.

### Configuration Migration

If you have custom search configurations, update them to use the new template format:

**Old Format:**
```json
{
  "search_queries": {
    "json": "table configuration examples"
  }
}
```

**New Format:**
```json
{
  "templates": {
    "json": {
      "template": "examples of how to create a JSON table configuration files in Genericsuite",
      "file_type_filter": "json",
      "priority": 1
    }
  }
}
```