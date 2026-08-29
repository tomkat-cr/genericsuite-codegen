# Enhanced Vector Search Implementation

## Overview

The GenericSuite CodeGen system implements an enhanced vector search approach that combines user queries with contextual GenericSuite rules and examples. This dual search strategy ensures generated code consistently follows established patterns and conventions.

**Implementation Status**: Enhanced search components have been implemented:
- `server/genericsuite_codegen/agent/enhanced_search_types.py`: Comprehensive data models, API types, and exception handling for dual search operations
- `server/genericsuite_codegen/agent/document_retrieval_tool.py`: Secure document retrieval tool with path validation and error handling
- `server/genericsuite_codegen/agent/search_templates.py`: Configurable search template manager with file-based configuration and validation

## Search Strategy

### Dual Vector Search Approach

1. **User Query Search**: Direct search based on user requirements
2. **Contextual Rules Search**: Automatic search for relevant GenericSuite patterns based on code generation type

### Context-Aware Generation Types

The system automatically performs contextual searches based on the type of code being generated:

#### JSON Configuration
- **Search Query**: "examples of how to create a JSON table configuration files in Genericsuite"
- **Purpose**: Retrieve GenericSuite table configuration patterns and examples
- **File Filter**: JSON files

#### Python LangChain Tools
- **Search Query**: "examples of how to create a Python Langchain Tool in Genericsuite"
- **Purpose**: Retrieve LangChain tool implementation patterns
- **File Filter**: Python files

#### MCP Server Tools
- **Search Query**: "examples of how to create a MCP server tool in Genericsuite"
- **Purpose**: Retrieve MCP tool implementation patterns
- **File Filter**: Python files

#### Frontend Code
- **Search Query**: "examples of how to create frontend code in Genericsuite"
- **Purpose**: Retrieve React/frontend implementation patterns
- **File Filter**: JSX/TSX files

#### Backend Code
- **Search Query**: "examples of how to create backend code in Genericsuite"
- **Purpose**: Retrieve FastAPI/backend implementation patterns
- **File Filter**: Python files

#### AI-Enhanced Frontend
- **Search Query**: "examples of how to create frontend with AI code in Genericsuite"
- **Purpose**: Retrieve AI-integrated frontend patterns
- **File Filter**: JSX/TSX files

#### AI-Enhanced Backend
- **Search Query**: "examples of how to create backend with AI code in Genericsuite"
- **Purpose**: Retrieve AI-integrated backend patterns
- **File Filter**: Python files

## Implementation Details

### Search Process

1. **Query Validation**: Validate user input (3-1000 characters)
2. **Context Determination**: Determine code generation type from request
3. **Dual Search Execution**: 
   - Execute user query search
   - Execute contextual rules search
4. **Result Merging**: Combine and prioritize results from both searches
5. **Context Formatting**: Format retrieved context for code generation
6. **Fallback Handling**: Use user query results only if contextual search fails

### Priority Rules

- GenericSuite conventions take priority over conflicting user requirements
- Contextual rules are merged with user requirements when compatible
- System continues with user query results if contextual search fails
- Local document storage is used for full content retrieval

### Document Retrieval Tool

The Agent includes a comprehensive document retrieval tool (`DocumentRetrievalTool`) for secure access to complete documents:

- **Implementation**: `server/genericsuite_codegen/agent/document_retrieval_tool.py`
- **Purpose**: Retrieve complete document content from the `local_repo_files` directory
- **Security Features**: 
  - Path validation to prevent directory traversal attacks
  - Binary file detection and rejection
  - Encoding detection with fallback handling
- **Capabilities**:
  - Single document retrieval with full metadata
  - Batch document retrieval for multiple files
  - Lightweight metadata-only queries
- **Error Handling**: Comprehensive error handling with specific error codes (`FILE_NOT_FOUND`, `PATH_TRAVERSAL`, `BINARY_FILE`, etc.)
- **Integration**: Automatically available to the Agent during code generation workflows

### Error Handling

- Missing documents are logged but don't block generation
- File access errors are handled gracefully with specific error messages
- Search failures fallback to original behavior
- Enhanced search failures don't impact core functionality
- Document retrieval failures are reported to the Agent for alternative approaches

## Configuration

### Search Templates

Search templates are managed by the `SearchTemplateManager` class and can be configured via:

**Configuration File**: `server/genericsuite_codegen/config/search_templates.json`
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
    }
  }
}
```

**Features**:
- File-based configuration with JSON validation
- Hardcoded fallback templates for reliability
- Dynamic template reloading without restart
- Template validation and error handling
- Support for custom code types and priorities

### File Type Filters

Appropriate file type filters are applied based on code generation type:

```python
FILE_TYPE_FILTERS = {
    "json": "json",
    "python": "py", 
    "frontend": "jsx",
    "backend": "py"
}
```

## Benefits

1. **Consistency**: Generated code follows GenericSuite patterns
2. **Quality**: Leverages proven examples and best practices
3. **Context-Awareness**: Automatically finds relevant examples
4. **Flexibility**: Maintains user requirement support
5. **Reliability**: Graceful fallback to original behavior
6. **Performance**: Optimized search with appropriate filters

## Usage

The enhanced vector search is automatically applied to all code generation requests through existing API endpoints. No configuration changes are required for basic usage.

### API Integration

All existing endpoints benefit from enhanced search:
- `/api/generate/json-config`
- `/api/generate/python-code`
- `/api/generate/frontend-code`
- `/api/generate/backend-code`
- `/api/chat/query`

### MCP Integration

MCP tools automatically use enhanced search:
- `generate_json_config`
- `generate_python_code`
- `generate_frontend_code`
- `generate_backend_code`
- `search_knowledge_base`