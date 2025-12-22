# GenericSuite CodeGen Server

This is the backend server for the GenericSuite CodeGen RAG AI system, featuring enhanced vector search capabilities and context-aware code generation.

## Features

- **Enhanced Vector Search**: Dual search combining user queries with contextual GenericSuite rules
- **Knowledge Base Search**: Vector similarity search in the GenericSuite documentation
- **Context-Aware Generation**: Automatic search for relevant examples based on code generation type:
  - JSON table configuration files
  - Python LangChain tools
  - MCP server tools
  - Frontend code (including AI-enhanced)
  - Backend code (including AI-enhanced)
- **Enhanced Search Types**: Comprehensive type definitions for dual search operations:
  - `CodeGenerationContext`: Context information for code generation
  - `DualSearchResult`: Results from dual vector search operations
  - `DocumentContent`: Complete document content with metadata
  - `SearchTemplate`: Configurable search templates for different code types
  - `EnhancedSearchConfig`: Configuration for enhanced search functionality
- **Search Template Manager**: Configurable search templates with file-based configuration and validation
- **Intelligent Knowledge Base**: MongoDB vector search with sentence transformers embeddings
- **Pydantic AI Agent**: Advanced AI agent with tool integration
- **Document Retrieval Tool**: Secure agent tool for accessing complete GenericSuite documents from local storage with path validation and error handling
- **FastAPI Backend**: High-performance API with streaming support

## Type Definitions

The server includes comprehensive type definitions for AI agent operations:

### Basic Agent Types (`server/genericsuite_codegen/agent/types.py`)

- **`KnowledgeBaseQuery`**: Query model for knowledge base search operations
- **`SearchResultModel`**: Model for search results with source attribution

### Enhanced Search Types (`server/genericsuite_codegen/agent/enhanced_search_types.py`)

The server includes comprehensive type definitions for enhanced vector search operations:

### Core Data Models

- **`CodeGenerationContext`**: Context information including code type, framework, confidence score, and detected patterns
- **`DualSearchResult`**: Results from dual vector search combining user queries with contextual rules
- **`DocumentContent`**: Complete document content with metadata, file type, and encoding information
- **`DocumentMetadata`**: Lightweight document metadata for efficient operations
- **`SearchTemplate`**: Configurable search templates with priority and file type filters
- **`EnhancedSearchConfig`**: Configuration for enhanced search functionality with merge strategies

### API Models

- **`EnhancedSearchQuery`**: Request model for enhanced search operations with dual search options
- **`EnhancedSearchResponse`**: Response model including formatted context and retrieved documents
- **`DocumentRetrievalRequest/Response`**: Models for single document retrieval operations
- **`BatchDocumentRetrievalRequest/Response`**: Models for batch document retrieval operations

### Exception Hierarchy

- **`EnhancedSearchError`**: Base exception for enhanced search operations
- **`ContextDeterminationError`**: Context determination failures
- **`DocumentRetrievalError`**: Document retrieval failures with error codes
- **`TemplateLoadError`**: Template loading failures
- **`DualSearchError`**: Dual search operation failures
- **`SearchMergeError`**: Search result merging failures

## Document Retrieval Tool

The server includes a comprehensive document retrieval tool (`DocumentRetrievalTool`) that provides secure access to complete documents from the local knowledge base:

### Features

- **Secure Path Validation**: Prevents directory traversal attacks with proper path sanitization
- **File Type Detection**: Automatic detection of file types and binary content
- **Encoding Detection**: Smart encoding detection with fallback to UTF-8
- **Batch Operations**: Support for retrieving multiple documents in a single operation
- **Error Handling**: Comprehensive error handling with specific error codes
- **Metadata Extraction**: Rich metadata including file size, modification time, and encoding

### Usage

The document retrieval tool is automatically available to the AI agent during code generation workflows:

```python
from genericsuite_codegen.agent.document_retrieval_tool import DocumentRetrievalTool

# Initialize the tool
doc_tool = DocumentRetrievalTool()

# Retrieve a single document
document = doc_tool.retrieve_document("path/to/document.md")

# Retrieve multiple documents
documents = doc_tool.retrieve_multiple_documents([
    "path/to/doc1.md",
    "path/to/doc2.py"
])

# Get document metadata without full content
metadata = doc_tool.get_document_metadata("path/to/document.md")
```

### Security Features

- **Path Validation**: All paths are validated to prevent access outside the `local_repo_files` directory
- **Binary File Detection**: Binary files are detected and rejected to prevent issues
- **Size Limits**: Configurable size limits for document retrieval
- **Error Codes**: Specific error codes for different failure scenarios (`FILE_NOT_FOUND`, `PATH_TRAVERSAL`, `BINARY_FILE`, etc.)

## Search Template Manager

The server includes a configurable search template manager (`SearchTemplateManager`) that provides flexible template management for different code generation types:

### Features

- **File-Based Configuration**: Templates can be configured via JSON files
- **Hardcoded Fallbacks**: Built-in default templates ensure system reliability
- **Dynamic Reloading**: Templates can be reloaded without restarting the server
- **Template Validation**: Comprehensive validation of template structure and content
- **Priority System**: Templates can be prioritized for better search results

### Configuration

Templates are configured in `server/genericsuite_codegen/config/search_templates.json`:

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
    }
  }
}
```

### Usage

```python
from genericsuite_codegen.agent.search_templates import SearchTemplateManager

# Initialize with configuration file
template_manager = SearchTemplateManager("config/search_templates.json")

# Get template for specific code type
template = template_manager.get_template("json")

# Get file type filter
file_filter = template_manager.get_file_type_filter("json")

# Reload templates dynamically
template_manager.reload_templates()
```

## Code Generation Types

The server automatically performs contextual searches based on the type of code being generated:

- **JSON Configuration**: Searches for "examples of how to create a JSON table configuration files in Genericsuite"
- **LangChain Tools**: Searches for "examples of how to create a Python Langchain Tool in Genericsuite"
- **MCP Server Tools**: Searches for "examples of how to create a MCP server tool in Genericsuite"
- **Frontend Code**: Searches for "examples of how to create frontend code in Genericsuite"
- **Backend Code**: Searches for "examples of how to create backend code in Genericsuite"
- **Frontend with AI**: Searches for "examples of how to create frontend with AI code in Genericsuite"
- **Backend with AI**: Searches for "examples of how to create backend with AI code in Genericsuite"

## Installation

```bash
poetry install
```

## Usage

```bash
poetry run python -m genericsuite_codegen
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[API Documentation](docs/API_DOCUMENTATION.md)**: Complete API reference with enhanced search capabilities
- **[Enhanced Search Examples](docs/ENHANCED_SEARCH_EXAMPLES.md)**: Examples of dual search results and document retrieval
- **[Configuration Guide](docs/CONFIGURATION_GUIDE.md)**: Configuration options and template customization
- **[Troubleshooting Guide](docs/TROUBLESHOOTING_GUIDE.md)**: Troubleshooting guide for enhanced search issues

## Development

```bash
# Start development server
make dev

# Run tests
make test

# Format and lint code
make format && make lint
```