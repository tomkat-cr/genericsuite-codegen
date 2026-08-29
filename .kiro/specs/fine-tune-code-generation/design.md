# Design Document

## Overview

This design enhances the GenericSuite CodeGen system to implement fine-tuned code generation that respects GenericSuite's rules and patterns. The system will perform dual vector searches (user query + contextual GenericSuite rules) and provide an Agent tool for retrieving complete documents from local storage to ensure generated code follows established conventions.

## Architecture

### High-Level Architecture

The enhanced system builds upon the existing Pydantic AI agent architecture with the following key components:

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

### Core Enhancement Components

1. **Enhanced Vector Search Engine**: Performs dual searches based on user query and contextual GenericSuite rules
2. **Context Determination Service**: Automatically determines the appropriate contextual search based on code generation type
3. **Document Retrieval Tool**: Agent tool for accessing complete documents from local storage
4. **Search Template Manager**: Configurable templates for different code generation contexts

## Components and Interfaces

### 1. Enhanced Vector Search Engine

**Location**: `server/genericsuite_codegen/agent/enhanced_search.py`

```python
from .enhanced_search_types import CodeGenerationContext, DualSearchResult

class EnhancedVectorSearch:
    """Enhanced vector search with dual query capability."""
    
    def __init__(self, kb_tool: KnowledgeBaseTool, template_manager: SearchTemplateManager):
        self.kb_tool = kb_tool
        self.template_manager = template_manager
    
    async def dual_search(
        self, 
        user_query: str, 
        code_context: CodeGenerationContext,
        max_context_length: int = 10000
    ) -> DualSearchResult:
        """Perform dual search: user query + contextual rules."""
        pass
    
    def merge_search_results(
        self, 
        user_results: List[SearchResult], 
        context_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Merge and prioritize results from both searches."""
        pass
```

### 2. Context Determination Service

**Location**: `server/genericsuite_codegen/agent/context_determination.py`

```python
from .enhanced_search_types import CodeGenerationContext

class ContextDeterminationService:
    """Service to determine code generation context from user queries."""
    
    def determine_context(self, user_query: str, task_type: str) -> CodeGenerationContext:
        """Determine the appropriate context for code generation."""
        pass
    
    def get_contextual_search_query(self, context: CodeGenerationContext) -> str:
        """Get the appropriate contextual search query for the context."""
        pass
```

### 3. Document Retrieval Tool

**Location**: `server/genericsuite_codegen/agent/document_retrieval_tool.py`

```python
from .enhanced_search_types import DocumentContent, DocumentMetadata

class DocumentRetrievalTool:
    """Agent tool for retrieving complete documents from local storage."""
    
    def __init__(self, local_repo_path: str = "local_repo_files"):
        self.local_repo_path = local_repo_path
    
    def retrieve_document(self, document_path: str) -> DocumentContent:
        """Retrieve complete document content from local storage."""
        pass
    
    def retrieve_multiple_documents(self, document_paths: List[str]) -> List[DocumentContent]:
        """Retrieve multiple documents from local storage."""
        pass
    
    def get_document_metadata(self, document_path: str) -> DocumentMetadata:
        """Get metadata for a document without retrieving full content."""
        pass
```

### 4. Search Template Manager

**Location**: `server/genericsuite_codegen/agent/search_templates.py`

```python
from .enhanced_search_types import SearchTemplate, EnhancedSearchConfig

class SearchTemplateManager:
    """Manager for configurable search templates."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.templates = self._load_templates(config_path)
    
    def get_template(self, code_type: str) -> str:
        """Get search template for specific code type."""
        pass
    
    def reload_templates(self) -> None:
        """Reload templates from configuration."""
        pass
```

## Data Models

### Type Definitions

**Location**: `server/genericsuite_codegen/agent/enhanced_search_types.py`

```python
@dataclass
class CodeGenerationContext:
    """Context information for code generation."""
    code_type: str  # json, langchain, mcp, frontend, backend
    framework: Optional[str]  # fastapi, react, etc.
    confidence: float  # confidence in context determination

@dataclass
class DualSearchResult:
    """Result from dual vector search."""
    user_results: List[SearchResult]
    context_results: List[SearchResult]
    merged_results: List[SearchResult]
    context_used: CodeGenerationContext
    
@dataclass
class DocumentContent:
    """Complete document content from local storage."""
    path: str
    content: str
    file_type: str
    size: int
    last_modified: datetime
    metadata: Dict[str, Any]

@dataclass
class DocumentMetadata:
    """Document metadata without full content."""
    path: str
    file_type: str
    size: int
    last_modified: datetime
    exists: bool

@dataclass
class SearchTemplate:
    """Search template configuration."""
    code_type: str
    template: str
    file_type_filter: Optional[str]
    priority: int

@dataclass
class EnhancedSearchConfig:
    """Configuration for enhanced search functionality."""
    templates: Dict[str, SearchTemplate]
    local_repo_path: str
    max_context_length: int
    fallback_enabled: bool
```

## Error Handling

### Exception Hierarchy

```python
class EnhancedSearchError(Exception):
    """Base exception for enhanced search operations."""
    pass

class ContextDeterminationError(EnhancedSearchError):
    """Raised when context determination fails."""
    pass

class DocumentRetrievalError(EnhancedSearchError):
    """Raised when document retrieval fails."""
    pass

class TemplateLoadError(EnhancedSearchError):
    """Raised when template loading fails."""
    pass
```

### Error Handling Strategy

1. **Graceful Degradation**: If enhanced search fails, fall back to original search behavior
2. **Partial Results**: Continue with available results if some operations fail
3. **Logging**: Comprehensive logging for debugging and monitoring
4. **User Feedback**: Clear error messages without exposing internal details

## Testing Strategy

### Unit Tests

1. **Enhanced Vector Search Engine**
   - Test dual search functionality
   - Test result merging and prioritization
   - Test fallback behavior

2. **Context Determination Service**
   - Test context detection for different query types
   - Test confidence scoring
   - Test edge cases and ambiguous queries

3. **Document Retrieval Tool**
   - Test document retrieval from local storage
   - Test error handling for missing files
   - Test metadata extraction

4. **Search Template Manager**
   - Test template loading and validation
   - Test template reloading
   - Test fallback templates

### Integration Tests

1. **End-to-End Code Generation**
   - Test complete workflow from query to generated code
   - Test different code generation types
   - Test with various GenericSuite patterns

2. **Agent Tool Integration**
   - Test document retrieval tool within agent context
   - Test tool error handling and recovery
   - Test tool performance with large documents

### Performance Tests

1. **Search Performance**
   - Benchmark dual search vs single search
   - Test with large knowledge bases
   - Test concurrent search operations

2. **Document Retrieval Performance**
   - Test retrieval of large documents
   - Test concurrent document access
   - Test caching effectiveness

## Implementation Plan Integration

This design integrates with the existing GenericSuite CodeGen architecture by:

1. **Extending Existing Components**: Building upon the current `KnowledgeBaseTool` and agent architecture
2. **Maintaining Compatibility**: Ensuring all existing functionality continues to work
3. **Adding New Capabilities**: Introducing enhanced search and document retrieval as optional features
4. **Configuration-Driven**: Making enhancements configurable and optional

### Key Integration Points

1. **Agent Tools**: The document retrieval tool will be added to the existing agent tools list
2. **Knowledge Base Tool**: Enhanced search will extend the existing `KnowledgeBaseTool` class
3. **API Endpoints**: Existing endpoints will automatically benefit from enhanced search
4. **MCP Server**: Enhanced search will be available through existing MCP tools

## Security Considerations

1. **File Access Control**: Document retrieval tool will validate file paths to prevent directory traversal
2. **Content Sanitization**: Retrieved document content will be sanitized before use
3. **Resource Limits**: Implement limits on document size and retrieval frequency
4. **Error Information**: Avoid exposing sensitive file system information in error messages

## Performance Considerations

1. **Caching**: Implement caching for frequently accessed documents and search results
2. **Lazy Loading**: Load documents only when needed by the agent
3. **Batch Operations**: Support batch document retrieval for efficiency
4. **Resource Management**: Monitor memory usage for large document operations

## Monitoring and Observability

1. **Metrics**: Track search performance, document retrieval success rates, and context determination accuracy
2. **Logging**: Comprehensive logging for debugging and audit trails
3. **Health Checks**: Include enhanced search components in system health checks
4. **Alerting**: Alert on failures or performance degradation