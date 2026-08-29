"""
Enhanced search type definitions for GenericSuite CodeGen.

This module provides type definitions for enhanced vector search capabilities
that combine user queries with contextual GenericSuite rules and patterns.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field

from genericsuite_codegen.database.setup import SearchResult


@dataclass
class CodeGenerationContext:
    """Context information for code generation."""
    # json, langchain, mcp, frontend, backend, frontend_ai, backend_ai
    code_type: str
    # fastapi, react, langchain, fastmcp, etc.
    framework: Optional[str] = None
    # confidence in context determination (0.0 to 1.0)
    confidence: float = 0.0
    # detected patterns in the query
    detected_patterns: List[str] = None

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.detected_patterns is None:
            self.detected_patterns = []


@dataclass
class DualSearchResult:
    """
    Result from dual vector search combining user query and contextual rules.
    """
    user_results: List[SearchResult]
    context_results: List[SearchResult]
    merged_results: List[SearchResult]
    context_used: CodeGenerationContext
    user_query: str
    contextual_query: str
    total_results: int
    sources: List[str]

    def __post_init__(self):
        """Calculate derived fields after dataclass creation."""
        if not hasattr(self, 'total_results') or self.total_results == 0:
            self.total_results = len(self.merged_results)

        if not hasattr(self, 'sources') or not self.sources:
            # Extract unique sources from merged results
            sources_set = set()
            for result in self.merged_results:
                sources_set.add(result.document_path)
            self.sources = list(sources_set)


@dataclass
class DocumentContent:
    """Complete document content retrieved from local storage."""
    path: str
    content: str
    file_type: str
    size: int
    last_modified: datetime
    metadata: Dict[str, Any]
    encoding: str = "utf-8"
    is_binary: bool = False

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentMetadata:
    """Document metadata without full content for lightweight operations."""
    path: str
    file_type: str
    size: int
    last_modified: datetime
    exists: bool
    is_readable: bool = True
    encoding: Optional[str] = None
    is_binary: bool = False
    error_message: Optional[str] = None


@dataclass
class SearchTemplate:
    """Search template configuration for different code generation types."""
    code_type: str
    template: str
    file_type_filter: Optional[str] = None
    priority: int = 1
    description: str = ""
    examples: List[str] = None

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.examples is None:
            self.examples = []


@dataclass
class EnhancedSearchConfig:
    """Configuration for enhanced search functionality."""
    templates: Dict[str, SearchTemplate]
    local_repo_path: str = "local_repo_files"
    max_context_length: int = 10000
    fallback_enabled: bool = True
    context_determination_enabled: bool = True
    document_retrieval_enabled: bool = True
    search_result_limit: int = 10
    similarity_threshold: float = 0.7
    # prioritize_context, balanced, prioritize_user
    merge_strategy: str = "prioritize_context"

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.templates is None:
            self.templates = {}


class EnhancedSearchQuery(BaseModel):
    """Query model for enhanced search operations."""
    user_query: str = Field(description="The user's search query")
    code_type: Optional[str] = Field(
        default=None,
        description="Explicit code type (json, langchain, mcp, frontend"
        ", backend)"
    )
    framework: Optional[str] = Field(
        default=None,
        description="Specific framework or technology"
    )
    max_context_length: int = Field(
        default=10000,
        description="Maximum context length in characters"
    )
    file_type_filter: Optional[str] = Field(
        default=None,
        description="Optional file type filter"
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return",
        ge=1,
        le=50
    )
    enable_dual_search: bool = Field(
        default=True,
        description="Enable dual search with contextual rules"
    )
    enable_document_retrieval: bool = Field(
        default=True,
        description="Enable full document retrieval from local storage"
    )


class EnhancedSearchResponse(BaseModel):
    """Response model for enhanced search operations."""
    dual_search_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dual search results as dictionary"
    )
    formatted_context: str = Field(
        description="Formatted context string for code generation"
    )
    sources: List[str] = Field(
        description="List of source document paths"
    )
    context_used: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Context information used for search"
    )
    retrieved_documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full documents retrieved from local storage"
    )
    search_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the search operation"
    )


class DocumentRetrievalRequest(BaseModel):
    """Request model for document retrieval operations."""
    document_path: str = Field(description="Path to the document to retrieve")
    include_metadata: bool = Field(
        default=True,
        description="Include document metadata in response"
    )
    max_size_mb: int = Field(
        default=10,
        description="Maximum file size to retrieve in MB",
        ge=1,
        le=100
    )


class DocumentRetrievalResponse(BaseModel):
    """Response model for document retrieval operations."""
    success: bool = Field(description="Whether retrieval was successful")
    document: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Retrieved document content and metadata"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if retrieval failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the retrieval operation"
    )


class BatchDocumentRetrievalRequest(BaseModel):
    """Request model for batch document retrieval operations."""
    document_paths: List[str] = Field(
        description="List of document paths to retrieve",
        min_length=1,
        max_length=20
    )
    include_metadata: bool = Field(
        default=True,
        description="Include document metadata in responses"
    )
    max_size_mb: int = Field(
        default=10,
        description="Maximum file size to retrieve per document in MB",
        ge=1,
        le=100
    )
    continue_on_error: bool = Field(
        default=True,
        description="Continue processing other documents if one fails"
    )


class BatchDocumentRetrievalResponse(BaseModel):
    """Response model for batch document retrieval operations."""
    successful_retrievals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Successfully retrieved documents"
    )
    failed_retrievals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Failed document retrievals with error messages"
    )
    total_requested: int = Field(
        description="Total number of documents requested")
    total_successful: int = Field(
        description="Total number of successful retrievals")
    total_failed: int = Field(description="Total number of failed retrievals")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the batch operation"
    )


# Exception classes for enhanced search operations

class EnhancedSearchError(Exception):
    """Base exception for enhanced search operations."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_code = error_code or "ENHANCED_SEARCH_ERROR"
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "original_exception": (
                str(self.original_exception)
                if self.original_exception else None
            )
        }


class ContextDeterminationError(EnhancedSearchError):
    """Raised when context determination fails."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        analysis_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if query:
            details['query'] = query
        if analysis_data:
            details['analysis_data'] = analysis_data
        kwargs['details'] = details
        kwargs['error_code'] = kwargs.get(
            'error_code', 'CONTEXT_DETERMINATION_ERROR')
        super().__init__(message, **kwargs)


class DocumentRetrievalError(EnhancedSearchError):
    """Raised when document retrieval fails."""

    def __init__(
        self,
        message: str,
        document_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if document_path:
            details['document_path'] = document_path
        if operation:
            details['operation'] = operation
        kwargs['details'] = details
        kwargs['error_code'] = kwargs.get(
            'error_code', 'DOCUMENT_RETRIEVAL_ERROR')
        super().__init__(message, **kwargs)


class TemplateLoadError(EnhancedSearchError):
    """Raised when template loading fails."""

    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        template_type: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if config_path:
            details['config_path'] = config_path
        if template_type:
            details['template_type'] = template_type
        kwargs['details'] = details
        kwargs['error_code'] = kwargs.get('error_code', 'TEMPLATE_LOAD_ERROR')
        super().__init__(message, **kwargs)


class DualSearchError(EnhancedSearchError):
    """Raised when dual search operations fail."""

    def __init__(
        self,
        message: str,
        user_query: Optional[str] = None,
        contextual_query: Optional[str] = None,
        search_phase: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if user_query:
            details['user_query'] = user_query
        if contextual_query:
            details['contextual_query'] = contextual_query
        if search_phase:
            details['search_phase'] = search_phase
        kwargs['details'] = details
        kwargs['error_code'] = kwargs.get('error_code', 'DUAL_SEARCH_ERROR')
        super().__init__(message, **kwargs)


class SearchMergeError(EnhancedSearchError):
    """Raised when search result merging fails."""

    def __init__(
        self,
        message: str,
        merge_strategy: Optional[str] = None,
        user_results_count: Optional[int] = None,
        context_results_count: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if merge_strategy:
            details['merge_strategy'] = merge_strategy
        if user_results_count is not None:
            details['user_results_count'] = user_results_count
        if context_results_count is not None:
            details['context_results_count'] = context_results_count
        kwargs['details'] = details
        kwargs['error_code'] = kwargs.get('error_code', 'SEARCH_MERGE_ERROR')
        super().__init__(message, **kwargs)


class ConfigurationError(EnhancedSearchError):
    """Raised when configuration validation fails."""

    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        invalid_values: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if config_section:
            details['config_section'] = config_section
        if invalid_values:
            details['invalid_values'] = invalid_values
        kwargs['details'] = details
        kwargs['error_code'] = kwargs.get('error_code', 'CONFIGURATION_ERROR')
        super().__init__(message, **kwargs)


class PerformanceError(EnhancedSearchError):
    """Raised when performance thresholds are exceeded."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        duration: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        if duration is not None:
            details['duration'] = duration
        if threshold is not None:
            details['threshold'] = threshold
        kwargs['details'] = details
        kwargs['error_code'] = kwargs.get('error_code', 'PERFORMANCE_ERROR')
        super().__init__(message, **kwargs)


# Constants for enhanced search operations

# Default search templates for different code types
DEFAULT_SEARCH_TEMPLATES = {
    "json": "examples of how to create a JSON table configuration files in"
    " Genericsuite",
    "langchain": "examples of how to create a Python Langchain Tool in"
    " Genericsuite",
    "mcp": "examples of how to create a MCP server tool in Genericsuite",
    "frontend": "examples of how to create frontend code in Genericsuite",
    "backend": "examples of how to create backend code in Genericsuite",
    "frontend_ai": "examples of how to create frontend with AI code in"
    " Genericsuite",
    "backend_ai": "examples of how to create backend with AI code in"
    " Genericsuite"
}

# File type filters for different code types
DEFAULT_FILE_TYPE_FILTERS = {
    "json": "json",
    "langchain": "py",
    "mcp": "py",
    "frontend": "jsx",
    "frontend_ai": "jsx",
    "backend": "py",
    "backend_ai": "py"
}

# Context determination patterns
CONTEXT_PATTERNS = {
    "json": [
        "json", "configuration", "config", "table", "form", "menu", "auth",
        "database", "schema", "field", "validation"
    ],
    "langchain": [
        "langchain", "tool", "agent", "chain", "llm", "embedding", "retriever",
        "memory", "callback", "prompt"
    ],
    "mcp": [
        "mcp", "model context protocol", "server", "client", "tool",
        "resource", "prompt", "stdio", "transport"
    ],
    "frontend": [
        "react", "component", "jsx", "tsx", "ui", "interface", "form", "page",
        "routing", "state", "props", "hook"
    ],
    "backend": [
        "fastapi", "flask", "chalice", "api", "endpoint", "route", "model",
        "database", "service", "middleware", "auth", "validation"
    ],
    "frontend_ai": [
        "react", "ai", "chatbot", "llm", "streaming", "conversation", "agent",
        "component", "interface", "chat"
    ],
    "backend_ai": [
        "fastapi", "ai", "agent", "llm", "embedding", "vector", "search",
        "rag", "retrieval", "generation", "api"
    ]
}

# Priority weights for search result merging
MERGE_PRIORITY_WEIGHTS = {
    "prioritize_context": {"context": 0.7, "user": 0.3},
    "balanced": {"context": 0.5, "user": 0.5},
    "prioritize_user": {"context": 0.3, "user": 0.7}
}
