"""
Pydantic models for FastAPI request/response validation.

This module defines all the data models used for API request validation,
response formatting, and data transfer objects in the GenericSuite CodeGen API.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict


class TaskType(str, Enum):
    """Enumeration of supported task types."""
    GENERAL = "general"
    JSON = "json"
    PYTHON = "python"
    FRONTEND = "frontend"
    BACKEND = "backend"


class BackendFramework(str, Enum):
    """Enumeration of supported backend frameworks."""
    FASTAPI = "fastapi"
    FLASK = "flask"
    CHALICE = "chalice"


class EmbeddingProvider(str, Enum):
    """Enumeration of supported embedding providers."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    LITELLM = "litellm"


# Base Models

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ErrorResponse(BaseResponse):
    """Error response model."""
    error_code: str = Field(description="Error code identifier")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")


class SuccessResponse(BaseResponse):
    """Success response model."""
    message: str = Field(description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")


# Application Info Models

class AppInfo(BaseModel):
    """Application information model."""
    name: str = Field(description="Application name")
    version: str = Field(description="Application version")
    description: str = Field(description="Application description")
    docs_url: str = Field(description="API documentation URL")
    health_url: str = Field(description="Health check URL")


class HealthResponse(BaseResponse):
    """Health check response model."""
    status: str = Field(description="Overall health status (healthy/unhealthy)")
    version: str = Field(description="Application version")
    components: Dict[str, Any] = Field(default_factory=dict, description="Component health status")


# Agent Query Models

class QueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str = Field(
        min_length=3,
        max_length=10000,
        description="User query or request"
    )
    task_type: TaskType = Field(
        default=TaskType.GENERAL,
        description="Type of task to perform"
    )
    framework: Optional[BackendFramework] = Field(
        default=None,
        description="Backend framework for code generation (required for backend tasks)"
    )
    context_limit: int = Field(
        default=4000,
        ge=500,
        le=8000,
        description="Maximum context length for knowledge base retrieval"
    )
    include_sources: bool = Field(
        default=True,
        description="Include source attribution in response"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for context continuity"
    )
    
    @validator('framework')
    def validate_framework_for_backend(cls, v, values):
        """Validate that framework is provided for backend tasks."""
        if values.get('task_type') == TaskType.BACKEND and v is None:
            raise ValueError('Framework is required for backend code generation tasks')
        return v


class QueryResponse(BaseResponse):
    """Response model for agent queries."""
    content: str = Field(description="Generated response content")
    sources: List[str] = Field(default_factory=list, description="Source documents used")
    task_type: TaskType = Field(description="Type of task performed")
    model_used: str = Field(description="Model used for generation")
    token_usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Token usage statistics"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID"
    )


# Conversation Models

class Message(BaseModel):
    """Individual message in a conversation."""
    role: str = Field(description="Message role (user/assistant)")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    sources: Optional[List[str]] = Field(default=None, description="Source documents for assistant messages")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage for this message")
    
    @validator('role')
    def validate_role(cls, v):
        """Validate message role."""
        if v not in ['user', 'assistant']:
            raise ValueError('Role must be either "user" or "assistant"')
        return v


class ConversationCreate(BaseModel):
    """Request model for creating a new conversation."""
    title: Optional[str] = Field(default=None, max_length=200, description="Conversation title")
    initial_message: Optional[str] = Field(default=None, description="Initial message to start the conversation")


class ConversationUpdate(BaseModel):
    """Request model for updating a conversation."""
    title: Optional[str] = Field(default=None, max_length=200, description="New conversation title")


class Conversation(BaseModel):
    """Conversation model."""
    id: str = Field(description="Conversation ID")
    title: str = Field(description="Conversation title")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    message_count: int = Field(description="Number of messages in conversation")


class ConversationList(BaseResponse):
    """Response model for conversation list."""
    conversations: List[Conversation] = Field(description="List of conversations")
    total: int = Field(description="Total number of conversations")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of items per page")


# Knowledge Base Models

class DocumentUpload(BaseModel):
    """Request model for document upload."""
    filename: str = Field(description="Original filename")
    content_type: str = Field(description="MIME type of the file")
    description: Optional[str] = Field(default=None, description="Optional description of the document")


class DocumentInfo(BaseModel):
    """Document information model."""
    id: str = Field(description="Document ID")
    filename: str = Field(description="Original filename")
    file_type: str = Field(description="File type/extension")
    size: int = Field(description="File size in bytes")
    upload_date: datetime = Field(description="Upload timestamp")
    description: Optional[str] = Field(default=None, description="Document description")
    chunk_count: int = Field(description="Number of chunks created from this document")


class KnowledgeBaseUpdate(BaseModel):
    """Request model for knowledge base updates."""
    force_refresh: bool = Field(
        default=False,
        description="Force refresh even if repository hasn't changed"
    )
    repository_url: Optional[str] = Field(
        default=None,
        description="Override repository URL for this update"
    )
    include_patterns: Optional[List[str]] = Field(
        default=None,
        description="File patterns to include (overrides default)"
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description="Additional file patterns to exclude"
    )


class KnowledgeBaseStatus(BaseResponse):
    """Knowledge base status response."""
    status: str = Field(description="Current status (idle/updating/error)")
    last_update: Optional[datetime] = Field(default=None, description="Last successful update")
    document_count: int = Field(description="Total number of documents")
    chunk_count: int = Field(description="Total number of chunks")
    repository_url: str = Field(description="Current repository URL")
    repository_commit: Optional[str] = Field(default=None, description="Current repository commit hash")
    error_message: Optional[str] = Field(default=None, description="Last error message if any")


class ProgressUpdate(BaseModel):
    """Progress update model for long-running operations."""
    operation_id: str = Field(description="Operation identifier")
    status: str = Field(description="Current status")
    progress: float = Field(ge=0.0, le=1.0, description="Progress percentage (0.0 to 1.0)")
    message: str = Field(description="Current operation message")
    started_at: datetime = Field(description="Operation start time")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")


# File Generation Models

class FileGenerationRequest(BaseModel):
    """Request model for file generation."""
    content: str = Field(description="Generated content to package")
    filename: str = Field(description="Desired filename")
    file_type: str = Field(description="File type (json, python, javascript, etc.)")
    description: Optional[str] = Field(default=None, description="File description")


class GeneratedFile(BaseModel):
    """Generated file model."""
    filename: str = Field(description="Generated filename")
    content: str = Field(description="File content")
    file_type: str = Field(description="File type")
    size: int = Field(description="File size in bytes")
    description: Optional[str] = Field(default=None, description="File description")


class FilePackage(BaseModel):
    """File package model for multi-file downloads."""
    package_name: str = Field(description="Package name")
    files: List[GeneratedFile] = Field(description="Files in the package")
    format: str = Field(description="Package format (zip, tar.gz)")
    total_size: int = Field(description="Total package size in bytes")


# Configuration Models

class AgentConfig(BaseModel):
    """Agent configuration model."""
    model_provider: LLMProvider = Field(description="LLM provider")
    model_name: str = Field(description="Model name")
    temperature: float = Field(ge=0.0, le=1.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens for responses")
    timeout: int = Field(ge=1, description="Request timeout in seconds")


class EmbeddingConfig(BaseModel):
    """Embedding configuration model."""
    provider: EmbeddingProvider = Field(description="Embedding provider")
    model_name: str = Field(description="Embedding model name")
    dimension: int = Field(ge=1, description="Embedding dimension")


class SystemConfig(BaseModel):
    """System configuration model."""
    agent: AgentConfig = Field(description="Agent configuration")
    embeddings: EmbeddingConfig = Field(description="Embedding configuration")
    database_uri: str = Field(description="Database connection URI")
    repository_url: str = Field(description="Default repository URL")
    cors_origins: List[str] = Field(description="Allowed CORS origins")
    debug: bool = Field(description="Debug mode enabled")


# Search and Statistics Models

class SearchQuery(BaseModel):
    """Search query model."""
    query: str = Field(min_length=1, max_length=1000, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    file_type_filter: Optional[str] = Field(default=None, description="Filter by file type")
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )


class SearchResult(BaseModel):
    """Search result model."""
    content: str = Field(description="Document content")
    file_path: str = Field(description="Source file path")
    file_type: str = Field(description="File type")
    similarity_score: float = Field(description="Similarity score")
    chunk_index: int = Field(description="Chunk index within document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseResponse):
    """Search response model."""
    results: List[SearchResult] = Field(description="Search results")
    total_results: int = Field(description="Total number of results found")
    query: str = Field(description="Original search query")
    execution_time: float = Field(description="Query execution time in seconds")


class Statistics(BaseModel):
    """System statistics model."""
    knowledge_base: Dict[str, Any] = Field(description="Knowledge base statistics")
    conversations: Dict[str, Any] = Field(description="Conversation statistics")
    agent: Dict[str, Any] = Field(description="Agent usage statistics")
    system: Dict[str, Any] = Field(description="System resource statistics")


# Validation Utilities

def validate_file_type(file_type: str) -> bool:
    """
    Validate file type.
    
    Args:
        file_type: File type to validate.
        
    Returns:
        bool: True if file type is supported.
    """
    supported_types = {
        'json', 'py', 'js', 'jsx', 'ts', 'tsx', 'md', 'txt', 'pdf',
        'toml', 'yaml', 'yml', 'env', 'html', 'css', 'sql'
    }
    return file_type.lower() in supported_types


def validate_conversation_id(conversation_id: str) -> bool:
    """
    Validate conversation ID format.
    
    Args:
        conversation_id: Conversation ID to validate.
        
    Returns:
        bool: True if ID format is valid.
    """
    import re
    # MongoDB ObjectId pattern
    return bool(re.match(r'^[0-9a-fA-F]{24}$', conversation_id))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename: Original filename.
        
    Returns:
        str: Sanitized filename.
    """
    import re
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    # Ensure it doesn't start with a dot or dash
    sanitized = re.sub(r'^[\.\-]+', '', sanitized)
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:250] + ('.' + ext if ext else '')
    
    return sanitized or 'unnamed_file'


# Response Helpers

def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> ErrorResponse:
    """
    Create a standardized error response.
    
    Args:
        error_code: Error code identifier.
        message: Human-readable error message.
        details: Additional error details.
        correlation_id: Request correlation ID.
        
    Returns:
        ErrorResponse: Formatted error response.
    """
    return ErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        correlation_id=correlation_id
    )


def create_success_response(
    message: str,
    data: Optional[Dict[str, Any]] = None
) -> SuccessResponse:
    """
    Create a standardized success response.
    
    Args:
        message: Success message.
        data: Response data.
        
    Returns:
        SuccessResponse: Formatted success response.
    """
    return SuccessResponse(
        message=message,
        data=data
    )