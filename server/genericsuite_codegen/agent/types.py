from typing import List, Dict, Any, Optional
from datetime import datetime

from dataclasses import dataclass
from pydantic import BaseModel, Field


# Knowledge Base Tools


class KnowledgeBaseQuery(BaseModel):
    """Query model for knowledge base search."""
    query: str = Field(description="The search query text")
    limit: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=20)
    file_type_filter: Optional[str] = Field(
        default=None,
        description="Optional file type filter (e.g., 'py', 'md', 'json')")


class SearchResultModel(BaseModel):
    """Model for search results with source attribution."""
    content: str = Field(description="The content of the document chunk")
    document_path: str = Field(description="Path to the source document")
    similarity_score: float = Field(
        description="Similarity score (0.0 to 1.0)")
    file_type: str = Field(description="Type of the source file")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")


class KnowledgeBaseSearchResults(BaseModel):
    """Complete search results with context and attribution."""
    results: List[SearchResultModel] = Field(
        description="List of search results")
    total_results: int = Field(description="Total number of results found")
    query: str = Field(description="Original search query")
    context_summary: str = Field(
        description="Summary of the retrieved context")
    sources: List[str] = Field(
        description="List of unique source document paths")


@dataclass
class ContextRanking:
    """Context ranking and relevance scoring."""
    content: str
    relevance_score: float
    source_path: str
    file_type: str
    metadata: Dict[str, Any]


# JSON Configuration Generation Tools


class JSONConfigRequest(BaseModel):
    """Request model for JSON configuration generation."""
    config_type: str = Field(
        description="Type of configuration (table, form, menu, auth)")
    requirements: str = Field(
        description="Requirements and specifications for the configuration")
    table_name: Optional[str] = Field(
        default=None, description="Name of the table (for table configs)")
    include_validation: bool = Field(
        default=True, description="Include validation rules")
    include_examples: bool = Field(
        default=True, description="Include example values and comments")


class JSONConfigResult(BaseModel):
    """Result model for JSON configuration generation."""
    configuration: Dict[str, Any] = Field(
        description="Generated JSON configuration")
    config_type: str = Field(description="Type of configuration generated")
    validation_notes: List[str] = Field(
        default_factory=list, description="Validation and usage notes")
    examples: Dict[str, Any] = Field(
        default_factory=dict, description="Example configurations")
    sources: List[str] = Field(
        default_factory=list, description="Source documents used")


# Create Context Retrieval Tool


class ContextQuery(BaseModel):
    """Query model for context retrieval."""
    query: str = Field(
        description="The search query for context retrieval")
    max_length: int = Field(
        default=4000, description="Maximum context length in characters")
    file_type: Optional[str] = Field(
        default=None, description="Optional file type filter")


class ContextResult(BaseModel):
    """Result model for context retrieval."""
    context: str = Field(description="Formatted context string")
    sources: List[str] = Field(description="List of source document paths")
    query: str = Field(description="Original query")


# Create Config Validation Tool


class ValidationRequest(BaseModel):
    """Request model for configuration validation."""
    configuration: Dict[str, Any] = Field(
        description="JSON configuration to validate")
    config_type: str = Field(
        description="Type of configuration (table, form, etc.)")


class ValidationResult(BaseModel):
    """Result model for configuration validation."""
    is_valid: bool = Field(
        description="Whether the configuration is valid")
    errors: List[str] = Field(description="List of validation errors")
    suggestions: List[str] = Field(
        description="Suggestions for improvement")


# Python Code Generation Tools


class PythonCodeRequest(BaseModel):
    """Request model for Python code generation."""
    code_type: str = Field(
        description="Type of code (langchain_tool, mcp_tool, utility,"
                    " api_endpoint)")
    requirements: str = Field(
        description="Requirements and specifications for the code")
    tool_name: str = Field(
        description="Name of the tool or function to generate")
    include_tests: bool = Field(default=True, description="Include unit tests")
    include_docs: bool = Field(
        default=True, description="Include comprehensive documentation")
    framework: Optional[str] = Field(
        default=None, description="Specific framework (fastmcp, langchain)")


class PythonCodeResult(BaseModel):
    """Result model for Python code generation."""
    code: str = Field(description="Generated Python code")
    code_type: str = Field(description="Type of code generated")
    imports: List[str] = Field(description="Required imports and dependencies")
    usage_example: str = Field(
        description="Example usage of the generated code")
    test_code: Optional[str] = Field(
        default=None, description="Unit test code")
    documentation: str = Field(description="Documentation and usage notes")
    sources: List[str] = Field(
        default_factory=list, description="Source documents used")


# Frontend and Backend Code Generation Tools


class FrontendCodeRequest(BaseModel):
    """Request model for frontend code generation."""
    component_type: str = Field(
        description="Type of component (form, table, page, component)")
    requirements: str = Field(
        description="Requirements and specifications for the frontend code")
    component_name: str = Field(
        description="Name of the component to generate")
    include_styling: bool = Field(
        default=True, description="Include CSS/styling")
    include_tests: bool = Field(
        default=True, description="Include component tests")
    ui_framework: str = Field(
        default="react", description="UI framework (react, vue, angular)")


class BackendCodeRequest(BaseModel):
    """Request model for backend code generation."""
    framework: str = Field(
        description="Backend framework (fastapi, flask, chalice)")
    code_type: str = Field(
        description="Type of code (api_endpoint, model, service, middleware)")
    requirements: str = Field(
        description="Requirements and specifications for the backend code")
    module_name: str = Field(
        description="Name of the module/endpoint to generate")
    include_auth: bool = Field(
        default=True, description="Include authentication")
    include_validation: bool = Field(
        default=True, description="Include input validation")
    include_tests: bool = Field(default=True, description="Include unit tests")


class CodeGenerationResult(BaseModel):
    """Result model for frontend/backend code generation."""
    code: str = Field(description="Generated code")
    code_type: str = Field(description="Type of code generated")
    framework: str = Field(description="Framework used")
    files: Dict[str, str] = Field(
        description="Additional files (tests, styles, etc.)")
    imports: List[str] = Field(description="Required imports and dependencies")
    usage_instructions: str = Field(
        description="Instructions for using the generated code")
    integration_notes: str = Field(
        description="Notes for integrating with GenericSuite")
    sources: List[str] = Field(
        default_factory=list, description="Source documents used")


# Agent Model


class AgentModel(BaseModel):
    provider: str = Field(description="Agent provider")
    model: str = Field(description="Agent model")
    temperature: float = Field(description="Temperature")
    max_tokens: int = Field(description="Max tokens")
    timeout: int = Field(description="Timeout")


class AgentConfig(BaseModel):
    """Configuration for the GenericSuite AI agent."""

    model_provider: str = Field(
        default="openai", description="LLM provider (openai, litellm)"
    )
    model_name: str = Field(default="gpt-4", description="Model name to use")
    temperature: float = Field(
        default=0.1, description="Model temperature (0.0 to 1.0)"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens for responses"
    )
    timeout: int = Field(default=60, description="Request timeout in seconds")
    api_key: Optional[str] = Field(
        default=None, description="API key for the provider")
    base_url: Optional[str] = Field(
        default=None, description="Custom base URL for API")


class QueryRequest(BaseModel):
    """Request model for agent queries."""

    query: str = Field(description="User query or request")
    task_type: str = Field(
        default="general",
        description="Type of task (general, json, python, frontend, backend)",
    )
    framework: Optional[str] = Field(
        default=None, description="Backend framework for code generation"
    )
    context_limit: int = Field(
        default=4000, description="Maximum context length")
    include_sources: bool = Field(
        default=True, description="Include source attribution in response"
    )


class AgentResponse(BaseModel):
    """Response model from the agent."""

    content: str = Field(description="Generated response content")
    sources: List[str] = Field(
        default_factory=list, description="Source documents used"
    )
    task_type: str = Field(description="Type of task performed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    model_used: str = Field(description="Model used for generation")
    token_usage: Optional[Dict[str, int]] = Field(
        default=None, description="Token usage statistics"
    )


@dataclass
class AgentContext:
    """Context information for agent operations."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = None
    preferences: Dict[str, Any] = None
