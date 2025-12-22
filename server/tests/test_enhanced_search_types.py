"""
Unit tests for enhanced search type definitions.
"""

import pytest
from datetime import datetime
from genericsuite_codegen.database.setup import SearchResult
from genericsuite_codegen.agent.enhanced_search_types import (
    CodeGenerationContext,
    DualSearchResult,
    DocumentContent,
    DocumentMetadata,
    SearchTemplate,
    EnhancedSearchConfig,
    DualSearchError,
    ContextDeterminationError,
    DocumentRetrievalError,
    TemplateLoadError,
    ConfigurationError,
    SearchMergeError,
    PerformanceError
)


class TestCodeGenerationContext:
    """Test cases for CodeGenerationContext."""

    def test_code_generation_context_creation(self):
        """Test creating a CodeGenerationContext with all fields."""
        context = CodeGenerationContext(
            code_type="json",
            framework="genericsuite",
            confidence=0.85,
            detected_patterns=["table_config", "json_schema"]
        )

        assert context.code_type == "json"
        assert context.framework == "genericsuite"
        assert context.confidence == 0.85
        assert context.detected_patterns == ["table_config", "json_schema"]

    def test_code_generation_context_defaults(self):
        """Test CodeGenerationContext with default values."""
        context = CodeGenerationContext(code_type="python")

        assert context.code_type == "python"
        assert context.framework is None
        assert context.confidence == 0.0
        assert context.detected_patterns == []

    def test_code_generation_context_post_init(self):
        """Test __post_init__ method initializes detected_patterns."""
        context = CodeGenerationContext(
            code_type="langchain",
            detected_patterns=None
        )

        assert context.detected_patterns == []


class TestDualSearchResult:
    """Test cases for DualSearchResult."""

    def test_dual_search_result_creation(self, mock_search_results):
        """Test creating a DualSearchResult with all fields."""
        user_results = mock_search_results[:2]
        context_results = mock_search_results[1:]
        merged_results = mock_search_results
        context = CodeGenerationContext(code_type="json")

        result = DualSearchResult(
            user_results=user_results,
            context_results=context_results,
            merged_results=merged_results,
            context_used=context,
            user_query="create json config",
            contextual_query="json examples in genericsuite",
            total_results=3,
            sources=["config.json", "tool.py", "pattern.md"]
        )

        assert len(result.user_results) == 2
        assert len(result.context_results) == 2
        assert len(result.merged_results) == 3
        assert result.context_used.code_type == "json"
        assert result.user_query == "create json config"
        assert result.contextual_query == "json examples in genericsuite"
        assert result.total_results == 3
        assert len(result.sources) == 3

    def test_dual_search_result_post_init(self, mock_search_results):
        """Test __post_init__ calculates derived fields."""
        context = CodeGenerationContext(code_type="python")

        result = DualSearchResult(
            user_results=[],
            context_results=[],
            merged_results=mock_search_results,
            context_used=context,
            user_query="test query",
            contextual_query="test contextual",
            total_results=0,
            sources=[]
        )

        # Should calculate total_results from merged_results
        assert result.total_results == len(mock_search_results)


class TestDocumentContent:
    """Test cases for DocumentContent."""

    def test_document_content_creation(self):
        """Test creating DocumentContent with all fields."""
        now = datetime.now()
        content = DocumentContent(
            path="test/file.py",
            content="print('hello')",
            file_type="py",
            size=14,
            last_modified=now,
            metadata={"encoding": "utf-8", "lines": 1}
        )

        assert content.path == "test/file.py"
        assert content.content == "print('hello')"
        assert content.file_type == "py"
        assert content.size == 14
        assert content.last_modified == now
        assert content.metadata["encoding"] == "utf-8"
        assert content.metadata["lines"] == 1

    def test_document_content_minimal(self):
        """Test creating DocumentContent with minimal fields."""
        content = DocumentContent(
            path="test.txt",
            content="test content",
            file_type="txt",
            size=12,
            last_modified=None,
            metadata={}
        )

        assert content.path == "test.txt"
        assert content.content == "test content"
        assert content.file_type == "txt"
        assert content.size == 12
        assert content.last_modified is None
        assert content.metadata == {}


class TestDocumentMetadata:
    """Test cases for DocumentMetadata."""

    def test_document_metadata_creation(self):
        """Test creating DocumentMetadata with all fields."""
        now = datetime.now()
        metadata = DocumentMetadata(
            path="test/file.py",
            file_type="py",
            size=100,
            last_modified=now,
            exists=True
        )

        assert metadata.path == "test/file.py"
        assert metadata.file_type == "py"
        assert metadata.size == 100
        assert metadata.last_modified == now
        assert metadata.exists is True

    def test_document_metadata_nonexistent(self):
        """Test DocumentMetadata for non-existent file."""
        metadata = DocumentMetadata(
            path="nonexistent.txt",
            file_type="txt",
            size=0,
            last_modified=None,
            exists=False
        )

        assert metadata.path == "nonexistent.txt"
        assert metadata.file_type == "txt"
        assert metadata.size == 0
        assert metadata.last_modified is None
        assert metadata.exists is False


class TestSearchTemplate:
    """Test cases for SearchTemplate."""

    def test_search_template_creation(self):
        """Test creating SearchTemplate with all fields."""
        template = SearchTemplate(
            code_type="json",
            template="JSON configuration examples in GenericSuite",
            file_type_filter="json",
            priority=1
        )

        assert template.code_type == "json"
        assert template.template == "JSON configuration examples in GenericSuite"
        assert template.file_type_filter == "json"
        assert template.priority == 1

    def test_search_template_optional_fields(self):
        """Test SearchTemplate with optional fields."""
        template = SearchTemplate(
            code_type="python",
            template="Python examples",
            file_type_filter=None,
            priority=2
        )

        assert template.code_type == "python"
        assert template.template == "Python examples"
        assert template.file_type_filter is None
        assert template.priority == 2


class TestEnhancedSearchConfig:
    """Test cases for EnhancedSearchConfig."""

    def test_enhanced_search_config_creation(self, sample_search_templates):
        """Test creating EnhancedSearchConfig with all fields."""
        config = EnhancedSearchConfig(
            templates=sample_search_templates,
            local_repo_path="local_repo_files",
            max_context_length=10000,
            fallback_enabled=True
        )

        assert len(config.templates) == 3
        assert config.local_repo_path == "local_repo_files"
        assert config.max_context_length == 10000
        assert config.fallback_enabled is True

    def test_enhanced_search_config_defaults(self):
        """Test EnhancedSearchConfig with default values."""
        config = EnhancedSearchConfig(
            templates={},
            local_repo_path="local_repo_files",
            max_context_length=5000,
            fallback_enabled=False
        )

        assert config.templates == {}
        assert config.local_repo_path == "local_repo_files"
        assert config.max_context_length == 5000
        assert config.fallback_enabled is False


class TestExceptionHierarchy:
    """Test cases for enhanced search exception hierarchy."""

    def test_dual_search_error(self):
        """Test DualSearchError exception."""
        error = DualSearchError("Dual search failed")
        assert str(error) == "Dual search failed"
        assert isinstance(error, Exception)

    def test_context_determination_error(self):
        """Test ContextDeterminationError exception."""
        error = ContextDeterminationError("Context determination failed")
        assert str(error) == "Context determination failed"
        assert isinstance(error, Exception)

    def test_document_retrieval_error(self):
        """Test DocumentRetrievalError exception."""
        error = DocumentRetrievalError("Document retrieval failed")
        assert str(error) == "Document retrieval failed"
        assert isinstance(error, Exception)

    def test_template_load_error(self):
        """Test TemplateLoadError exception."""
        error = TemplateLoadError("Template loading failed")
        assert str(error) == "Template loading failed"
        assert isinstance(error, Exception)

    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError("Configuration error")
        assert str(error) == "Configuration error"
        assert isinstance(error, Exception)

    def test_search_merge_error(self):
        """Test SearchMergeError exception."""
        error = SearchMergeError("Search merge failed")
        assert str(error) == "Search merge failed"
        assert isinstance(error, Exception)

    def test_performance_error(self):
        """Test PerformanceError exception."""
        error = PerformanceError("Performance threshold exceeded")
        assert str(error) == "Performance threshold exceeded"
        assert isinstance(error, Exception)


class TestConstants:
    """Test cases for module constants."""

    def test_merge_priority_weights_exist(self):
        """Test that MERGE_PRIORITY_WEIGHTS constant exists and has expected structure."""
        from genericsuite_codegen.agent.enhanced_search_types import MERGE_PRIORITY_WEIGHTS

        assert isinstance(MERGE_PRIORITY_WEIGHTS, dict)
        assert "balanced" in MERGE_PRIORITY_WEIGHTS
        assert "prioritize_context" in MERGE_PRIORITY_WEIGHTS
        assert isinstance(MERGE_PRIORITY_WEIGHTS["balanced"], dict)
        assert "context" in MERGE_PRIORITY_WEIGHTS["balanced"]
        assert "user" in MERGE_PRIORITY_WEIGHTS["balanced"]

    def test_context_patterns_exist(self):
        """Test that CONTEXT_PATTERNS constant exists."""
        from genericsuite_codegen.agent.enhanced_search_types import CONTEXT_PATTERNS

        assert isinstance(CONTEXT_PATTERNS, dict)
        assert len(CONTEXT_PATTERNS) > 0

    def test_default_search_templates_exist(self):
        """Test that DEFAULT_SEARCH_TEMPLATES constant exists."""
        from genericsuite_codegen.agent.enhanced_search_types import DEFAULT_SEARCH_TEMPLATES

        assert isinstance(DEFAULT_SEARCH_TEMPLATES, dict)
        assert len(DEFAULT_SEARCH_TEMPLATES) > 0
