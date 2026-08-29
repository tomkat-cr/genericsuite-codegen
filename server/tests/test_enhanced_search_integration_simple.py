"""
Simplified integration tests for enhanced search components.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from genericsuite_codegen.database.setup import SearchResult
from genericsuite_codegen.agent.enhanced_search import EnhancedVectorSearch
from genericsuite_codegen.agent.enhanced_search_types import (
    CodeGenerationContext,
    DualSearchResult
)
from genericsuite_codegen.agent.context_determination import ContextDeterminationService
from genericsuite_codegen.agent.search_templates import SearchTemplateManager
from genericsuite_codegen.agent.document_retrieval_tool import DocumentRetrievalTool


class TestEnhancedSearchIntegrationSimple:
    """Simplified integration tests for enhanced search workflow."""

    @pytest.fixture
    def mock_kb_tool_simple(self):
        """Create a simple mock KnowledgeBaseTool."""
        mock_tool = Mock()

        # Simple mock that returns basic results
        async def mock_search(query, **kwargs):
            return [
                SearchResult(
                    content=f"Result for query: {query}",
                    metadata={"source": "test.py", "type": "python"},
                    similarity_score=0.8,
                    document_path="test.py"
                )
            ]

        mock_tool.search_knowledge_base = AsyncMock(side_effect=mock_search)
        return mock_tool

    @pytest.fixture
    def integrated_search_simple(self, mock_kb_tool_simple):
        """Create a simple integrated enhanced search system."""
        template_manager = SearchTemplateManager()
        context_service = ContextDeterminationService(template_manager)

        enhanced_search = EnhancedVectorSearch(
            kb_tool=mock_kb_tool_simple,
            template_manager=template_manager,
            context_service=context_service
        )

        return enhanced_search

    def test_context_determination_integration(self, integrated_search_simple):
        """Test that context determination works with template manager."""
        context = integrated_search_simple.context_service.determine_context(
            "create a JSON table configuration", "json"
        )

        assert isinstance(context, CodeGenerationContext)
        assert context.code_type == "json"
        assert context.confidence > 0.0

    def test_template_manager_integration(self, integrated_search_simple):
        """Test that template manager provides templates."""
        template = integrated_search_simple.template_manager.get_template(
            "json")

        assert isinstance(template, str)
        assert len(template) > 0

    @pytest.mark.asyncio
    async def test_basic_dual_search_integration(self, integrated_search_simple):
        """Test basic dual search integration."""
        context = CodeGenerationContext(code_type="json")

        result = await integrated_search_simple.dual_search(
            user_query="create json config",
            code_context=context,
            max_context_length=5000
        )

        assert isinstance(result, DualSearchResult)
        assert result.user_query == "create json config"
        assert isinstance(result.contextual_query, str)

    def test_document_retrieval_integration(self, temp_repo_dir):
        """Test document retrieval tool integration."""
        doc_tool = DocumentRetrievalTool(temp_repo_dir)

        # Test retrieving documents that might be referenced in search results
        documents = doc_tool.retrieve_multiple_documents([
            "test.py",
            "config.json",
            "README.md"
        ])

        assert len(documents) == 3
        assert all(doc.content for doc in documents)

    def test_search_result_merging(self, integrated_search_simple):
        """Test that search result merging works."""
        result1 = SearchResult(
            content="content1",
            metadata={"source": "file1.py"},
            similarity_score=0.8,
            document_path="file1.py"
        )
        result2 = SearchResult(
            content="content2",
            metadata={"source": "file2.py"},
            similarity_score=0.7,
            document_path="file2.py"
        )

        merged = integrated_search_simple.merge_search_results([result1], [
                                                               result2])

        assert len(merged) == 2
        # Check that both results are represented (content should match)
        merged_contents = [r.content for r in merged]
        assert "content1" in merged_contents
        assert "content2" in merged_contents

    def test_configuration_integration(self, integrated_search_simple):
        """Test configuration integration."""
        # Test that we can get statistics
        integrated_search_simple.template_manager = Mock()
        integrated_search_simple.template_manager.get_all_templates.return_value = {
            "json": Mock()}

        stats = integrated_search_simple.get_search_statistics()

        assert isinstance(stats, dict)
        assert "available_templates" in stats

    def test_query_validation_integration(self, integrated_search_simple):
        """Test query validation integration."""
        # Valid query
        is_valid, error = integrated_search_simple.validate_search_query(
            "valid query")
        assert is_valid is True
        assert error is None

        # Invalid query
        is_valid, error = integrated_search_simple.validate_search_query("")
        assert is_valid is False
        assert error is not None

    def test_component_initialization(self):
        """Test that all components can be initialized together."""
        mock_kb_tool = Mock()
        mock_kb_tool.search_knowledge_base = AsyncMock()

        template_manager = SearchTemplateManager()
        context_service = ContextDeterminationService(template_manager)
        enhanced_search = EnhancedVectorSearch(
            kb_tool=mock_kb_tool,
            template_manager=template_manager,
            context_service=context_service
        )

        # All components should be properly initialized
        assert enhanced_search.kb_tool == mock_kb_tool
        assert enhanced_search.template_manager == template_manager
        assert enhanced_search.context_service == context_service

    def test_template_and_context_integration(self):
        """Test that templates and context determination work together."""
        template_manager = SearchTemplateManager()
        context_service = ContextDeterminationService(template_manager)

        # Context service should use template manager
        assert context_service.template_manager == template_manager

        # Should be able to get contextual queries
        context = CodeGenerationContext(code_type="json")
        query = context_service.get_contextual_search_query(context)

        assert isinstance(query, str)
        assert len(query) > 0
