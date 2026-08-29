"""
Unit tests for API endpoint methods.
"""

import pytest
import uuid
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from genericsuite_codegen.api.endpoint_methods import EndpointMethods
from genericsuite_codegen.api.types import (
    QueryRequest,
    QueryResponse,
    ConversationCreate,
    ConversationUpdate,
    Conversation,
    KnowledgeBaseUpdate,
    KnowledgeBaseStatus,
    DocumentInfo,
    FileGenerationRequest,
    GeneratedFile,
    SearchQuery,
    KnowledgeBaseSearchResults,
    HealthResponse
)
from genericsuite_codegen.agent.types import AgentResponse, AgentContext
from genericsuite_codegen.conversations.service import ConversationsService


class TestEndpointMethods:
    """Test EndpointMethods class."""

    def setup_method(self):
        """Set up test environment."""
        with patch('genericsuite_codegen.api.endpoint_methods.get_database_connection'), \
                patch('genericsuite_codegen.api.endpoint_methods.get_agent'):
            self.endpoint_methods = EndpointMethods()

    @patch('genericsuite_codegen.api.endpoint_methods.get_database_connection')
    @patch('genericsuite_codegen.api.endpoint_methods.get_agent')
    def test_initialization(self, mock_get_agent, mock_get_db):
        """Test EndpointMethods initialization."""
        mock_db = Mock()
        mock_agent = Mock()
        mock_get_db.return_value = mock_db
        mock_get_agent.return_value = mock_agent

        endpoint_methods = EndpointMethods()

        assert endpoint_methods.db is mock_db
        assert endpoint_methods.agent is mock_agent

    async def test_query_agent_success(self):
        """Test successful agent query."""
        # Mock agent response
        mock_agent_response = AgentResponse(
            response="Test response",
            success=True,
            sources=["source1.py", "source2.py"],
            metadata={"tokens_used": 100}
        )

        self.endpoint_methods.agent.query_async = AsyncMock(
            return_value=mock_agent_response)

        request = QueryRequest(
            query="Test query",
            conversation_id="conv_123",
            user_id="user_123"
        )

        correlation_id = str(uuid.uuid4())

        result = await self.endpoint_methods.query_agent(request, correlation_id)

        assert result["success"] is True
        assert result["data"]["response"] == "Test response"
        assert result["data"]["sources"] == ["source1.py", "source2.py"]
        assert result["correlation_id"] == correlation_id

    async def test_query_agent_failure(self):
        """Test agent query failure."""
        # Mock agent response with error
        mock_agent_response = AgentResponse(
            response="",
            success=False,
            error="Query failed",
            sources=[],
            metadata={}
        )

        self.endpoint_methods.agent.query_async = AsyncMock(
            return_value=mock_agent_response)

        request = QueryRequest(
            query="Test query",
            conversation_id="conv_123",
            user_id="user_123"
        )

        correlation_id = str(uuid.uuid4())

        result = await self.endpoint_methods.query_agent(request, correlation_id)

        assert result["success"] is False
        assert "Query failed" in result["error"]

    async def test_query_agent_exception(self):
        """Test agent query with exception."""
        self.endpoint_methods.agent.query_async = AsyncMock(
            side_effect=Exception("Unexpected error"))

        request = QueryRequest(
            query="Test query",
            conversation_id="conv_123",
            user_id="user_123"
        )

        correlation_id = str(uuid.uuid4())

        result = await self.endpoint_methods.query_agent(request, correlation_id)

        assert result["success"] is False
        assert "Unexpected error" in result["error"]

    def test_get_conversations_success(self):
        """Test successful conversation retrieval."""
        # Mock database response
        mock_conversations = [
            {
                "_id": "conv_1",
                "title": "Test Conversation 1",
                "user_id": "user_123",
                "messages": [],
                "creation_date": datetime.now(),
                "update_date": datetime.now()
            },
            {
                "_id": "conv_2",
                "title": "Test Conversation 2",
                "user_id": "user_123",
                "messages": [],
                "creation_date": datetime.now(),
                "update_date": datetime.now()
            }
        ]

        conversations = ConversationsService()
        conversations.list = Mock(
            return_value=mock_conversations)

        result = conversations.list(
            "user_123", limit=10, offset=0)

        assert result["success"] is True
        assert len(result["data"]["conversations"]) == 2
        assert result["data"]["total"] == 2

    def test_get_conversations_database_error(self):
        """Test conversation retrieval with database error."""
        conversations = ConversationsService()
        conversations.list = Mock(
            side_effect=Exception("Database error"))
        result = conversations.list("user_123")

        assert result["success"] is False
        assert "Database error" in result["error"]

    def test_create_conversation_success(self):
        """Test successful conversation creation."""
        mock_conversation_id = "conv_new"
        self.endpoint_methods.db.create_conversation = Mock(
            return_value=mock_conversation_id)

        request = ConversationCreate(
            title="New Conversation",
            user_id="user_123"
        )

        result = self.endpoint_methods.create_conversation(request)

        assert result["success"] is True
        assert result["data"]["conversation_id"] == mock_conversation_id

    def test_create_conversation_database_error(self):
        """Test conversation creation with database error."""
        self.endpoint_methods.db.create_conversation = Mock(
            side_effect=Exception("Creation failed"))

        request = ConversationCreate(
            title="New Conversation",
            user_id="user_123"
        )

        result = self.endpoint_methods.create_conversation(request)

        assert result["success"] is False
        assert "Creation failed" in result["error"]

    def test_update_conversation_success(self):
        """Test successful conversation update."""
        self.endpoint_methods.db.update_conversation = Mock(return_value=True)

        request = ConversationUpdate(
            conversation_id="conv_123",
            title="Updated Title"
        )

        conversation = ConversationsService()
        result = conversation.update(request, "user_123")

        assert result["success"] is True

    def test_update_conversation_not_found(self):
        """Test conversation update when conversation not found."""
        self.endpoint_methods.db.update_conversation = Mock(return_value=False)

        request = ConversationUpdate(
            conversation_id="nonexistent",
            title="Updated Title"
        )

        conversation = ConversationsService()
        result = conversation.update(request, "user_123")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_delete_conversation_success(self):
        """Test successful conversation deletion."""
        self.endpoint_methods.db.delete_conversation = Mock(return_value=True)

        result = self.endpoint_methods.delete_conversation(
            "conv_123", "user_123")

        assert result["success"] is True

    def test_delete_conversation_not_found(self):
        """Test conversation deletion when conversation not found."""
        self.endpoint_methods.db.delete_conversation = Mock(return_value=False)

        result = self.endpoint_methods.delete_conversation(
            "nonexistent", "user_123")

        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_update_knowledge_base_success(self):
        """Test successful knowledge base update."""
        # Mock ingestion process
        mock_ingestion_result = {
            "documents_processed": 100,
            "chunks_created": 500,
            "embeddings_generated": 500
        }

        with patch('genericsuite_codegen.api.endpoint_methods.run_ingestion_pipeline') as mock_ingestion:
            mock_ingestion.return_value = mock_ingestion_result

            request = KnowledgeBaseUpdate(
                repository_url="https://github.com/test/repo.git",
                force_refresh=True
            )

            result = await self.endpoint_methods.update_knowledge_base(request)

            assert result["success"] is True
            assert result["data"]["documents_processed"] == 100

    async def test_update_knowledge_base_failure(self):
        """Test knowledge base update failure."""
        with patch('genericsuite_codegen.api.endpoint_methods.run_ingestion_pipeline') as mock_ingestion:
            mock_ingestion.side_effect = Exception("Ingestion failed")

            request = KnowledgeBaseUpdate(
                repository_url="https://github.com/test/repo.git"
            )

            result = await self.endpoint_methods.update_knowledge_base(request)

            assert result["success"] is False
            assert "Ingestion failed" in result["error"]

    def test_get_knowledge_base_status_success(self):
        """Test successful knowledge base status retrieval."""
        # Mock database statistics
        self.endpoint_methods.db.get_document_count = Mock(return_value=100)
        self.endpoint_methods.db.get_last_update_time = Mock(
            return_value=datetime.now())

        result = self.endpoint_methods.get_knowledge_base_status()

        assert result["success"] is True
        assert result["data"]["document_count"] == 100
        assert "last_updated" in result["data"]

    def test_get_knowledge_base_status_database_error(self):
        """Test knowledge base status with database error."""
        self.endpoint_methods.db.get_document_count = Mock(
            side_effect=Exception("Database error"))

        result = self.endpoint_methods.get_knowledge_base_status()

        assert result["success"] is False
        assert "Database error" in result["error"]

    async def test_upload_document_success(self):
        """Test successful document upload."""
        # Mock file upload
        mock_file = Mock()
        mock_file.filename = "test.py"
        mock_file.content_type = "text/plain"
        mock_file.read = AsyncMock(return_value=b"print('hello')")

        with patch('genericsuite_codegen.api.endpoint_methods.process_uploaded_document') as mock_process:
            mock_process.return_value = {
                "document_id": "doc_123",
                "chunks_created": 5,
                "embeddings_generated": 5
            }

            result = await self.endpoint_methods.upload_document(mock_file, "user_123")

            assert result["success"] is True
            assert result["data"]["document_id"] == "doc_123"

    async def test_upload_document_invalid_file(self):
        """Test document upload with invalid file."""
        # Mock invalid file
        mock_file = Mock()
        mock_file.filename = "image.png"
        mock_file.content_type = "image/png"

        result = await self.endpoint_methods.upload_document(mock_file, "user_123")

        assert result["success"] is False
        assert "not supported" in result["error"]

    async def test_upload_document_processing_error(self):
        """Test document upload with processing error."""
        mock_file = Mock()
        mock_file.filename = "test.py"
        mock_file.content_type = "text/plain"
        mock_file.read = AsyncMock(return_value=b"print('hello')")

        with patch('genericsuite_codegen.api.endpoint_methods.process_uploaded_document') as mock_process:
            mock_process.side_effect = Exception("Processing failed")

            result = await self.endpoint_methods.upload_document(mock_file, "user_123")

            assert result["success"] is False
            assert "Processing failed" in result["error"]

    async def test_generate_files_success(self):
        """Test successful file generation."""
        # Mock agent response with code
        mock_agent_response = AgentResponse(
            response="```python\nprint('hello')\n```",
            success=True,
            sources=[],
            metadata={}
        )

        self.endpoint_methods.agent.query_async = AsyncMock(
            return_value=mock_agent_response)

        request = FileGenerationRequest(
            file_type="python",
            requirements="Create a hello world script",
            user_id="user_123"
        )

        with patch('genericsuite_codegen.api.endpoint_methods.extract_code_blocks') as mock_extract:
            mock_extract.return_value = [
                {"language": "python",
                    "code": "print('hello')", "filename": "hello.py"}
            ]

            result = await self.endpoint_methods.generate_files(request)

            assert result["success"] is True
            assert len(result["data"]["files"]) == 1
            assert result["data"]["files"][0]["filename"] == "hello.py"

    async def test_generate_files_no_code_found(self):
        """Test file generation when no code is found in response."""
        mock_agent_response = AgentResponse(
            response="No code generated",
            success=True,
            sources=[],
            metadata={}
        )

        self.endpoint_methods.agent.query_async = AsyncMock(
            return_value=mock_agent_response)

        request = FileGenerationRequest(
            file_type="python",
            requirements="Create a hello world script",
            user_id="user_123"
        )

        with patch('genericsuite_codegen.api.endpoint_methods.extract_code_blocks') as mock_extract:
            mock_extract.return_value = []

            result = await self.endpoint_methods.generate_files(request)

            assert result["success"] is False
            assert "No code blocks found" in result["error"]

    async def test_search_knowledge_base_success(self):
        """Test successful knowledge base search."""
        # Mock search results
        mock_search_results = [
            {
                "content": "Test content 1",
                "metadata": {"source": "test1.py"},
                "similarity_score": 0.9,
                "document_path": "test1.py"
            },
            {
                "content": "Test content 2",
                "metadata": {"source": "test2.py"},
                "similarity_score": 0.8,
                "document_path": "test2.py"
            }
        ]

        self.endpoint_methods.db.search_similar_vectors = Mock(
            return_value=mock_search_results)

        # Mock embedding generation
        with patch('genericsuite_codegen.api.endpoint_methods.generate_query_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]

            query = SearchQuery(
                query="test query",
                limit=10,
                user_id="user_123"
            )

            result = await self.endpoint_methods.search_knowledge_base(query)

            assert result["success"] is True
            assert len(result["data"]["results"]) == 2
            assert result["data"]["results"][0]["similarity_score"] == 0.9

    async def test_search_knowledge_base_embedding_error(self):
        """Test knowledge base search with embedding generation error."""
        with patch('genericsuite_codegen.api.endpoint_methods.generate_query_embedding') as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")

            query = SearchQuery(
                query="test query",
                limit=10,
                user_id="user_123"
            )

            result = await self.endpoint_methods.search_knowledge_base(query)

            assert result["success"] is False
            assert "Embedding failed" in result["error"]

    def test_get_statistics_success(self):
        """Test successful statistics retrieval."""
        # Mock database statistics
        self.endpoint_methods.db.get_document_count = Mock(return_value=100)
        self.endpoint_methods.db.get_conversation_count = Mock(return_value=50)
        self.endpoint_methods.db.get_user_count = Mock(return_value=10)

        result = self.endpoint_methods.get_statistics()

        assert result["success"] is True
        assert result["data"]["knowledge_base"]["document_count"] == 100
        assert result["data"]["conversations"]["total_conversations"] == 50
        assert result["data"]["system"]["total_users"] == 10

    def test_get_statistics_database_error(self):
        """Test statistics retrieval with database error."""
        self.endpoint_methods.db.get_document_count = Mock(
            side_effect=Exception("Database error"))

        result = self.endpoint_methods.get_statistics()

        assert result["success"] is False
        assert "Database error" in result["error"]

    def test_health_check_success(self):
        """Test successful health check."""
        # Mock healthy components
        self.endpoint_methods.db.health_check = Mock(return_value=True)
        self.endpoint_methods.agent.health_check = Mock(return_value=True)

        result = self.endpoint_methods.health_check()

        assert result["success"] is True
        assert result["data"]["status"] == "healthy"
        assert result["data"]["components"]["database"] is True
        assert result["data"]["components"]["agent"] is True

    def test_health_check_database_unhealthy(self):
        """Test health check with unhealthy database."""
        self.endpoint_methods.db.health_check = Mock(return_value=False)
        self.endpoint_methods.agent.health_check = Mock(return_value=True)

        result = self.endpoint_methods.health_check()

        assert result["success"] is True
        assert result["data"]["status"] == "degraded"
        assert result["data"]["components"]["database"] is False

    def test_health_check_agent_unhealthy(self):
        """Test health check with unhealthy agent."""
        self.endpoint_methods.db.health_check = Mock(return_value=True)
        self.endpoint_methods.agent.health_check = Mock(return_value=False)

        result = self.endpoint_methods.health_check()

        assert result["success"] is True
        assert result["data"]["status"] == "degraded"
        assert result["data"]["components"]["agent"] is False

    def test_health_check_all_unhealthy(self):
        """Test health check with all components unhealthy."""
        self.endpoint_methods.db.health_check = Mock(return_value=False)
        self.endpoint_methods.agent.health_check = Mock(return_value=False)

        result = self.endpoint_methods.health_check()

        assert result["success"] is True
        assert result["data"]["status"] == "unhealthy"

    def test_health_check_exception(self):
        """Test health check with exception."""
        self.endpoint_methods.db.health_check = Mock(
            side_effect=Exception("Health check failed"))

        result = self.endpoint_methods.health_check()

        assert result["success"] is False
        assert "Health check failed" in result["error"]


class TestEndpointMethodsValidation:
    """Test input validation in endpoint methods."""

    def setup_method(self):
        """Set up test environment."""
        with patch('genericsuite_codegen.api.endpoint_methods.get_database_connection'), \
                patch('genericsuite_codegen.api.endpoint_methods.get_agent'):
            self.endpoint_methods = EndpointMethods()

    async def test_query_agent_empty_query(self):
        """Test agent query with empty query."""
        request = QueryRequest(
            query="",
            user_id="user_123"
        )

        correlation_id = str(uuid.uuid4())

        result = await self.endpoint_methods.query_agent(request, correlation_id)

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    async def test_query_agent_missing_user_id(self):
        """Test agent query without user ID."""
        request = QueryRequest(
            query="Test query"
        )

        correlation_id = str(uuid.uuid4())

        result = await self.endpoint_methods.query_agent(request, correlation_id)

        assert result["success"] is False
        assert "user_id" in result["error"].lower()

    def test_create_conversation_empty_title(self):
        """Test conversation creation with empty title."""
        request = ConversationCreate(
            title="",
            user_id="user_123"
        )

        result = self.endpoint_methods.create_conversation(request)

        assert result["success"] is False
        assert "title" in result["error"].lower()

    async def test_upload_document_no_filename(self):
        """Test document upload without filename."""
        mock_file = Mock()
        mock_file.filename = None

        result = await self.endpoint_methods.upload_document(mock_file, "user_123")

        assert result["success"] is False
        assert "filename" in result["error"].lower()

    async def test_generate_files_invalid_file_type(self):
        """Test file generation with invalid file type."""
        request = FileGenerationRequest(
            file_type="invalid_type",
            requirements="Create something",
            user_id="user_123"
        )

        result = await self.endpoint_methods.generate_files(request)

        assert result["success"] is False
        assert "file_type" in result["error"].lower()

    async def test_search_knowledge_base_empty_query(self):
        """Test knowledge base search with empty query."""
        query = SearchQuery(
            query="",
            user_id="user_123"
        )

        result = await self.endpoint_methods.search_knowledge_base(query)

        assert result["success"] is False
        assert "query" in result["error"].lower()


class TestEndpointMethodsUtilities:
    """Test utility functions used by endpoint methods."""

    def setup_method(self):
        """Set up test environment."""
        with patch('genericsuite_codegen.api.endpoint_methods.get_database_connection'), \
                patch('genericsuite_codegen.api.endpoint_methods.get_agent'):
            self.endpoint_methods = EndpointMethods()

    def test_validate_file_type_supported(self):
        """Test file type validation for supported types."""
        assert self.endpoint_methods._validate_file_type("python") is True
        assert self.endpoint_methods._validate_file_type("json") is True
        assert self.endpoint_methods._validate_file_type("javascript") is True

    def test_validate_file_type_unsupported(self):
        """Test file type validation for unsupported types."""
        assert self.endpoint_methods._validate_file_type("invalid") is False
        assert self.endpoint_methods._validate_file_type("") is False

    def test_sanitize_user_input(self):
        """Test user input sanitization."""
        # Test normal input
        clean_input = self.endpoint_methods._sanitize_user_input(
            "normal input")
        assert clean_input == "normal input"

        # Test input with potential injection
        malicious_input = "<script>alert('xss')</script>"
        clean_input = self.endpoint_methods._sanitize_user_input(
            malicious_input)
        assert "<script>" not in clean_input

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        correlation_id = self.endpoint_methods._generate_correlation_id()

        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0

        # Should generate unique IDs
        another_id = self.endpoint_methods._generate_correlation_id()
        assert correlation_id != another_id

    def test_format_error_response(self):
        """Test error response formatting."""
        error = Exception("Test error")
        correlation_id = "test_correlation_id"

        response = self.endpoint_methods._format_error_response(
            error, correlation_id)

        assert response["success"] is False
        assert "Test error" in response["error"]
        assert response["correlation_id"] == correlation_id

    def test_format_success_response(self):
        """Test success response formatting."""
        data = {"key": "value"}
        correlation_id = "test_correlation_id"

        response = self.endpoint_methods._format_success_response(
            data, correlation_id)

        assert response["success"] is True
        assert response["data"] == data
        assert response["correlation_id"] == correlation_id
