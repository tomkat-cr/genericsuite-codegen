"""
Integration tests for API endpoints.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from genericsuite_codegen.api.main import app
from genericsuite_codegen.api.types import Conversation
from genericsuite_codegen.agent.types import AgentResponse


def generate_object_id() -> str:
    """
    Generate a valid ObjectId as a 24-character hex string.

    Returns:
        str: A valid 24-character hex ObjectId string.
    """
    import secrets
    return secrets.token_hex(12)


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""

    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)

    @patch('genericsuite_codegen.agent.agent.GenericSuiteAgent.query')
    def test_query_endpoint_integration(self, mock_agent_query):
        """Test query endpoint integration."""

        mock_agent_response = AgentResponse(
            content="Test response from agent",
            sources=["source1.py", "source2.py"],
            task_type="general",
            model_used="gpt-4o",
            token_usage={
                "prompt_tokens": 50,
                "completion_tokens": 50,
                "total_tokens": 100
            },
        )
        mock_agent_query.return_value = mock_agent_response

        # Test query request
        query_data = {
            "query": "How do I create a GenericSuite table configuration?",
            "user_id": "test_user",
            "conversation_id": None
        }

        response = self.client.post("/query", json=query_data)

        # print(f"response.status_code: {response.status_code}")
        # print(f"response.read(): {response.read()}")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data['data']["content"] == "Test response from agent"
        assert data['data']["sources"] == ["source1.py", "source2.py"]
        assert data['data']["model_used"] == "gpt-4o"
        assert data['data']["token_usage"] is not None
        assert data['data']["token_usage"]["prompt_tokens"] == 50
        assert data['data']["token_usage"]["completion_tokens"] == 50
        assert data['data']["token_usage"]["total_tokens"] == 100
        assert data['data']["task_type"] == "general"
        assert "conversation_id" in data
        assert data['data']["conversation_id"] is not None
        assert data['data']["timestamp"] is not None

    @patch('genericsuite_codegen.api.endpoint_methods.get_database_connection')
    def test_conversations_endpoint_integration(self, mock_get_db):
        """Test conversations endpoints integration."""
        # Mock database
        mock_db = Mock()
        mock_conversations = [
            Conversation(
                id="conv1",
                title="Test Conversation 1",
                user_id="test_user",
                message_count=0,
                messages=[],
                created_at="2023-01-01T00:00:00",
                updated_at="2023-01-01T00:00:00"
            )
        ]
        mock_db.get_conversations.return_value = mock_conversations
        mock_get_db.return_value = mock_db

        # Test GET conversations
        response = self.client.get("/conversations")

        assert response.status_code == 200
        data = response.json()

        print(f"response.status_code: {response.status_code}")
        print(f"response.read(): {response.read()}")

        assert data["success"] is True
        assert data["data"]["timestamp"] is not None
        assert data["data"]["page"] == 1
        assert data["data"]["page_size"] == 20
        assert data["data"]["total"] == 1
        assert len(data["data"]["conversations"]) == 1
        assert data["data"]["conversations"][0]["title"] == \
            "Test Conversation 1"

    @patch('genericsuite_codegen.api.endpoint_methods.get_database_connection')
    def test_create_conversation_integration(self, mock_get_db):
        """Test conversation creation integration."""
        # Mock database
        mock_db = Mock()
        mock_db.create_conversation.return_value = generate_object_id()
        mock_get_db.return_value = mock_db

        # Test POST conversation
        conversation_data = {
            "title": "New Test Conversation",
            "user_id": "test_user"
        }

        response = self.client.post(
            "/conversations", json=conversation_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["conversation_id"] == generate_object_id()

    @patch('genericsuite_codegen.api.endpoint_methods.run_ingestion_pipeline')
    def test_knowledge_base_update_integration(self, mock_ingestion):
        """Test knowledge base update integration."""
        # Mock ingestion pipeline
        mock_ingestion_result = {
            "documents_processed": 50,
            "chunks_created": 200,
            "embeddings_generated": 200,
            "processing_time": 30.5
        }
        mock_ingestion.return_value = mock_ingestion_result

        # Test knowledge base update
        update_data = {
            "repository_url": "https://github.com/test/repo.git",
            "force_refresh": True
        }

        response = self.client.post(
            "/update-knowledge-base", json=update_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["documents_processed"] == 50
        assert data["data"]["chunks_created"] == 200

    @patch('genericsuite_codegen.api.endpoint_methods.get_database_connection')
    def test_knowledge_base_status_integration(self, mock_get_db):
        """Test knowledge base status integration."""
        # Mock database
        mock_db = Mock()
        mock_db.get_document_count.return_value = 100
        mock_db.get_last_update_time.return_value = "2023-01-01T00:00:00"
        mock_get_db.return_value = mock_db

        response = self.client.get("/update-knowledge-base/status")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["document_count"] == 100
        assert "last_updated" in data["data"]

    @patch('genericsuite_codegen.agent.agent.GenericSuiteAgent.query')
    def test_file_generation_integration(self, mock_agent_query):
        """Test file generation integration."""
        # Mock agent
        mock_agent_response = AgentResponse(
            content="```python\ndef hello():\n"
            "    print('Hello, World!')\n```",
            sources=[],
            task_type="general",
            model_used="gpt-4o",
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            },
        )
        mock_agent_query.return_value = mock_agent_response

        # Mock code extraction
        with patch('genericsuite_codegen.api.endpoint_methods'
                   '.extract_code_blocks') as mock_extract:
            mock_extract.return_value = [
                {
                    "language": "python",
                    "code": "def hello():\n    print('Hello, World!')",
                    "filename": "hello.py"
                }
            ]

            # Test file generation
            generation_data = {
                "file_type": "python",
                "requirements": "Create a hello world function",
                "user_id": "test_user"
            }

            response = self.client.post(
                "/generate/files", json=generation_data)

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert len(data["data"]["files"]) == 1
            assert data["data"]["files"][0]["filename"] == "hello.py"

    @patch('genericsuite_codegen.api.endpoint_methods.get_database_connection')
    @patch(
        'genericsuite_codegen.api.endpoint_methods.generate_query_embedding')
    def test_search_integration(self, mock_generate_embedding, mock_get_db):
        """Test search endpoint integration."""
        # Mock embedding generation
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]

        # Mock database search
        mock_db = Mock()
        mock_search_results = [
            {
                "content": "Test search result",
                "metadata": {"source": "test.py"},
                "similarity_score": 0.9,
                "document_path": "test.py"
            }
        ]
        mock_db.search_similar_vectors.return_value = mock_search_results
        mock_get_db.return_value = mock_db

        # Test search
        search_data = {
            "query": "test search query",
            "limit": 5,
            "user_id": "test_user"
        }

        response = self.client.post("/search", json=search_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]["results"]) == 1
        assert data["data"]["results"][0]["similarity_score"] == 0.9

    @patch('genericsuite_codegen.api.endpoint_methods.get_database_connection')
    @patch('genericsuite_codegen.agent.agent.GenericSuiteAgent.health_check')
    def test_health_check_integration(self, mock_agent_health_check,
                                      mock_get_db):
        """Test health check integration."""
        # Mock healthy components
        mock_db = Mock()
        mock_db.health_check.return_value = True
        mock_get_db.return_value = mock_db

        mock_agent_health_check.return_value = {
            "status": "healthy",
            "model": "gpt-4o",
            "provider": "openai",
            "test_response_length": 100,
            "sources_available": True,
            "enhanced_search": {
                "status": "healthy",
                "components": {
                    "database": True,
                    "agent": True
                }
            }
        }

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
        assert data["data"]["components"]["database"] is True
        assert data["data"]["components"]["agent"] is True

    @patch('genericsuite_codegen.api.endpoint_methods.get_database_connection')
    def test_statistics_integration(self, mock_get_db):
        """Test statistics endpoint integration."""
        # Mock database statistics
        mock_db = Mock()
        mock_db.get_document_count.return_value = 100
        mock_db.get_conversation_count.return_value = 25
        mock_db.get_user_count.return_value = 5
        mock_get_db.return_value = mock_db

        response = self.client.get("/statistics")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["knowledge_base"]["document_count"] == 100
        assert data["data"]["conversations"]["total_conversations"] == 25
        assert data["data"]["system"]["total_users"] == 5

    def test_error_handling_integration(self):
        """Test API error handling integration."""
        # Test invalid request data
        response = self.client.post("/query", json={})

        assert response.status_code == 422  # Validation error

        # Test non-existent endpoint
        response = self.client.get("/nonexistent")

        assert response.status_code == 404

    @patch('genericsuite_codegen.api.endpoint_methods.get_agent')
    def test_streaming_response_integration(self, mock_agent_query):
        """Test streaming response integration."""
        # Mock agent with streaming response
        mock_agent = Mock()
        mock_agent_response = AgentResponse(
            content="This is a long response that should be streamed",
            sources=[],
            task_type="general",
            model_used="gpt-4o",
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            },
        )
        mock_agent.query_async = AsyncMock(return_value=mock_agent_response)
        mock_agent_query.return_value = mock_agent

        # Test streaming query
        query_data = {
            "query": "Generate a long response",
            "user_id": "test_user",
            "stream": True
        }

        response = self.client.post("/query", json=query_data)

        # Should still return 200 even for streaming
        assert response.status_code == 200

    def test_cors_integration(self):
        """Test CORS integration."""
        # Test preflight request
        response = self.client.options(
            "/query",
            headers={
                "Origin": "http://localhost:3002",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )

        # Should handle CORS preflight
        assert response.status_code in [200, 204]

    @patch('genericsuite_codegen.api.endpoint_methods'
           '.process_uploaded_document')
    def test_file_upload_integration(self, mock_process_document):
        """Test file upload integration."""
        # Mock document processing
        mock_process_result = {
            "document_id": "doc123",
            "chunks_created": 5,
            "embeddings_generated": 5
        }
        mock_process_document.return_value = mock_process_result

        # Test file upload
        test_file_content = b"print('test file content')"

        response = self.client.post(
            "/upload-document",
            files={"file": ("test.py", test_file_content, "text/plain")},
            data={"user_id": "test_user"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["document_id"] == "doc123"

    def test_request_validation_integration(self):
        """Test request validation integration."""
        # Test missing required fields
        response = self.client.post("/query", json={"query": ""})
        assert response.status_code == 422

        # Test invalid data types
        response = self.client.post("/query", json={"query": 123})
        assert response.status_code == 422

        # Test valid request
        with patch('genericsuite_codegen.api.endpoint_methods.get_agent') \
                as mock_agent_query:
            mock_agent = Mock()
            mock_agent_response = AgentResponse(
                content="Valid response",
                sources=[],
                task_type="general",
                model_used="gpt-4o",
                token_usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300
                },
            )
            mock_agent.query_async = AsyncMock(
                return_value=mock_agent_response)
            mock_agent_query.return_value = mock_agent

            response = self.client.post("/query", json={
                "query": "Valid query",
                "user_id": "test_user"
            })
            assert response.status_code == 200

    def test_rate_limiting_integration(self):
        """Test rate limiting integration (if implemented)."""
        # This would test rate limiting if implemented
        # For now, just verify multiple requests work

        with patch('genericsuite_codegen.api.endpoint_methods.get_agent') \
                as mock_agent_query:
            mock_agent = Mock()
            mock_agent_response = AgentResponse(
                content="Response",
                sources=[],
                task_type="general",
                model_used="gpt-4o",
                token_usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300
                },
            )
            mock_agent.query_async = AsyncMock(
                return_value=mock_agent_response)
            mock_agent_query.return_value = mock_agent

            # Make multiple requests
            for i in range(5):
                response = self.client.post("/query", json={
                    "query": f"Query {i}",
                    "user_id": "test_user"
                })
                assert response.status_code == 200
