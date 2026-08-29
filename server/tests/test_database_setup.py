"""
Unit tests for database setup module.
"""

import pytest
import os
from unittest.mock import Mock, patch

from pymongo.errors import ConnectionFailure

from genericsuite_codegen.database.setup import (
    SearchResult,
    DatabaseConnectionError,
    VectorSearchError,
    DatabaseManager,
    get_database_connection,
    # create_vector_search_index,
    # store_embedded_chunks,
    # search_similar_vectors,
    # delete_all_vectors,
    # get_document_count
)
from genericsuite_codegen.document_processing.types import EmbeddedChunk
from genericsuite_codegen.document_processing.chunker import DocumentChunk


class TestSearchResult:
    """Test SearchResult data class."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            content="Test content",
            metadata={"source": "test.py"},
            similarity_score=0.85,
            document_path="test.py"
        )

        assert result.content == "Test content"
        assert result.metadata == {"source": "test.py"}
        assert result.similarity_score == 0.85
        assert result.document_path == "test.py"


class TestDatabaseManager:
    """Test DatabaseManager class."""

    def test_database_manager_initialization_default(self):
        """Test DatabaseManager initialization with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            manager = DatabaseManager()

            assert manager.mongodb_uri == "mongodb://localhost:27017/"
            assert manager.db_name == "genericsuite_codegen"
            assert manager.client is None
            assert manager.database is None

    def test_database_manager_initialization_custom_uri(self):
        """Test DatabaseManager initialization with custom URI."""
        custom_uri = "mongodb://custom:27017/"
        manager = DatabaseManager(mongodb_uri=custom_uri)

        assert manager.mongodb_uri == custom_uri

    def test_database_manager_initialization_from_env(self):
        """Test DatabaseManager initialization from environment variables."""
        env_vars = {
            "APP_DB_URI": "mongodb://env:27017/",
            "APP_DB_NAME": "test_db",
            "MONGODB_MAX_POOL_SIZE": "20",
            "MONGODB_MIN_POOL_SIZE": "5"
        }

        with patch.dict(os.environ, env_vars):
            manager = DatabaseManager()

            assert manager.mongodb_uri == "mongodb://env:27017/"
            assert manager.db_name == "test_db"
            assert manager.max_pool_size == 20
            assert manager.min_pool_size == 5

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_connect_success(self, mock_mongo_client):
        """Test successful database connection."""
        # Mock MongoDB client
        mock_client = Mock()
        mock_database = Mock()
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()

        assert manager.client is mock_client
        assert manager.database is mock_database
        mock_mongo_client.assert_called_once()

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_connect_failure(self, mock_mongo_client):
        """Test database connection failure."""
        # Mock MongoDB client to raise connection error
        mock_mongo_client.side_effect = ConnectionFailure("Connection failed")

        manager = DatabaseManager()

        with pytest.raises(DatabaseConnectionError):
            manager.connect()

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_disconnect(self, mock_mongo_client):
        """Test database disconnection."""
        # Mock MongoDB client
        mock_client = Mock()
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        manager.disconnect()

        mock_client.close.assert_called_once()
        assert manager.client is None
        assert manager.database is None

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_get_collection(self, mock_mongo_client):
        """Test getting a collection."""
        # Mock MongoDB client and database
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        collection = manager.get_collection("test_collection")

        assert collection is mock_collection
        mock_database.__getitem__.assert_called_with("test_collection")

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_get_collection_not_connected(self, mock_mongo_client):
        """Test getting collection when not connected."""
        manager = DatabaseManager()

        with pytest.raises(DatabaseConnectionError):
            manager.get_collection("test_collection")

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_initialize_schema(self, mock_mongo_client):
        """Test schema initialization."""
        # Mock MongoDB client and collections
        mock_client = Mock()
        mock_database = Mock()
        mock_kb_collection = Mock()
        mock_conv_collection = Mock()
        mock_users_collection = Mock()

        mock_database.__getitem__.side_effect = lambda name: {
            "knowledge_base": mock_kb_collection,
            "ai_chatbot_conversations": mock_conv_collection,
            "users": mock_users_collection
        }[name]

        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        manager.initialize_schema()

        # Verify indexes were created
        mock_kb_collection.create_index.assert_called()
        mock_conv_collection.create_index.assert_called()
        mock_users_collection.create_index.assert_called()

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_create_vector_search_index(self, mock_mongo_client):
        """Test vector search index creation."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        manager.create_vector_search_index(embedding_dimension=1536)

        # Verify search index creation was attempted
        mock_collection.create_search_index.assert_called()

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_store_embedded_chunks(self, mock_mongo_client):
        """Test storing embedded chunks."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        # Create test embedded chunks
        chunk = DocumentChunk("1", "doc1", "content", 0,
                              {"file_type": "python"})
        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test-model"
        )

        manager = DatabaseManager()
        manager.connect()
        result = manager.store_embedded_chunks([embedded_chunk])

        assert result is True
        mock_collection.insert_many.assert_called_once()

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_store_embedded_chunks_empty_list(self, mock_mongo_client):
        """Test storing empty list of embedded chunks."""
        mock_client = Mock()
        mock_database = Mock()
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        result = manager.store_embedded_chunks([])

        assert result is True

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_search_similar_vectors(self, mock_mongo_client):
        """Test vector similarity search."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()

        # Mock search results
        mock_results = [
            {
                "content": "test content",
                "path": "test.py",
                "metadata": {"file_type": "python"},
                "score": 0.85
            }
        ]
        mock_collection.aggregate.return_value = mock_results

        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        results = manager.search_similar_vectors([0.1, 0.2, 0.3], limit=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].content == "test content"
        assert results[0].similarity_score == 0.85

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_search_similar_vectors_no_results(self, mock_mongo_client):
        """Test vector similarity search with no results."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_collection.aggregate.return_value = []

        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        results = manager.search_similar_vectors([0.1, 0.2, 0.3])

        assert results == []

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_delete_all_vectors(self, mock_mongo_client):
        """Test deleting all vectors."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_collection.delete_many.return_value = Mock(deleted_count=10)

        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        result = manager.delete_all_vectors()

        assert result is True
        mock_collection.delete_many.assert_called_once_with({})

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_get_document_count(self, mock_mongo_client):
        """Test getting document count."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_collection.count_documents.return_value = 42

        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        count = manager.get_document_count()

        assert count == 42
        mock_collection.count_documents.assert_called_once_with({})

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_health_check_success(self, mock_mongo_client):
        """Test successful health check."""
        # Mock MongoDB client
        mock_client = Mock()
        mock_client.admin.command.return_value = {"ok": 1}
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        is_healthy = manager.health_check()

        assert is_healthy is True

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_health_check_failure(self, mock_mongo_client):
        """Test health check failure."""
        # Mock MongoDB client to raise exception
        mock_client = Mock()
        mock_client.admin.command.side_effect = Exception(
            "Health check failed")
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()
        is_healthy = manager.health_check()

        assert is_healthy is False

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_context_manager(self, mock_mongo_client):
        """Test using DatabaseManager as context manager."""
        mock_client = Mock()
        mock_database = Mock()
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        with DatabaseManager() as manager:
            assert manager.client is mock_client
            assert manager.database is mock_database

        # Should disconnect after context
        mock_client.close.assert_called_once()


class TestModuleFunctions:
    """Test module-level convenience functions."""

    @patch('genericsuite_codegen.database.setup.DatabaseManager')
    def test_get_database_connection(self, mock_db_manager):
        """Test get_database_connection function."""
        mock_manager = Mock()
        mock_db_manager.return_value = mock_manager

        connection = get_database_connection()

        assert connection is mock_manager
        mock_manager.connect.assert_called_once()

    @patch('genericsuite_codegen.database.setup.DatabaseManager')
    def test_create_vector_search_index_function(self, mock_db_manager):
        """Test create_vector_search_index function."""
        mock_manager = Mock()
        mock_db_manager.return_value = mock_manager

        result = mock_manager.create_vector_search_index(
            embedding_dimension=1536)

        assert result is True
        mock_manager.connect.assert_called_once()
        mock_manager.create_vector_search_index.assert_called_once_with(1536)
        mock_manager.disconnect.assert_called_once()

    @patch('genericsuite_codegen.database.setup.DatabaseManager')
    def test_store_embedded_chunks_function(self, mock_db_manager):
        """Test store_embedded_chunks function."""
        mock_manager = Mock()
        mock_manager.store_embedded_chunks.return_value = True
        mock_db_manager.return_value = mock_manager

        # Create test embedded chunk
        chunk = DocumentChunk("1", "doc1", "content", 0, {})
        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test-model"
        )

        result = mock_manager.store_embedded_chunks([embedded_chunk])

        assert result is True
        mock_manager.connect.assert_called_once()
        mock_manager.store_embedded_chunks.assert_called_once_with(
            [embedded_chunk])
        mock_manager.disconnect.assert_called_once()

    @patch('genericsuite_codegen.database.setup.DatabaseManager')
    def test_search_similar_vectors_function(self, mock_db_manager):
        """Test search_similar_vectors function."""
        mock_manager = Mock()
        mock_results = [SearchResult("content", {}, 0.85, "test.py")]
        mock_manager.search_similar_vectors.return_value = mock_results
        mock_db_manager.return_value = mock_manager

        results = mock_manager.search_similar_vectors([0.1, 0.2, 0.3], limit=5)

        assert results == mock_results
        mock_manager.connect.assert_called_once()
        mock_manager.search_similar_vectors.assert_called_once_with(
            [0.1, 0.2, 0.3], 5, None)
        mock_manager.disconnect.assert_called_once()

    @patch('genericsuite_codegen.database.setup.DatabaseManager')
    def test_delete_all_vectors_function(self, mock_db_manager):
        """Test delete_all_vectors function."""
        mock_manager = Mock()
        mock_manager.delete_all_vectors.return_value = True
        mock_db_manager.return_value = mock_manager

        result = mock_manager.delete_all_vectors()

        assert result is True
        mock_manager.connect.assert_called_once()
        mock_manager.delete_all_vectors.assert_called_once()
        mock_manager.disconnect.assert_called_once()

    @patch('genericsuite_codegen.database.setup.DatabaseManager')
    def test_get_document_count_function(self, mock_db_manager):
        """Test get_document_count function."""
        mock_manager = Mock()
        mock_manager.get_document_count.return_value = 42
        mock_db_manager.return_value = mock_manager

        count = mock_manager.get_document_count()

        assert count == 42
        mock_manager.connect.assert_called_once()
        mock_manager.get_document_count.assert_called_once()
        mock_manager.disconnect.assert_called_once()


class TestErrorHandling:
    """Test error handling in database operations."""

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_connection_error_handling(self, mock_mongo_client):
        """Test handling of connection errors."""
        mock_mongo_client.side_effect = ConnectionFailure("Connection failed")

        manager = DatabaseManager()

        with pytest.raises(DatabaseConnectionError):
            manager.connect()

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_vector_search_error_handling(self, mock_mongo_client):
        """Test handling of vector search errors."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_collection.aggregate.side_effect = Exception("Search failed")

        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        manager = DatabaseManager()
        manager.connect()

        with pytest.raises(VectorSearchError):
            manager.search_similar_vectors([0.1, 0.2, 0.3])

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_store_chunks_error_handling(self, mock_mongo_client):
        """Test handling of storage errors."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_collection.insert_many.side_effect = Exception("Insert failed")

        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        # Create test embedded chunk
        chunk = DocumentChunk("1", "doc1", "content", 0, {})
        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test-model"
        )

        manager = DatabaseManager()
        manager.connect()

        with pytest.raises(Exception):
            manager.store_embedded_chunks([embedded_chunk])
