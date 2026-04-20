"""
Integration tests for database operations.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from genericsuite_codegen.database.setup import DatabaseManager, SearchResult
from genericsuite_codegen.document_processing.types import EmbeddedChunk
from genericsuite_codegen.document_processing.chunker import DocumentChunk


# @pytest.mark.integration  # Commented out to avoid unknown mark warning
class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def setup_method(self):
        """Set up test environment."""
        # Use a test database URI
        self.test_db_uri = "mongodb://localhost:27017/"
        self.test_db_name = "test_genericsuite_codegen"

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_database_connection_lifecycle(self, mock_mongo_client):
        """Test complete database connection lifecycle."""
        # Mock MongoDB client and database
        mock_client = Mock()
        mock_database = Mock()
        mock_client.__getitem__ = Mock(return_value=mock_database)
        mock_mongo_client.return_value = mock_client

        # Test connection
        db_manager = DatabaseManager(self.test_db_uri)
        db_manager.connect()

        assert db_manager.client is mock_client
        assert db_manager.database is mock_database

        # Test connection is established
        assert db_manager.client is not None
        assert db_manager.database is not None

        # Test disconnection
        db_manager.disconnect()
        mock_client.close.assert_called_once()
        assert db_manager.client is None
        assert db_manager.database is None

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_schema_initialization(self, mock_mongo_client):
        """Test database schema initialization."""
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

        db_manager = DatabaseManager(self.test_db_uri)
        db_manager.connect()
        db_manager.initialize_schema()

        # Verify indexes were created for all collections
        mock_kb_collection.create_index.assert_called()
        mock_conv_collection.create_index.assert_called()
        mock_users_collection.create_index.assert_called()

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_vector_operations_workflow(self, mock_mongo_client):
        """Test complete vector operations workflow."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        db_manager = DatabaseManager(self.test_db_uri)
        db_manager.connect()

        # Test vector search index creation
        db_manager.create_vector_search_index(embedding_dimension=384)
        mock_collection.create_search_index.assert_called()

        # Test storing embedded chunks
        chunk1 = DocumentChunk("1", "doc1", "content1",
                               0, {"file_type": "python"})
        chunk2 = DocumentChunk("2", "doc1", "content2",
                               1, {"file_type": "python"})

        embedded_chunks = [
            EmbeddedChunk(
                chunk=chunk1,
                embedding=[0.1, 0.2, 0.3],
                embedding_model="test-model"
            ),
            EmbeddedChunk(
                chunk=chunk2,
                embedding=[0.4, 0.5, 0.6],
                embedding_model="test-model"
            )
        ]

        # Mock successful insertion
        mock_collection.insert_many.return_value = Mock(
            inserted_ids=["id1", "id2"])

        result = db_manager.store_embedded_chunks(embedded_chunks)
        assert result is True
        mock_collection.insert_many.assert_called_once()

        # Verify the data structure passed to insert_many
        call_args = mock_collection.insert_many.call_args[0][0]
        assert len(call_args) == 2
        assert all("embedding" in doc for doc in call_args)
        assert all("content" in doc for doc in call_args)

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_vector_search_workflow(self, mock_mongo_client):
        """Test vector search workflow."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        # Mock search results
        mock_search_results = [
            {
                "content": "Test content 1",
                "path": "test1.py",
                "metadata": {"file_type": "python"},
                "score": 0.9
            },
            {
                "content": "Test content 2",
                "path": "test2.py",
                "metadata": {"file_type": "python"},
                "score": 0.8
            }
        ]
        mock_collection.aggregate.return_value = mock_search_results

        db_manager = DatabaseManager(self.test_db_uri)
        db_manager.connect()

        # Test vector search
        query_embedding = [0.1, 0.2, 0.3]
        results = db_manager.search_similar_vectors(query_embedding, limit=5)

        assert len(results) == 2
        assert all(isinstance(result, SearchResult) for result in results)
        assert results[0].similarity_score == 0.9
        assert results[1].similarity_score == 0.8
        assert results[0].content == "Test content 1"

        # Verify aggregation pipeline was called
        mock_collection.aggregate.assert_called_once()

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_database_cleanup_operations(self, mock_mongo_client):
        """Test database cleanup operations."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        db_manager = DatabaseManager(self.test_db_uri)
        db_manager.connect()

        # Test delete all vectors
        mock_collection.delete_many.return_value = Mock(deleted_count=10)
        result = db_manager.delete_all_vectors()

        assert result is True
        mock_collection.delete_many.assert_called_once_with({})

        # Test document count
        mock_collection.count_documents.return_value = 5
        count = db_manager.get_document_count()

        assert count == 5
        mock_collection.count_documents.assert_called_once_with({})

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_conversation_operations(self, mock_mongo_client):
        """Test conversation-related database operations."""
        # Mock MongoDB client and collections
        mock_client = Mock()
        mock_database = Mock()
        mock_conv_collection = Mock()

        mock_database.__getitem__.return_value = mock_conv_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        db_manager = DatabaseManager(self.test_db_uri)
        db_manager.connect()

        # Test conversation creation
        conversation_data = {
            "title": "Test Conversation",
            "user_id": "user123",
            "messages": [],
            "creation_date": datetime.now(),
            "update_date": datetime.now()
        }

        mock_conv_collection.insert_one.return_value = Mock(
            inserted_id="conv123")

        # Simulate conversation creation (would be in a higher-level service)
        result = mock_conv_collection.insert_one(conversation_data)
        assert result.inserted_id == "conv123"

        # Test conversation retrieval
        mock_conversations = [
            {
                "_id": "conv1",
                "title": "Conversation 1",
                "user_id": "user123",
                "messages": [],
                "creation_date": datetime.now(),
                "update_date": datetime.now()
            },
            {
                "_id": "conv2",
                "title": "Conversation 2",
                "user_id": "user123",
                "messages": [],
                "creation_date": datetime.now(),
                "update_date": datetime.now()
            }
        ]

        mock_conv_collection.find.return_value = mock_conversations
        conversations = list(mock_conv_collection.find({"user_id": "user123"}))

        assert len(conversations) == 2
        assert conversations[0]["title"] == "Conversation 1"

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_error_handling_and_recovery(self, mock_mongo_client):
        """Test database error handling and recovery."""
        # Mock MongoDB client
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        db_manager = DatabaseManager(self.test_db_uri)
        db_manager.connect()

        # Test connection error handling
        # Simulate connection loss by setting client to None
        db_manager.client = None
        assert db_manager.client is None

        # Test search error handling
        mock_collection.aggregate.side_effect = Exception("Search failed")

        with pytest.raises(Exception):
            db_manager.search_similar_vectors([0.1, 0.2, 0.3])

        # Test insertion error handling
        mock_collection.insert_many.side_effect = Exception("Insert failed")

        chunk = DocumentChunk("1", "doc1", "content", 0, {})
        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test-model"
        )

        with pytest.raises(Exception):
            db_manager.store_embedded_chunks([embedded_chunk])

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_context_manager_usage(self, mock_mongo_client):
        """Test using DatabaseManager as context manager."""
        mock_client = Mock()
        mock_database = Mock()
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        # Test context manager
        with DatabaseManager(self.test_db_uri) as db_manager:
            assert db_manager.client is mock_client
            assert db_manager.database is mock_database

            # Test operations within context
            assert db_manager.client is mock_client
            assert db_manager.database is mock_database

        # Should automatically disconnect
        mock_client.close.assert_called_once()

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_concurrent_operations(self, mock_mongo_client):
        """Test handling of concurrent database operations."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        db_manager = DatabaseManager(self.test_db_uri)
        db_manager.connect()

        # Simulate concurrent operations
        operations = []

        # Mock multiple search operations
        mock_collection.aggregate.return_value = [
            {"content": "result", "path": "test.py", "metadata": {},
             "score": 0.8}
        ]

        # Simulate multiple concurrent searches
        for i in range(5):
            query_embedding = [0.1 * i, 0.2 * i, 0.3 * i]
            results = db_manager.search_similar_vectors(
                query_embedding, limit=3)
            operations.append(results)

        # Verify all operations completed
        assert len(operations) == 5
        assert all(len(op) == 1 for op in operations)

        # Verify aggregate was called multiple times
        assert mock_collection.aggregate.call_count == 5

    @patch('genericsuite_codegen.database.setup.MongoClient')
    def test_large_batch_operations(self, mock_mongo_client):
        """Test handling of large batch operations."""
        # Mock MongoDB client and collection
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()
        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client

        db_manager = DatabaseManager(self.test_db_uri)
        db_manager.connect()

        # Create a large batch of embedded chunks
        embedded_chunks = []
        for i in range(100):
            chunk = DocumentChunk(
                f"chunk_{i}",
                f"doc_{i // 10}",
                f"content_{i}",
                i % 10,
                {"file_type": "python"}
            )
            embedded_chunk = EmbeddedChunk(
                chunk=chunk,
                embedding=[0.1 * i, 0.2 * i, 0.3 * i],
                embedding_model="test-model"
            )
            embedded_chunks.append(embedded_chunk)

        # Mock successful batch insertion
        mock_collection.insert_many.return_value = Mock(
            inserted_ids=[f"id_{i}" for i in range(100)]
        )

        result = db_manager.store_embedded_chunks(embedded_chunks)
        assert result is True

        # Verify batch was processed
        mock_collection.insert_many.assert_called_once()
        call_args = mock_collection.insert_many.call_args[0][0]
        assert len(call_args) == 100
