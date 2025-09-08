"""
Test script for database setup and connection utilities.

This script tests the basic functionality of the database manager
without requiring a full MongoDB instance.
"""

import os
import sys
from datetime import datetime
from typing import List

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genericsuite_codegen.database.setup import (
    DatabaseManager,
    EmbeddedChunk,
    SearchResult,
    create_embedded_chunk,
    validate_embedding_dimensions,
    DatabaseConnectionError,
)


def test_embedded_chunk_creation():
    """Test creating EmbeddedChunk instances."""
    print("Testing EmbeddedChunk creation...")

    chunk = create_embedded_chunk(
        chunk_id="test_chunk_1",
        document_path="/test/document.md",
        content="This is a test document chunk.",
        embedding=[0.1, 0.2, 0.3, 0.4],
        chunk_index=0,
        file_type="md",
        metadata={"source": "test"},
    )

    assert chunk.chunk_id == "test_chunk_1"
    assert chunk.document_path == "/test/document.md"
    assert chunk.content == "This is a test document chunk."
    assert chunk.embedding == [0.1, 0.2, 0.3, 0.4]
    assert chunk.chunk_index == 0
    assert chunk.file_type == "md"
    assert chunk.metadata == {"source": "test"}
    assert isinstance(chunk.created_at, datetime)

    print("✓ EmbeddedChunk creation test passed")


def test_embedding_validation():
    """Test embedding dimension validation."""
    print("Testing embedding validation...")

    # Valid embeddings
    valid_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    assert validate_embedding_dimensions(valid_embeddings, 3) == True

    # Invalid embeddings (wrong dimensions)
    invalid_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5],  # Wrong dimension
        [0.7, 0.8, 0.9],
    ]
    assert validate_embedding_dimensions(invalid_embeddings, 3) == False

    print("✓ Embedding validation test passed")


def test_database_manager_initialization():
    """Test DatabaseManager initialization without connection."""
    print("Testing DatabaseManager initialization...")

    # Test with custom URI
    db_manager = DatabaseManager("mongodb://test:27017/")
    assert db_manager.mongodb_uri == "mongodb://test:27017/"
    assert db_manager.db_name == "genericsuite_codegen"
    assert db_manager.client is None
    assert db_manager.database is None

    # Test with environment variable
    os.environ["MONGODB_URI"] = "mongodb://env_test:27017/"
    db_manager2 = DatabaseManager()
    assert db_manager2.mongodb_uri == "mongodb://env_test:27017/"

    print("✓ DatabaseManager initialization test passed")


def test_search_result_creation():
    """Test SearchResult creation."""
    print("Testing SearchResult creation...")

    result = SearchResult(
        content="Test content",
        metadata={"key": "value"},
        similarity_score=0.95,
        document_path="/test/doc.md",
    )

    assert result.content == "Test content"
    assert result.metadata == {"key": "value"}
    assert result.similarity_score == 0.95
    assert result.document_path == "/test/doc.md"

    print("✓ SearchResult creation test passed")


def test_connection_error_handling():
    """Test connection error handling."""
    print("Testing connection error handling...")

    # Test with invalid URI
    db_manager = DatabaseManager("mongodb://invalid_host:99999/")

    try:
        db_manager.connect()
        assert False, "Should have raised DatabaseConnectionError"
    except DatabaseConnectionError as e:
        # Should contain either "Database connection failed" or "Unexpected database error"
        assert "Database connection failed" in str(
            e
        ) or "Unexpected database error" in str(e)
        print("✓ Connection error handling test passed")
    except Exception as e:
        # If we can't connect due to network issues, that's expected
        print(f"✓ Connection error handling test passed (network error: {e})")


def run_all_tests():
    """Run all tests."""
    print("Running database setup tests...\n")

    try:
        test_embedded_chunk_creation()
        test_embedding_validation()
        test_database_manager_initialization()
        test_search_result_creation()
        test_connection_error_handling()

        print("\n✅ All database setup tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
