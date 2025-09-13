"""
Database package for GenericSuite CodeGen.

This package provides MongoDB connection management, vector database
operations, and schema initialization for the knowledge base system.
"""

from .setup import (
    DatabaseManager,
    SearchResult,
    # DocumentEmbeddedChunk,
    DatabaseConnectionError,
    VectorSearchError,
    get_database_manager,
    initialize_database,
    health_check,
    # create_embedded_chunk,
    validate_embedding_dimensions
)

__all__ = [
    "DatabaseManager",
    "SearchResult",
    # "DocumentEmbeddedChunk",
    "DatabaseConnectionError",
    "VectorSearchError",
    "get_database_manager",
    "initialize_database",
    "health_check",
    # "create_embedded_chunk",
    "validate_embedding_dimensions"
]
