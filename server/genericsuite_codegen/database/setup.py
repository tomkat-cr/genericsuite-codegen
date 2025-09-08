"""
Database setup and connection utilities for GenericSuite CodeGen.

This module provides MongoDB connection management, vector database operations,
and schema initialization for the knowledge base, conversations, and users collections.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio
from contextlib import asynccontextmanager

import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    DuplicateKeyError,
    PyMongoError,
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector similarity search."""

    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    document_path: str


@dataclass
class EmbeddedChunk:
    """Document chunk with embedding vector."""

    chunk_id: str
    document_path: str
    content: str
    embedding: List[float]
    chunk_index: int
    file_type: str
    metadata: Dict[str, Any]
    created_at: datetime


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""

    pass


class VectorSearchError(Exception):
    """Raised when vector search operations fail."""

    pass


class DatabaseManager:
    """
    MongoDB database manager with vector search capabilities.

    Handles connection management, schema initialization, and vector operations
    for the GenericSuite CodeGen knowledge base.
    """

    def __init__(self, mongodb_uri: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            mongodb_uri: MongoDB connection URI. If None, reads from environment.
        """
        self.mongodb_uri = mongodb_uri or os.getenv(
            "MONGODB_URI", "mongodb://localhost:27017/"
        )
        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        self.db_name = "genericsuite_codegen"

        # Collection names
        self.knowledge_base_collection = "knowledge_base"
        self.conversations_collection = "ai_chatbot_conversations"
        self.users_collection = "users"

        # Connection pool settings
        self.max_pool_size = int(os.getenv("MONGODB_MAX_POOL_SIZE", "10"))
        self.min_pool_size = int(os.getenv("MONGODB_MIN_POOL_SIZE", "1"))
        self.max_idle_time_ms = int(os.getenv("MONGODB_MAX_IDLE_TIME_MS", "30000"))
        self.server_selection_timeout_ms = int(
            os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000")
        )

    def connect(self) -> None:
        """
        Establish connection to MongoDB with connection pooling and error handling.

        Raises:
            DatabaseConnectionError: If connection fails after retries.
        """
        try:
            logger.info(f"Connecting to MongoDB at {self.mongodb_uri}")

            self.client = MongoClient(
                self.mongodb_uri,
                maxPoolSize=self.max_pool_size,
                minPoolSize=self.min_pool_size,
                maxIdleTimeMS=self.max_idle_time_ms,
                serverSelectionTimeoutMS=self.server_selection_timeout_ms,
                retryWrites=True,
                retryReads=True,
            )

            # Test connection
            self.client.admin.command("ping")
            self.database = self.client[self.db_name]

            logger.info("Successfully connected to MongoDB")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise DatabaseConnectionError(f"Database connection failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise DatabaseConnectionError(f"Unexpected database error: {e}")

    def disconnect(self) -> None:
        """Close database connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            logger.info("Disconnected from MongoDB")

    @asynccontextmanager
    async def get_connection(self):
        """
        Async context manager for database connections.

        Yields:
            Database: MongoDB database instance.
        """
        if self.database is None:
            self.connect()

        try:
            yield self.database
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            raise

    def initialize_schema(self) -> None:
        """
        Initialize database schema and create indexes for all collections.

        Creates collections and indexes for:
        - knowledge_base: Vector search and text search indexes
        - ai_chatbot_conversations: User and date indexes
        - users: Email and status indexes
        """
        if self.database is None:
            self.connect()

        try:
            logger.info("Initializing database schema...")

            # Initialize knowledge_base collection
            self._initialize_knowledge_base_collection()

            # Initialize conversations collection
            self._initialize_conversations_collection()

            # Initialize users collection
            self._initialize_users_collection()

            logger.info("Database schema initialization completed successfully")

        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise DatabaseConnectionError(f"Schema initialization error: {e}")

    def _initialize_knowledge_base_collection(self) -> None:
        """Initialize knowledge_base collection with vector search index."""
        collection = self.database[self.knowledge_base_collection]

        # Create indexes for efficient querying
        indexes = [
            ("path", ASCENDING),
            ("file_type", ASCENDING),
            ("chunk_index", ASCENDING),
            ("created_at", DESCENDING),
            ([("path", ASCENDING), ("chunk_index", ASCENDING)], {"unique": True}),
        ]

        for index in indexes:
            if isinstance(index, tuple) and len(index) == 2:
                collection.create_index(index[0], **index[1])
            else:
                collection.create_index(index)

        # Create vector search index for embeddings
        # Note: This requires MongoDB Atlas or MongoDB 6.0+ with vector search enabled
        try:
            vector_index = {
                "name": "vector_index",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": 384,  # Default for gte-small model
                            "similarity": "cosine",
                        }
                    ]
                },
            }

            # Check if vector search is available
            existing_indexes = list(collection.list_search_indexes())
            vector_index_exists = any(
                idx.get("name") == "vector_index" for idx in existing_indexes
            )

            if not vector_index_exists:
                collection.create_search_index(vector_index)
                logger.info("Created vector search index for knowledge_base collection")

        except Exception as e:
            logger.warning(
                f"Vector search index creation failed (may not be supported): {e}"
            )
            # Continue without vector search index - will use alternative search methods

        logger.info("Initialized knowledge_base collection")

    def _initialize_conversations_collection(self) -> None:
        """Initialize ai_chatbot_conversations collection."""
        collection = self.database[self.conversations_collection]

        # Create indexes
        indexes = [
            ("user_id", ASCENDING),
            ("creation_date", DESCENDING),
            ("update_date", DESCENDING),
            ([("user_id", ASCENDING), ("creation_date", DESCENDING)]),
        ]

        for index in indexes:
            if isinstance(index, tuple) and len(index) == 2:
                collection.create_index(index[0], **index[1])
            else:
                collection.create_index(index)

        logger.info("Initialized ai_chatbot_conversations collection")

    def _initialize_users_collection(self) -> None:
        """Initialize users collection."""
        collection = self.database[self.users_collection]

        # Create indexes
        indexes = [
            ("email", ASCENDING, {"unique": True}),
            ("status", ASCENDING),
            ("creation_date", DESCENDING),
            ("superuser", ASCENDING),
        ]

        for index in indexes:
            if len(index) == 3:
                collection.create_index(index[0], **index[2])
            else:
                collection.create_index(index)

        logger.info("Initialized users collection")

    # Vector Database Operations

    def store_embeddings(self, embedded_chunks: List[EmbeddedChunk]) -> bool:
        """
        Store document embeddings in the knowledge base collection.

        Args:
            embedded_chunks: List of document chunks with embeddings.

        Returns:
            bool: True if storage successful, False otherwise.

        Raises:
            VectorSearchError: If storage operation fails.
        """
        if self.database is None:
            self.connect()

        try:
            collection = self.database[self.knowledge_base_collection]

            # Prepare documents for insertion
            documents = []
            for chunk in embedded_chunks:
                doc = {
                    "_id": chunk.chunk_id,
                    "path": chunk.document_path,
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "file_type": chunk.file_type,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                    "created_at": chunk.created_at,
                }
                documents.append(doc)

            # Insert documents with error handling for duplicates
            if documents:
                try:
                    result = collection.insert_many(documents, ordered=False)
                    logger.info(
                        f"Successfully stored {len(result.inserted_ids)} embeddings"
                    )
                    return True
                except DuplicateKeyError as e:
                    # Handle duplicate keys by updating existing documents
                    logger.warning(
                        f"Duplicate keys found, updating existing documents: {e}"
                    )
                    return self._update_existing_embeddings(documents)

            return True

        except PyMongoError as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise VectorSearchError(f"Embedding storage failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error storing embeddings: {e}")
            raise VectorSearchError(f"Unexpected storage error: {e}")

    def _update_existing_embeddings(self, documents: List[Dict[str, Any]]) -> bool:
        """Update existing documents with new embeddings."""
        try:
            collection = self.database[self.knowledge_base_collection]

            for doc in documents:
                collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

            logger.info(f"Updated {len(documents)} existing embeddings")
            return True

        except Exception as e:
            logger.error(f"Failed to update existing embeddings: {e}")
            return False

    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        file_type_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_embedding: Query vector for similarity search.
            limit: Maximum number of results to return.
            file_type_filter: Optional filter by file type.

        Returns:
            List[SearchResult]: Similar documents with similarity scores.

        Raises:
            VectorSearchError: If search operation fails.
        """
        if self.database is None:
            self.connect()

        try:
            collection = self.database[self.knowledge_base_collection]

            # Try vector search first (if available)
            try:
                return self._vector_search(
                    collection, query_embedding, limit, file_type_filter
                )
            except Exception as e:
                logger.warning(
                    f"Vector search failed, falling back to cosine similarity: {e}"
                )
                return self._cosine_similarity_search(
                    collection, query_embedding, limit, file_type_filter
                )

        except PyMongoError as e:
            logger.error(f"Search operation failed: {e}")
            raise VectorSearchError(f"Search failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            raise VectorSearchError(f"Unexpected search error: {e}")

    def _vector_search(
        self,
        collection: Collection,
        query_embedding: List[float],
        limit: int,
        file_type_filter: Optional[str],
    ) -> List[SearchResult]:
        """Perform vector search using MongoDB Atlas vector search."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "content": 1,
                    "path": 1,
                    "file_type": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        # Add file type filter if specified
        if file_type_filter:
            pipeline.insert(1, {"$match": {"file_type": file_type_filter}})

        results = []
        for doc in collection.aggregate(pipeline):
            result = SearchResult(
                content=doc["content"],
                metadata=doc.get("metadata", {}),
                similarity_score=doc.get("score", 0.0),
                document_path=doc["path"],
            )
            results.append(result)

        return results

    def _cosine_similarity_search(
        self,
        collection: Collection,
        query_embedding: List[float],
        limit: int,
        file_type_filter: Optional[str],
    ) -> List[SearchResult]:
        """Fallback cosine similarity search using aggregation pipeline."""
        # Build match filter
        match_filter = {}
        if file_type_filter:
            match_filter["file_type"] = file_type_filter

        pipeline = [
            {"$match": match_filter} if match_filter else {"$match": {}},
            {
                "$addFields": {
                    "similarity": {
                        "$let": {
                            "vars": {
                                "dot_product": {
                                    "$reduce": {
                                        "input": {
                                            "$range": [0, {"$size": "$embedding"}]
                                        },
                                        "initialValue": 0,
                                        "in": {
                                            "$add": [
                                                "$$value",
                                                {
                                                    "$multiply": [
                                                        {
                                                            "$arrayElemAt": [
                                                                "$embedding",
                                                                "$$this",
                                                            ]
                                                        },
                                                        {
                                                            "$arrayElemAt": [
                                                                query_embedding,
                                                                "$$this",
                                                            ]
                                                        },
                                                    ]
                                                },
                                            ]
                                        },
                                    }
                                },
                                "query_magnitude": {
                                    "$sqrt": {
                                        "$reduce": {
                                            "input": query_embedding,
                                            "initialValue": 0,
                                            "in": {
                                                "$add": [
                                                    "$$value",
                                                    {"$multiply": ["$$this", "$$this"]},
                                                ]
                                            },
                                        }
                                    }
                                },
                                "doc_magnitude": {
                                    "$sqrt": {
                                        "$reduce": {
                                            "input": "$embedding",
                                            "initialValue": 0,
                                            "in": {
                                                "$add": [
                                                    "$$value",
                                                    {"$multiply": ["$$this", "$$this"]},
                                                ]
                                            },
                                        }
                                    }
                                },
                            },
                            "in": {
                                "$divide": [
                                    "$$dot_product",
                                    {
                                        "$multiply": [
                                            "$$query_magnitude",
                                            "$$doc_magnitude",
                                        ]
                                    },
                                ]
                            },
                        }
                    }
                }
            },
            {"$sort": {"similarity": -1}},
            {"$limit": limit},
            {
                "$project": {
                    "content": 1,
                    "path": 1,
                    "file_type": 1,
                    "metadata": 1,
                    "similarity": 1,
                }
            },
        ]

        # Remove empty match stage
        if not match_filter:
            pipeline = pipeline[1:]

        results = []
        for doc in collection.aggregate(pipeline):
            result = SearchResult(
                content=doc["content"],
                metadata=doc.get("metadata", {}),
                similarity_score=doc.get("similarity", 0.0),
                document_path=doc["path"],
            )
            results.append(result)

        return results

    def delete_all_vectors(self) -> bool:
        """
        Delete all vectors from the knowledge base collection.

        Returns:
            bool: True if deletion successful, False otherwise.
        """
        if self.database is None:
            self.connect()

        try:
            collection = self.database[self.knowledge_base_collection]
            result = collection.delete_many({})

            logger.info(f"Deleted {result.deleted_count} vectors from knowledge base")
            return True

        except PyMongoError as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting vectors: {e}")
            return False

    def delete_vectors_by_path(self, document_path: str) -> bool:
        """
        Delete vectors for a specific document path.

        Args:
            document_path: Path of the document to delete.

        Returns:
            bool: True if deletion successful, False otherwise.
        """
        if self.database is None:
            self.connect()

        try:
            collection = self.database[self.knowledge_base_collection]
            result = collection.delete_many({"path": document_path})

            logger.info(
                f"Deleted {result.deleted_count} vectors for path: {document_path}"
            )
            return True

        except PyMongoError as e:
            logger.error(f"Failed to delete vectors for path {document_path}: {e}")
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error deleting vectors for path {document_path}: {e}"
            )
            return False

    def get_document_count(self) -> int:
        """
        Get total number of documents in the knowledge base.

        Returns:
            int: Number of documents in the knowledge base.
        """
        if self.database is None:
            self.connect()

        try:
            collection = self.database[self.knowledge_base_collection]
            return collection.count_documents({})

        except PyMongoError as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error getting document count: {e}")
            return 0

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base collection.

        Returns:
            Dict[str, Any]: Statistics including document count, file types, etc.
        """
        if self.database is None:
            self.connect()

        try:
            collection = self.database[self.knowledge_base_collection]

            # Get basic stats
            total_docs = collection.count_documents({})

            # Get file type distribution
            file_type_pipeline = [
                {"$group": {"_id": "$file_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ]
            file_types = list(collection.aggregate(file_type_pipeline))

            # Get recent documents
            recent_docs = collection.find().sort("created_at", -1).limit(5)
            recent_paths = [doc["path"] for doc in recent_docs]

            return {
                "total_documents": total_docs,
                "file_types": file_types,
                "recent_documents": recent_paths,
                "collection_size": collection.estimated_document_count(),
            }

        except PyMongoError as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error getting knowledge base stats: {e}")
            return {"error": str(e)}


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get or create the global database manager instance.

    Returns:
        DatabaseManager: Global database manager instance.
    """
    global _db_manager

    if _db_manager is None:
        _db_manager = DatabaseManager()

    return _db_manager


def initialize_database() -> DatabaseManager:
    """
    Initialize database connection and schema.

    Returns:
        DatabaseManager: Initialized database manager.

    Raises:
        DatabaseConnectionError: If initialization fails.
    """
    db_manager = get_database_manager()
    db_manager.connect()
    db_manager.initialize_schema()

    return db_manager


async def health_check() -> Dict[str, Any]:
    """
    Perform database health check.

    Returns:
        Dict[str, Any]: Health check results.
    """
    try:
        db_manager = get_database_manager()

        if db_manager.database is None:
            db_manager.connect()

        # Test connection
        db_manager.client.admin.command("ping")

        # Get basic stats
        stats = db_manager.get_knowledge_base_stats()

        return {
            "status": "healthy",
            "connected": True,
            "database": db_manager.db_name,
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "connected": False, "error": str(e)}


# Utility functions for common operations


def create_embedded_chunk(
    chunk_id: str,
    document_path: str,
    content: str,
    embedding: List[float],
    chunk_index: int,
    file_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> EmbeddedChunk:
    """
    Create an EmbeddedChunk instance with current timestamp.

    Args:
        chunk_id: Unique identifier for the chunk.
        document_path: Path to the source document.
        content: Text content of the chunk.
        embedding: Vector embedding for the chunk.
        chunk_index: Index of the chunk within the document.
        file_type: Type of the source file.
        metadata: Optional metadata dictionary.

    Returns:
        EmbeddedChunk: New embedded chunk instance.
    """
    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_path=document_path,
        content=content,
        embedding=embedding,
        chunk_index=chunk_index,
        file_type=file_type,
        metadata=metadata or {},
        created_at=datetime.utcnow(),
    )


def validate_embedding_dimensions(
    embeddings: List[List[float]], expected_dim: int
) -> bool:
    """
    Validate that all embeddings have the expected dimensions.

    Args:
        embeddings: List of embedding vectors.
        expected_dim: Expected dimension count.

    Returns:
        bool: True if all embeddings have correct dimensions.
    """
    return all(len(emb) == expected_dim for emb in embeddings)


def get_database_connection() -> DatabaseManager:
    """
    Get database connection.

    Returns:
        DatabaseManager: Database manager instance.
    """
    db_manager = get_database_manager()

    if db_manager.database is None:
        db_manager.connect()
    return db_manager


async def test_database_connection(db: DatabaseManager) -> bool:
    """
    Test database connection and return health status.

    Args:
        db_manager: Database manager instance.

    Returns:
        Dict[str, Any]: Health status dictionary.
    """
    try:
        # Test connection
        db.client.admin.command("ping")

        # Get basic stats
        stats = db.get_knowledge_base_stats()

        return True

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage and testing
    import asyncio

    async def main():
        """Test database operations."""
        try:
            # Initialize database
            db_manager = initialize_database()

            # Perform health check
            health = await health_check()
            print(f"Health check: {health}")

            # Get stats
            stats = db_manager.get_knowledge_base_stats()
            print(f"Knowledge base stats: {stats}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if _db_manager:
                _db_manager.disconnect()

    asyncio.run(main())
