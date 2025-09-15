"""
Document processing module for GenericSuite CodeGen.

This module provides comprehensive document processing capabilities including:
- File processing for multiple formats (TXT, PDF, code files)
- Text chunking with configurable strategies
- Embedding generation using OpenAI and HuggingFace models
- Complete document ingestion orchestration

Main components:
- processors: File format processors and filtering
- chunker: Text chunking functionality
- embeddings: Embedding generation system
- ingestion: Complete workflow orchestration
"""

from .processors import (
    Document,
    DocumentProcessorManager,
    create_processor_manager,
    process_repository,
    get_repository_stats
)

from .chunker import (
    DocumentChunk,
    DocumentChunker,
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    CodeChunker,
    AdaptiveChunker,
    chunk_document,
    get_chunking_stats
)

from .embeddings import (
    EmbeddedChunk,
    EmbeddingGenerator,
    OpenAIEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    create_embedding_generator,
    get_available_providers,
    validate_embedding_config,
    generate_embeddings_async
)

from .ingestion import (
    IngestionStatus,
    IngestionProgress,
    RepositoryCloner,
    DocumentIngestionOrchestrator,
    create_ingestion_orchestrator,
    run_ingestion
)

__all__ = [
    # Core data classes
    'Document',
    'DocumentChunk', 
    'EmbeddedChunk',
    'IngestionProgress',
    
    # Main classes
    'DocumentProcessorManager',
    'DocumentChunker',
    'EmbeddingGenerator',
    'DocumentIngestionOrchestrator',
    'RepositoryCloner',
    
    # Chunking strategies
    'FixedSizeChunker',
    'SentenceChunker', 
    'ParagraphChunker',
    'CodeChunker',
    'AdaptiveChunker',
    
    # Embedding providers
    'OpenAIEmbeddingProvider',
    'HuggingFaceEmbeddingProvider',
    
    # Enums
    'IngestionStatus',
    
    # Convenience functions
    'create_processor_manager',
    'process_repository',
    'get_repository_stats',
    'chunk_document',
    'get_chunking_stats',
    'create_embedding_generator',
    'get_available_providers',
    'validate_embedding_config',
    'generate_embeddings_async',
    'create_ingestion_orchestrator',
    'run_ingestion'
]