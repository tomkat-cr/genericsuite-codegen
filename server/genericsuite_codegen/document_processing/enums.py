from enum import Enum


class IngestionStatus(Enum):
    """Status of ingestion process."""
    NOT_STARTED = "not_started"
    CLONING = "cloning"
    PROCESSING_FILES = "processing_files"
    CHUNKING = "chunking"
    GENERATING_EMBEDDINGS = "generating_embeddings"
    STORING_VECTORS = "storing_vectors"
    COMPLETED = "completed"
    FAILED = "failed"
