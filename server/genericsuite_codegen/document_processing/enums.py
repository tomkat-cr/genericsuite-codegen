from enum import Enum


class IngestionStatus(Enum):
    """Status of ingestion process."""
    SCHEDULED = "scheduled"
    NOT_STARTED = "not_started"
    CLONING = "cloning"
    PROCESSING_FILES = "processing_files"
    CHUNKING = "chunking"
    GENERATING_EMBEDDINGS = "generating_embeddings"
    STORING_VECTORS = "storing_vectors"
    COPYING_FILES = "copying_files"
    COMPLETED = "completed"
    FAILED = "failed"
