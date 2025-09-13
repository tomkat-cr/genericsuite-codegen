from typing import Dict, Any, Optional, List
from datetime import datetime

from pydantic import BaseModel, Field
from dataclasses import dataclass

from .enums import IngestionStatus


@dataclass
class IngestionProgress:
    """Progress information for ingestion process."""
    status: IngestionStatus
    current_step: str
    total_steps: int
    completed_steps: int
    current_file: Optional[str] = None
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100

    @property
    def file_progress_percentage(self) -> float:
        """Calculate file processing progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100

    @property
    def chunk_progress_percentage(self) -> float:
        """Calculate chunk processing progress percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'current_file': self.current_file,
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'total_chunks': self.total_chunks,
            'processed_chunks': self.processed_chunks,
            'error_message': self.error_message,
            'started_at': self.started_at,
            'completed_at': self.completed_at
        }


class EmbeddingModel(BaseModel):
    provider: str = Field(description="Embedding provider")
    model: str = Field(description="Embedding model")
    dimension: int = Field(description="Embedding dimension")
    max_tokens: int = Field(description="Max tokens")


class IngestionRepositoryInfoLastCommit(BaseModel):
    hash: str = Field(description="Commit hash")
    message: str = Field(description="Commit message")
    author: str = Field(description="Commit author")
    date: datetime = Field(description="Commit date")


class IngestionRepositoryInfo(BaseModel):
    path: str = Field(description="Repository path")
    remote_url: str = Field(description="Repository remote URL")
    current_branch: str = Field(description="Repository current branch")
    last_commit: IngestionRepositoryInfoLastCommit = Field(
        description="Repository last commit")
    is_dirty: bool = Field(description="Repository is dirty")


class IngestionStatistics(BaseModel):
    total_documents: Optional[int] = Field(
        default=None, description="Total documents")
    total_chunks: Optional[int] = Field(
        default=None, description="Total chunks")
    total_embeddings: Optional[int] = Field(
        default=None, description="Total embeddings")
    duration_seconds: Optional[float] = Field(
        default=None, description="Duration in seconds")
    embedding_model: Optional[EmbeddingModel] = Field(
        default=None, description="Embedding model")
    repository_info: Optional[IngestionRepositoryInfo] = Field(
        default=None, description="Repository info")


class IngestionResult(BaseModel):
    success: bool = Field(description="Success status")
    status: str = Field(description="Status")
    statistics: IngestionStatistics = Field(
        default=IngestionStatistics(), description="Statistics")
    progress: Optional[IngestionProgress] = Field(
        default=None, description="Progress")
    error: Optional[str] = Field(default=None, description="Error message")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


@dataclass
class EmbeddedChunk:
    """Represents a document chunk with its embedding."""
    chunk_id: str
    document_id: str
    content: str
    embedding: List[float]
    embedding_model: str
    chunk_index: int
    metadata: Dict[str, Any]
    created_at: datetime
