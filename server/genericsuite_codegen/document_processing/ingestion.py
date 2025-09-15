"""
Document ingestion orchestrator.

This module provides the main orchestrator for the complete document processing
workflow from repository cloning to vector storage, with progress tracking
and error handling.
"""

from typing import List, Dict, Any, Optional, Callable
import os
from pathlib import Path
from datetime import datetime
import logging
import json
import shutil

try:
    import git
except ImportError:
    git = None

from .processors import DocumentProcessorManager, Document
from .chunker import DocumentChunker, DocumentChunk, chunk_document
from .embeddings import EmbeddingGenerator, EmbeddedChunk
from .enums import IngestionStatus
from .types import (
    IngestionProgress,
    IngestionRepositoryInfo,
    IngestionStatistics,
    IngestionResult,
    IngestionRepositoryInfoLastCommit,
)

DEBUG = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if DEBUG else logging.WARNING)

TEMP_DIR = os.getenv("SERVER_TEMP_DIR", "/tmp")
PROGRESS_FILE = TEMP_DIR + "/ingestion_progress.json"


class RepositoryCloner:
    """Handles repository cloning operations."""

    def __init__(self, local_dir: str):
        """Initialize repository cloner."""
        self.local_dir = Path(local_dir)

        if git is None:
            raise ImportError(
                "GitPython not installed. Install with: pip install GitPython")

    def clone_repository(self, repo_url: str, force_refresh: bool = True
                         ) -> bool:
        """
        Clone or update a repository.

        Args:
            repo_url: URL of the repository to clone
            force_refresh: If True, delete existing directory and re-clone

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create parent directory if it doesn't exist
            self.local_dir.parent.mkdir(parents=True, exist_ok=True)

            # Get the repo name from the repo_url
            repo_name = repo_url.split("/")[-1].split(".")[0]

            # Set the local directory to the parent directory + the repo name
            self.local_dir = self.local_dir / repo_name

            # Handle existing directory
            if self.local_dir.exists():
                if force_refresh:
                    logger.info(
                        f"Removing existing directory: {self.local_dir}")
                    shutil.rmtree(self.local_dir)
                else:
                    # Try to update existing repository
                    try:
                        repo = git.Repo(self.local_dir)
                        logger.info(
                            f"Updating existing repository: {self.local_dir}")
                        repo.remotes.origin.pull()
                        return True
                    except Exception as e:
                        logger.warning(
                            f"Could not update existing repository: {e}")
                        logger.info("Removing and re-cloning...")
                        shutil.rmtree(self.local_dir)

            # Clone repository
            logger.info(f"Cloning repository {repo_url} to {self.local_dir}")
            git.Repo.clone_from(repo_url, self.local_dir)
            logger.info("Repository cloned successfully")
            return True

        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False

    def get_repository_info(self) -> IngestionRepositoryInfo:
        """Get information about the cloned repository."""
        if not self.local_dir.exists():
            return {}

        try:
            repo = git.Repo(self.local_dir)

            return IngestionRepositoryInfo(
                path=str(self.local_dir),
                remote_url=repo.remotes.origin.url,
                current_branch=repo.active_branch.name,
                last_commit=IngestionRepositoryInfoLastCommit(
                    hash=repo.head.commit.hexsha,
                    message=repo.head.commit.message.strip(),
                    author=str(repo.head.commit.author),
                    date=datetime.fromtimestamp(
                        repo.head.commit.committed_date)
                ),
                is_dirty=repo.is_dirty()
            )
        except Exception as e:
            logger.warning(f"Could not get repository info: {e}")
            return {'path': str(self.local_dir)}


class DocumentIngestionOrchestrator:
    """Main orchestrator for document ingestion workflow."""

    def __init__(self,
                 repo_url: str,
                 local_dir: str,
                 database_manager,
                 embedding_provider: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 chunking_strategy: str = "adaptive",
                 progress_callback: Optional[Callable[[IngestionProgress],
                                                      None]] = None):
        """
        Initialize document ingestion orchestrator.

        Args:
            repo_url: URL of repository to process
            local_dir: Local directory for cloning
            database_manager: Database manager for storing vectors
            embedding_provider: Embedding provider ('openai' or 'huggingface')
            embedding_model: Specific model to use
            chunking_strategy: Chunking strategy to use
            progress_callback: Optional callback for progress updates
        """
        self.repo_url = repo_url
        self.local_dir = local_dir
        self.database_manager = database_manager
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.chunking_strategy = chunking_strategy
        self.progress_callback = progress_callback

        # Initialize components
        self.cloner = RepositoryCloner(local_dir)
        self.processor_manager = None
        self.chunker = DocumentChunker()
        self.embedding_generator = None

        # Progress tracking
        self.progress = IngestionProgress(
            status=IngestionStatus.NOT_STARTED,
            current_step="Initializing",
            total_steps=6,  # clone, process, chunk, embed, store, complete
            completed_steps=0
        )

        self._save_progress()

    def _remove_progress_file(self):
        """Remove progress file."""
        try:
            os.remove(PROGRESS_FILE)
        except Exception as e:
            logger.error(f"Error removing progress file: {e}")

    def _save_progress(self):
        """Save progress to file."""
        try:
            with open(PROGRESS_FILE, "w") as f:
                json.dump(self.progress.to_dict(), f)
        except Exception as e:
            logger.error(f"Error saving progress to file: {e}")

    def _update_progress(self,
                         status: Optional[IngestionStatus] = None,
                         current_step: Optional[str] = None,
                         increment_completed: bool = False,
                         **kwargs):
        """Update progress and call callback if provided."""
        if status is not None:
            self.progress.status = status

        if current_step is not None:
            self.progress.current_step = current_step

        if increment_completed:
            self.progress.completed_steps += 1

        # Update other fields
        for key, value in kwargs.items():
            if hasattr(self.progress, key):
                setattr(self.progress, key, value)

        # Call progress callback
        if self.progress_callback:
            try:
                self.progress_callback(self.progress)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")

        self._save_progress()

    def cleanup_existing_vectors(self) -> bool:
        """Clean up existing vectors to prevent duplicates."""
        try:
            logger.info("Cleaning up existing vectors...")
            self.database_manager.delete_all_vectors()
            logger.info("Existing vectors cleaned up successfully")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up existing vectors: {e}")
            return False

    def clone_repository(self, force_refresh: bool = True) -> bool:
        """Clone the repository."""
        self._update_progress(
            status=IngestionStatus.CLONING,
            current_step="Cloning repository"
        )

        success = self.cloner.clone_repository(self.repo_url, force_refresh)

        if success:
            self._update_progress(increment_completed=True)
        else:
            self._update_progress(
                status=IngestionStatus.FAILED,
                error_message="Failed to clone repository"
            )

        return success

    def process_files(self) -> List[Document]:
        """Process all files in the repository."""
        self._update_progress(
            status=IngestionStatus.PROCESSING_FILES,
            current_step="Processing files"
        )

        try:
            # Initialize processor manager
            self.processor_manager = DocumentProcessorManager(self.local_dir)

            # Get file statistics
            stats = self.processor_manager.get_file_stats()
            self._update_progress(total_files=stats['total_files'])

            # Process files
            documents = []
            files_to_process = self.processor_manager.get_all_files()

            for i, file_path in enumerate(files_to_process):
                self._update_progress(
                    current_file=str(file_path),
                    processed_files=i
                )

                try:
                    document = self.processor_manager.process_file(file_path)
                    if document:
                        documents.append(document)
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}")

            self._update_progress(
                processed_files=len(files_to_process),
                increment_completed=True
            )

            logger.info(
                f"Processed {len(documents)} documents from"
                f" {len(files_to_process)} files")
            return documents

        except Exception as e:
            logger.error(f"Error processing files: {e}")
            self._update_progress(
                status=IngestionStatus.FAILED,
                error_message=f"Error processing files: {e}"
            )
            return []

    def chunk_documents(self, documents: List[Document]
                        ) -> List[DocumentChunk]:
        """Chunk all documents."""
        self._update_progress(
            status=IngestionStatus.CHUNKING,
            current_step="Chunking documents"
        )

        try:
            all_chunks = []

            for i, document in enumerate(documents):
                self._update_progress(
                    current_file=document.path,
                    processed_files=i,
                    total_files=len(documents)
                )

                try:
                    chunks = chunk_document(
                        document, strategy=self.chunking_strategy)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(
                        f"Error chunking document {document.path}: {e}")

            self._update_progress(
                total_chunks=len(all_chunks),
                processed_chunks=len(all_chunks),
                increment_completed=True
            )

            logger.info(
                f"Created {len(all_chunks)} chunks from"
                f" {len(documents)} documents")
            return all_chunks

        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            self._update_progress(
                status=IngestionStatus.FAILED,
                error_message=f"Error chunking documents: {e}"
            )
            return []

    def generate_embeddings(self, chunks: List[DocumentChunk]
                            ) -> List[EmbeddedChunk]:
        """Generate embeddings for all chunks."""
        self._update_progress(
            status=IngestionStatus.GENERATING_EMBEDDINGS,
            current_step="Generating embeddings"
        )

        try:
            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator(
                provider=self.embedding_provider,
                model=self.embedding_model
            )

            # Generate embeddings
            embedded_chunks = self.embedding_generator\
                .generate_embeddings_for_chunks(chunks)

            self._update_progress(
                processed_chunks=len(embedded_chunks),
                increment_completed=True
            )

            logger.info(
                f"Generated embeddings for {len(embedded_chunks)} chunks")
            return embedded_chunks

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            self._update_progress(
                status=IngestionStatus.FAILED,
                error_message=f"Error generating embeddings: {e}"
            )
            return []

    def store_vectors(self, embedded_chunks: List[EmbeddedChunk]) -> bool:
        """Store vectors in the database."""
        self._update_progress(
            status=IngestionStatus.STORING_VECTORS,
            current_step="Storing vectors in database"
        )

        try:
            success = self.database_manager.store_embeddings(embedded_chunks)

            if success:
                self._update_progress(increment_completed=True)
                logger.info(
                    f"Stored {len(embedded_chunks)} vectors in database")
            else:
                self._update_progress(
                    status=IngestionStatus.FAILED,
                    error_message="Failed to store vectors in database"
                )

            return success

        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            self._update_progress(
                status=IngestionStatus.FAILED,
                error_message=f"Error storing vectors: {e}"
            )
            return False

    def run_full_ingestion(self, force_refresh: bool = True) -> Dict[str, Any]:
        """
        Run the complete ingestion workflow.

        Args:
            force_refresh: If True, clean up existing data and re-process
            everything

        Returns:
            Dictionary with ingestion results and statistics
        """
        self.progress.started_at = datetime.now()

        try:
            # Step 1: Clean up existing vectors if force refresh
            if force_refresh:
                if not self.cleanup_existing_vectors():
                    return self._get_failure_result(
                        "Failed to clean up existing vectors")

            # Step 2: Clone repository
            if not self.clone_repository(force_refresh):
                return self._get_failure_result("Failed to clone repository")

            # Step 3: Process files
            documents = self.process_files()
            if not documents:
                return self._get_failure_result("No documents processed")

            # Step 4: Chunk documents
            chunks = self.chunk_documents(documents)
            if not chunks:
                return self._get_failure_result("No chunks created")

            # Step 5: Generate embeddings
            embedded_chunks = self.generate_embeddings(chunks)
            if not embedded_chunks:
                return self._get_failure_result("No embeddings generated")

            # Step 6: Store vectors
            if not self.store_vectors(embedded_chunks):
                return self._get_failure_result("Failed to store vectors")

            # Complete
            self.progress.completed_at = datetime.now()
            self._update_progress(
                status=IngestionStatus.COMPLETED,
                current_step="Ingestion completed successfully",
                increment_completed=True
            )

            self._save_progress()

            # Return success result
            return self._get_success_result(documents, chunks, embedded_chunks)

        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {e}")
            self.progress.completed_at = datetime.now()
            self._update_progress(
                status=IngestionStatus.FAILED,
                error_message=f"Unexpected error: {e}"
            )
            return self._get_failure_result(f"Unexpected error: {e}")

    def _get_success_result(self,
                            documents: List[Document],
                            chunks: List[DocumentChunk],
                            embedded_chunks: List[EmbeddedChunk]
                            ) -> Dict[str, Any]:
        """Create success result dictionary."""
        duration = None
        if self.progress.started_at and self.progress.completed_at:
            duration = (self.progress.completed_at -
                        self.progress.started_at).total_seconds()

        return IngestionResult(
            success=True,
            status=self.progress.status.value,
            statistics=IngestionStatistics(
                total_documents=len(documents),
                total_chunks=len(chunks),
                total_embeddings=len(embedded_chunks),
                duration_seconds=duration,
                embedding_model=self.embedding_generator.get_model_info()
                if self.embedding_generator else None,
                repository_info=self.cloner.get_repository_info()
            ),
            progress=self.progress
        )

    def _get_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create failure result dictionary."""
        duration = None
        if self.progress.started_at and self.progress.completed_at:
            duration = (self.progress.completed_at -
                        self.progress.started_at).total_seconds()

        return IngestionResult(
            success=False,
            status=self.progress.status.value,
            error=error_message,
            statistics=IngestionStatistics(
                duration_seconds=duration
            ),
            progress=self.progress
        )

    def get_progress(self) -> IngestionProgress:
        """Get current progress."""
        # return self.progress
        return load_progress_from_file()


def load_progress_from_file():
    """Load progress from file."""
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r") as f:
                return IngestionProgress(**json.load(f))
        else:
            return IngestionProgress(
                status=IngestionStatus.NOT_STARTED,
                current_step="Idle",
                total_steps=0,  # idle
                completed_steps=0
            )
    except Exception as e:
        logger.error(f"Error loading progress from file: {e}")
        return None


# Convenience functions
def create_ingestion_orchestrator(repo_url: str,
                                  local_dir: str,
                                  database_manager,
                                  **kwargs) -> DocumentIngestionOrchestrator:
    """Create a document ingestion orchestrator."""
    return DocumentIngestionOrchestrator(
        repo_url=repo_url,
        local_dir=local_dir,
        database_manager=database_manager,
        **kwargs
    )


def run_ingestion(repo_url: str,
                  local_dir: str,
                  database_manager,
                  force_refresh: bool = True,
                  **kwargs) -> Dict[str, Any]:
    """Run complete document ingestion workflow."""
    orchestrator = create_ingestion_orchestrator(
        repo_url=repo_url,
        local_dir=local_dir,
        database_manager=database_manager,
        **kwargs
    )

    return orchestrator.run_full_ingestion(force_refresh=force_refresh)


def get_ingestion_progress() -> IngestionProgress:
    """Get ingestion progress."""
    return load_progress_from_file()
