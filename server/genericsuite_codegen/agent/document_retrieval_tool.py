"""
Document retrieval tool for the GenericSuite CodeGen AI agent.

This module provides tools for retrieving complete documents from local
storage, enabling the agent to access full GenericSuite knowledge base
articles for accurate code generation.
"""

from os import getenv
from typing import List
from datetime import datetime
from pathlib import Path

from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_warning,
    log_error,
)

from .enhanced_search_types import (
    DocumentContent,
    DocumentMetadata,
    DocumentRetrievalError
)
from .enhanced_search_logging import (
    get_enhanced_search_logger,
    performance_tracking,
    log_performance,
    handle_enhanced_search_errors
)

DEBUG = False

DOCUMENT_RETRIEVAL_MAX_FILE_SIZE_MB = int(getenv(
    "DOCUMENT_RETRIEVAL_MAX_FILE_SIZE_MB", "10"))   # 10MB default


class DocumentRetrievalTool:
    """
    Agent tool for retrieving complete documents from local storage.

    Provides secure access to documents in the local_repo_files directory
    with proper error handling and path validation to prevent security
    issues.
    """

    def __init__(self, local_repo_path: str = None):
        """
        Initialize the document retrieval tool.

        Args:
            local_repo_path: Path to the local repository files directory.
        """
        self.local_repo_path = local_repo_path or getenv(
            "LOCAL_REPO_DIR", "./local_repo_files")
        _ = DEBUG and log_debug(
            f"Initialized DocumentRetrievalTool with local repo path: "
            f"{self.local_repo_path}")
        self.base_path = Path(self.local_repo_path).resolve()
        _ = DEBUG and log_debug(
            f"Initialized DocumentRetrievalTool with base path: "
            f"{self.base_path}")

        # Ensure the base path exists
        if not self.base_path.exists():
            log_warning(
                f"Local repository path does not exist: {self.base_path}")
        elif not self.base_path.is_dir():
            raise DocumentRetrievalError(
                f"Local repository path is not a directory: "
                f"{self.base_path}"
            )

    @log_performance("document_retrieval")
    @handle_enhanced_search_errors(fallback_enabled=False)
    def retrieve_document(self, document_path: str) -> DocumentContent:
        """
        Retrieve complete document content from local storage.

        Args:
            document_path: Relative path to the document within
                local_repo_files.

        Returns:
            DocumentContent: Complete document content with metadata.

        Raises:
            DocumentRetrievalError: If document retrieval fails.
        """
        with performance_tracking(
            "document_retrieval_detailed",
            document_path=document_path,
            base_path=str(self.base_path)
        ):
            try:
                # Validate input
                if not document_path:
                    raise DocumentRetrievalError(
                        "Document path cannot be empty",
                        document_path=document_path,
                        operation="retrieve_document",
                        error_code="EMPTY_PATH"
                    )

                # Validate and resolve the document path
                validated_path = self._validate_document_path(document_path)

                # Check if file exists and is readable
                if not validated_path.exists():
                    raise DocumentRetrievalError(
                        f"Document not found: {document_path}",
                        document_path=document_path,
                        operation="retrieve_document",
                        error_code="FILE_NOT_FOUND"
                    )

                if not validated_path.is_file():
                    raise DocumentRetrievalError(
                        f"Path is not a file: {document_path}",
                        document_path=document_path,
                        operation="retrieve_document",
                        error_code="NOT_A_FILE"
                    )

                # Get file metadata
                stat_info = validated_path.stat()
                file_size = stat_info.st_size
                last_modified = datetime.fromtimestamp(stat_info.st_mtime)

                # Check file size limits (10MB default)
                max_size = DOCUMENT_RETRIEVAL_MAX_FILE_SIZE_MB * 1024 * 1024
                if file_size > max_size:
                    raise DocumentRetrievalError(
                        f"File too large: {file_size} bytes (max: {max_size})",
                        document_path=document_path,
                        operation="retrieve_document",
                        error_code="FILE_TOO_LARGE",
                        details={"file_size": file_size, "max_size": max_size}
                    )

                # Determine file type
                file_type = self._get_file_type(validated_path)

                # Check if file is binary
                is_binary = self._is_binary_file(validated_path)

                if is_binary:
                    raise DocumentRetrievalError(
                        f"Cannot retrieve binary file: {document_path}",
                        document_path=document_path,
                        operation="retrieve_document",
                        error_code="BINARY_FILE"
                    )

                # Read file content with appropriate encoding
                encoding = self._detect_encoding(validated_path)

                try:
                    with open(validated_path, 'r', encoding=encoding) as file:
                        content = file.read()
                except UnicodeDecodeError as e:
                    log_warning(
                        f"Unicode decode error for {document_path}, "
                        f"trying utf-8 with errors='replace': {e}")
                    try:
                        with open(validated_path, 'r', encoding='utf-8',
                                  errors='replace') as file:
                            content = file.read()
                        encoding = 'utf-8'
                    except Exception as fallback_error:
                        raise DocumentRetrievalError(
                            "Failed to read file with fallback encoding:"
                            f" {fallback_error}",
                            document_path=document_path,
                            operation="retrieve_document",
                            error_code="ENCODING_ERROR",
                            original_exception=fallback_error
                        )

                # Create metadata dictionary
                metadata = {
                    'absolute_path': str(validated_path),
                    'relative_path': document_path,
                    'file_extension': validated_path.suffix,
                    'parent_directory': str(
                        validated_path.parent.relative_to(self.base_path)),
                    'encoding_used': encoding,
                    'retrieval_timestamp': datetime.now().isoformat()
                }

                # Create and return DocumentContent
                document_content = DocumentContent(
                    path=document_path,
                    content=content,
                    file_type=file_type,
                    size=file_size,
                    last_modified=last_modified,
                    metadata=metadata,
                    encoding=encoding,
                    is_binary=is_binary
                )

                _ = DEBUG and log_debug(
                    f"Successfully retrieved document: {document_path} "
                    f"({file_size} bytes, {encoding} encoding)")
                return document_content

            except DocumentRetrievalError:
                # Re-raise DocumentRetrievalError as-is
                raise
            except Exception as e:
                log_error(
                    "Unexpected error retrieving document "
                    f"{document_path}: {e}")
                raise DocumentRetrievalError(
                    f"Failed to retrieve document {document_path}: {str(e)}",
                    document_path=document_path,
                    operation="retrieve_document",
                    error_code="RETRIEVAL_ERROR",
                    original_exception=e
                )

    @log_performance("batch_document_retrieval")
    @handle_enhanced_search_errors(fallback_enabled=True, fallback_value=[])
    def retrieve_multiple_documents(
            self, document_paths: List[str]) -> List[DocumentContent]:
        """
        Retrieve multiple documents from local storage.

        Args:
            document_paths: List of relative paths to documents.

        Returns:
            List[DocumentContent]: List of retrieved documents.

        Note:
            This method continues processing even if some documents fail to
            retrieve. Failed retrievals are logged but don't stop the process.
        """
        search_logger = get_enhanced_search_logger()
        retrieved_documents = []
        failed_retrievals = 0

        with performance_tracking(
            "batch_document_retrieval_detailed",
            total_documents=len(document_paths)
        ) as metrics:
            try:
                if not document_paths:
                    log_warning(
                        "No document paths provided for batch retrieval")
                    return []

                # Limit batch size to prevent resource exhaustion
                max_batch_size = 50
                if len(document_paths) > max_batch_size:
                    log_warning(
                        f"Large batch size requested: {len(document_paths)}, "
                        f"limiting to {max_batch_size}"
                    )
                    document_paths = document_paths[:max_batch_size]

                _ = DEBUG and log_debug(
                    f"Retrieving {len(document_paths)} documents")

                for i, document_path in enumerate(document_paths):
                    try:
                        document = self.retrieve_document(document_path)
                        retrieved_documents.append(document)

                        # Log progress for large batches
                        if len(document_paths) > 10 and (i + 1) % 10 == 0:
                            _ = DEBUG and log_debug(
                                f"Retrieved {i + 1}/{len(document_paths)}"
                                " documents")

                    except DocumentRetrievalError as e:
                        failed_retrievals += 1
                        log_warning(
                            f"Failed to retrieve document "
                            f"{document_path}: {e}")
                        # Continue with other documents
                        continue
                    except Exception as e:
                        failed_retrievals += 1
                        log_error(
                            f"Unexpected error retrieving document "
                            f"{document_path}: {e}")
                        # Continue with other documents
                        continue

                # Log batch retrieval operation
                search_logger.log_document_retrieval(
                    document_paths=document_paths,
                    successful_retrievals=len(retrieved_documents),
                    failed_retrievals=failed_retrievals,
                    duration=metrics.duration or 0.0,
                    success=True
                )

                _ = DEBUG and log_debug(
                    f"Batch retrieval completed: "
                    f"{len(retrieved_documents)} successful, "
                    f"{failed_retrievals} failed out of"
                    f" {len(document_paths)} total")

                return retrieved_documents

            except Exception as e:
                # Log failed batch operation
                search_logger.log_document_retrieval(
                    document_paths=document_paths,
                    successful_retrievals=len(retrieved_documents),
                    failed_retrievals=failed_retrievals,
                    duration=metrics.duration or 0.0,
                    success=False,
                    error_details={"message": str(e)}
                )

                log_error(f"Batch document retrieval failed: {e}")
                raise DocumentRetrievalError(
                    f"Batch document retrieval failed: {e}",
                    operation="retrieve_multiple_documents",
                    error_code="BATCH_RETRIEVAL_ERROR",
                    details={
                        "total_requested": len(document_paths),
                        "successful_before_failure": len(retrieved_documents),
                        "failed_before_failure": failed_retrievals
                    },
                    original_exception=e
                )

    def get_document_metadata(self, document_path: str) -> DocumentMetadata:
        """
        Get metadata for a document without retrieving full content.

        Args:
            document_path: Relative path to the document.

        Returns:
            DocumentMetadata: Document metadata information.
        """
        try:
            # Validate and resolve the document path
            validated_path = self._validate_document_path(document_path)

            # Check if file exists
            if not validated_path.exists():
                return DocumentMetadata(
                    path=document_path,
                    file_type="unknown",
                    size=0,
                    last_modified=datetime.min,
                    exists=False,
                    is_readable=False,
                    error_message="File not found"
                )

            if not validated_path.is_file():
                return DocumentMetadata(
                    path=document_path,
                    file_type="unknown",
                    size=0,
                    last_modified=datetime.min,
                    exists=True,
                    is_readable=False,
                    error_message="Path is not a file"
                )

            # Get file metadata
            stat_info = validated_path.stat()
            file_size = stat_info.st_size
            last_modified = datetime.fromtimestamp(stat_info.st_mtime)
            file_type = self._get_file_type(validated_path)
            is_binary = self._is_binary_file(validated_path)
            encoding = (None if is_binary
                        else self._detect_encoding(validated_path))

            return DocumentMetadata(
                path=document_path,
                file_type=file_type,
                size=file_size,
                last_modified=last_modified,
                exists=True,
                is_readable=not is_binary,
                encoding=encoding,
                is_binary=is_binary
            )

        except Exception as e:
            log_error(f"Error getting metadata for {document_path}: {e}")
            return DocumentMetadata(
                path=document_path,
                file_type="unknown",
                size=0,
                last_modified=datetime.min,
                exists=False,
                is_readable=False,
                error_message=str(e)
            )

    def _validate_document_path(self, document_path: str) -> Path:
        """
        Validate document path to prevent directory traversal attacks.

        Args:
            document_path: Relative path to validate.

        Returns:
            Path: Validated absolute path.

        Raises:
            DocumentRetrievalError: If path is invalid or unsafe.
        """
        if not document_path:
            raise DocumentRetrievalError(
                "Document path cannot be empty",
                error_code="INVALID_PATH"
            )

        # Remove leading slashes and normalize path
        clean_path = document_path.lstrip('/')

        # Create path relative to base directory
        try:
            full_path = (self.base_path / clean_path).resolve()
        except Exception:
            raise DocumentRetrievalError(
                f"Invalid path format: {document_path}",
                error_code="INVALID_PATH"
            )

        # Ensure the resolved path is within the base directory
        try:
            full_path.relative_to(self.base_path)
        except ValueError:
            raise DocumentRetrievalError(
                f"Path outside allowed directory: {document_path}",
                error_code="PATH_TRAVERSAL"
            )

        return full_path

    def _get_file_type(self, file_path: Path) -> str:
        """
        Determine file type from file extension.

        Args:
            file_path: Path to the file.

        Returns:
            str: File type/extension.
        """
        suffix = file_path.suffix.lower()
        if suffix:
            return suffix[1:]  # Remove the dot
        return "unknown"

    def _is_binary_file(self, file_path: Path) -> bool:
        """
        Check if a file is binary.

        Args:
            file_path: Path to the file.

        Returns:
            bool: True if file is binary, False otherwise.
        """
        # Common binary file extensions
        binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
            '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac',
            '.sqlite', '.db', '.pyc', '.pyo'
        }

        if file_path.suffix.lower() in binary_extensions:
            return True

        # For unknown extensions, try to read a small portion to detect
        # binary content
        try:
            with open(file_path, 'rb') as file:
                chunk = file.read(1024)  # Read first 1KB

            # Check for null bytes (common in binary files)
            if b'\x00' in chunk:
                return True

            # Check for high ratio of non-printable characters
            printable_chars = sum(
                1 for byte in chunk
                if 32 <= byte <= 126 or byte in [9, 10, 13])
            if len(chunk) > 0 and printable_chars / len(chunk) < 0.7:
                return True

        except Exception:
            # If we can't read the file, assume it might be binary
            return True

        return False

    def _detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding.

        Args:
            file_path: Path to the file.

        Returns:
            str: Detected encoding, defaults to 'utf-8'.
        """
        # Try common encodings in order of preference
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    # Try to read a small portion
                    file.read(1024)
                return encoding
            except UnicodeDecodeError:
                continue
            except Exception:
                break

        # Default to utf-8 if detection fails
        return 'utf-8'
