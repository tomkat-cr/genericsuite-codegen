"""
Document processors for multiple file formats.

This module provides processors for different file types including text files,
PDF files, and various code files. It also includes file filtering logic
to respect .gitignore rules and file extension rules.
"""

import re
from pathlib import Path
from typing import List, Set, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    import pypdf
except ImportError:
    pypdf = None

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a processed document."""
    id: str
    # >>-->
    path: str
    content: str
    file_type: str
    metadata: Dict[str, Any]
    created_at: datetime


class FileFilter:
    """Handles file filtering based on .gitignore rules and extension rules."""

    # File extensions to include (from BuildPrompt.md)
    INCLUDED_EXTENSIONS = {
        # Configuration Files
        '.babelrc', '.cfg', '.dockerignore', '.env', '.example', '.gitignore',
        '.ini', '.nvmrc', '.npmrc', '.python-version', '.toml', '.yaml',
        '.yml',

        # Code Files
        '.py', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs', '.hbs',

        # Web Assets
        '.html', '.css',

        # Documentation
        '.md', '.txt', '.pdf',

        # Package Management
        '.json',

        # Build/Development
        '.sh', '.sample', '.for_test'
    }

    # File extensions to exclude (from BuildPrompt.md)
    EXCLUDED_EXTENSIONS = {
        # Web Assets
        '.svg', '.png', '.jpg', '.ico',

        # Documentation
        '.pptx',

        # Package Management
        '.lock',

        # Security/Certificates
        '.crt', '.key',

        # Build/Development
        '.bak',

        # System Files
        '.DS_Store',

        # Numbered Extensions
        '.0', '.5', '.10', '.11', '.12', '.13'
    }

    def __init__(self, repo_path: str):
        """Initialize file filter with repository path."""
        self.repo_path = Path(repo_path)
        self.gitignore_patterns = self._load_gitignore_patterns()

    def _load_gitignore_patterns(self) -> List[str]:
        """Load patterns from .gitignore files."""
        patterns = []

        # Load root .gitignore
        gitignore_path = self.repo_path / '.gitignore'
        if gitignore_path.exists():
            patterns.extend(self._parse_gitignore(gitignore_path))

        # Load .gitignore files in subdirectories
        for gitignore_file in self.repo_path.rglob('.gitignore'):
            if gitignore_file != gitignore_path:
                patterns.extend(self._parse_gitignore(gitignore_file))

        return patterns

    def _parse_gitignore(self, gitignore_path: Path) -> List[str]:
        """Parse a .gitignore file and return patterns."""
        patterns = []
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        patterns.append(line)
        except Exception as e:
            logger.warning(
                f"Could not read .gitignore file {gitignore_path}: {e}")

        return patterns

    def _matches_gitignore_pattern(self, file_path: Path, pattern: str
                                   ) -> bool:
        """Check if a file path matches a gitignore pattern."""
        # Convert gitignore pattern to regex
        # This is a simplified implementation
        pattern = pattern.replace('*', '.*')
        pattern = pattern.replace('?', '.')

        # Handle directory patterns
        if pattern.endswith('/'):
            pattern = pattern[:-1]
            return file_path.is_dir() and re.match(pattern,
                                                   str(file_path.name))

        # Handle patterns with path separators
        if '/' in pattern:
            relative_path = file_path.relative_to(self.repo_path)
            return re.match(pattern, str(relative_path))

        # Handle filename patterns
        return re.match(pattern, file_path.name)

    def should_include_file(self, file_path: Path) -> bool:
        """Determine if a file should be included based on filtering rules."""
        # Convert to Path object if string
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            return False

        # Skip directories
        if file_path.is_dir():
            return False

        # Get file extension
        extension = file_path.suffix.lower()

        # Check excluded extensions first
        if extension in self.EXCLUDED_EXTENSIONS:
            return False

        # Check included extensions
        if extension not in self.INCLUDED_EXTENSIONS:
            # Special case for files without extensions
            if not extension and file_path.name in {
                    '.babelrc', '.dockerignore', '.gitignore'}:
                pass  # Include these special files
            else:
                return False

        # Check gitignore patterns
        for pattern in self.gitignore_patterns:
            if self._matches_gitignore_pattern(file_path, pattern):
                return False

        # Skip hidden files and directories (except specific ones)
        parts = file_path.parts
        for part in parts:
            if part.startswith('.') and part not in {'.env', '.gitignore',
                                                     '.dockerignore',
                                                     '.babelrc'}:
                # Allow .env.example and similar
                if not (part.startswith('.env') or part.endswith('.example')):
                    return False

        return True


class BaseProcessor:
    """Base class for document processors."""

    def __init__(self):
        self.supported_extensions: Set[str] = set()

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions

    def process(self, file_path: Path) -> Optional[Document]:
        """Process a file and return a Document object."""
        raise NotImplementedError("Subclasses must implement process method")

    def _create_document(self, file_path: Path, content: str, file_type: str
                         ) -> Document:
        """Create a Document object with metadata."""
        stat = file_path.stat()

        return Document(
            id=str(file_path),
            path=str(file_path),
            content=content,
            file_type=file_type,
            metadata={
                'size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'extension': file_path.suffix.lower(),
                'name': file_path.name,
                'parent_dir': str(file_path.parent)
            },
            created_at=datetime.now()
        )


class TextProcessor(BaseProcessor):
    """Processor for text-based files."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {
            '.txt', '.md', '.py', '.js', '.jsx', '.ts', '.tsx', '.json',
            '.html', '.css', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.sh', '.env', '.gitignore', '.dockerignore', '.babelrc',
            '.npmrc', '.nvmrc', '.python-version', '.sample', '.for_test',
            '.mjs', '.cjs', '.hbs'
        }

    def process(self, file_path: Path) -> Optional[Document]:
        """Process a text file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            content = None

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                logger.warning(
                    f"Could not decode file {file_path} with any encoding")
                return None

            # Determine file type based on extension
            extension = file_path.suffix.lower()
            if extension in {'.py'}:
                file_type = 'python'
            elif extension in {'.js', '.jsx', '.mjs', '.cjs'}:
                file_type = 'javascript'
            elif extension in {'.ts', '.tsx'}:
                file_type = 'typescript'
            elif extension in {'.html'}:
                file_type = 'html'
            elif extension in {'.css'}:
                file_type = 'css'
            elif extension in {'.md'}:
                file_type = 'markdown'
            elif extension in {'.json'}:
                file_type = 'json'
            elif extension in {'.yaml', '.yml'}:
                file_type = 'yaml'
            elif extension in {'.toml'}:
                file_type = 'toml'
            elif extension in {'.sh'}:
                file_type = 'shell'
            else:
                file_type = 'text'

            return self._create_document(file_path, content, file_type)

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return None


class PDFProcessor(BaseProcessor):
    """Processor for PDF files."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.pdf'}

    def process(self, file_path: Path) -> Optional[Document]:
        """Process a PDF file."""
        if pypdf is None:
            logger.error("pypdf not available. Cannot process PDF files.")
            return None

        try:
            content = ""

            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)

                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            content += f"\n--- Page {page_num + 1} ---\n"
                            content += page_text
                    except Exception as e:
                        logger.warning(
                            f"Could not extract text from page {page_num + 1}"
                            f" of {file_path}: {e}")

            if not content.strip():
                logger.warning(
                    f"No text content extracted from PDF {file_path}")
                return None

            return self._create_document(file_path, content, 'pdf')

        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            return None


class DocumentProcessorManager:
    """Manages multiple document processors and file filtering."""

    def __init__(self, repo_path: str):
        """Initialize the processor manager."""
        self.repo_path = Path(repo_path)
        self.file_filter = FileFilter(repo_path)

        # Initialize processors
        self.processors = [
            TextProcessor(),
            PDFProcessor()
        ]

    def get_all_files(self) -> List[Path]:
        """Get all files in the repository that should be processed."""
        all_files = []

        for file_path in self.repo_path.rglob('*'):
            if self.file_filter.should_include_file(file_path):
                all_files.append(file_path)

        return sorted(all_files)

    def process_file(self, file_path: Path) -> Optional[Document]:
        """Process a single file using the appropriate processor."""
        # Find a processor that can handle this file
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor.process(file_path)

        logger.warning(f"No processor found for file {file_path}")
        return None

    def process_repository(self) -> List[Document]:
        """Process all files in the repository."""
        documents = []
        files_to_process = self.get_all_files()

        logger.info(
            f"Processing {len(files_to_process)} files from {self.repo_path}")

        for file_path in files_to_process:
            try:
                document = self.process_file(file_path)
                if document:
                    documents.append(document)
                    logger.debug(f"Processed: {file_path}")
                else:
                    logger.warning(f"Failed to process: {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

        logger.info(f"Successfully processed {len(documents)} documents")
        return documents

    def get_file_stats(self) -> Dict[str, Any]:
        """Get statistics about files in the repository."""
        all_files = self.get_all_files()

        stats = {
            'total_files': len(all_files),
            'file_types': {},
            'total_size': 0
        }

        for file_path in all_files:
            try:
                extension = file_path.suffix.lower() or 'no_extension'
                stats['file_types'][extension] = stats['file_types'].get(
                    extension, 0) + 1
                stats['total_size'] += file_path.stat().st_size
            except Exception as e:
                logger.warning(f"Could not get stats for {file_path}: {e}")

        return stats


# Convenience functions for external use
def create_processor_manager(repo_path: str) -> DocumentProcessorManager:
    """Create a document processor manager for the given repository path."""
    return DocumentProcessorManager(repo_path)


def process_repository(repo_path: str) -> List[Document]:
    """Process all files in a repository and return documents."""
    manager = create_processor_manager(repo_path)
    return manager.process_repository()


def get_repository_stats(repo_path: str) -> Dict[str, Any]:
    """Get statistics about files in a repository."""
    manager = create_processor_manager(repo_path)
    return manager.get_file_stats()
