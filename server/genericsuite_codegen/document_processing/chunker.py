"""
Text chunking functionality for document processing.

This module provides configurable text chunking algorithms optimized for
embedding generation while preserving metadata during the chunking process.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any]


class ChunkingStrategy:
    """Base class for chunking strategies."""

    def chunk(self, text: str, document_metadata: Dict[str, Any]) -> List[str]:
        """Chunk text into smaller pieces."""
        raise NotImplementedError("Subclasses must implement chunk method")


class FixedSizeChunker(ChunkingStrategy):
    """Chunks text into fixed-size pieces with optional overlap."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, document_metadata: Dict[str, Any]) -> List[str]:
        """Chunk text into fixed-size pieces."""
        if not text.strip():
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If this is not the last chunk, try to break at a word boundary
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + 1, end - self.overlap)

            # Prevent infinite loop
            if start >= len(text):
                break

        return chunks


class SentenceChunker(ChunkingStrategy):
    """Chunks text by sentences with size limits."""

    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        """
        Initialize sentence-based chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

        # Sentence boundary patterns
        self.sentence_pattern = re.compile(r'[.!?]+\s+')

    def chunk(self, text: str, document_metadata: Dict[str, Any]) -> List[str]:
        """Chunk text by sentences."""
        if not text.strip():
            return []

        # Split into sentences
        sentences = self.sentence_pattern.split(text)
        if not sentences:
            return [text]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding this sentence would exceed max size
            potential_chunk = current_chunk + " " + sentence \
                if current_chunk else sentence

            if len(potential_chunk) <= self.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it meets minimum size
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)

                # Start new chunk with current sentence
                if len(sentence) <= self.max_chunk_size:
                    current_chunk = sentence
                else:
                    # Sentence is too long, split it using fixed-size chunker
                    fixed_chunker = FixedSizeChunker(self.max_chunk_size, 50)
                    sentence_chunks = fixed_chunker.chunk(
                        sentence, document_metadata)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""

        # Add the last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)

        return chunks


class ParagraphChunker(ChunkingStrategy):
    """Chunks text by paragraphs with size limits."""

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 200):
        """
        Initialize paragraph-based chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, document_metadata: Dict[str, Any]) -> List[str]:
        """Chunk text by paragraphs."""
        if not text.strip():
            return []

        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not paragraphs:
            # Fallback to single newlines
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        if not paragraphs:
            return [text]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed max size
            potential_chunk = current_chunk + "\n\n" + \
                paragraph if current_chunk else paragraph

            if len(potential_chunk) <= self.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it meets minimum size
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)

                # Handle the current paragraph
                if len(paragraph) <= self.max_chunk_size:
                    current_chunk = paragraph
                else:
                    # Paragraph is too long, split it using sentence chunker
                    sentence_chunker = SentenceChunker(
                        self.max_chunk_size, self.min_chunk_size)
                    paragraph_chunks = sentence_chunker.chunk(
                        paragraph, document_metadata)
                    chunks.extend(paragraph_chunks)
                    current_chunk = ""

        # Add the last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)

        return chunks


class CodeChunker(ChunkingStrategy):
    """Specialized chunker for code files."""

    def __init__(self, max_chunk_size: int = 1200, min_chunk_size: int = 150):
        """
        Initialize code-specific chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, document_metadata: Dict[str, Any]) -> List[str]:
        """Chunk code by logical blocks (functions, classes, etc.)."""
        if not text.strip():
            return []

        file_type = document_metadata.get('file_type', 'text')

        # For Python files, try to split by functions and classes
        if file_type == 'python':
            return self._chunk_python_code(text)
        elif file_type in ['javascript', 'typescript']:
            return self._chunk_js_code(text)
        else:
            # Fallback to paragraph chunking for other code types
            paragraph_chunker = ParagraphChunker(
                self.max_chunk_size, self.min_chunk_size)
            return paragraph_chunker.chunk(text, document_metadata)

    def _chunk_python_code(self, text: str) -> List[str]:
        """Chunk Python code by functions and classes."""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        # TODO: why not use the current_indent?
        # current_indent = 0

        for line in lines:
            stripped = line.strip()

            # Check for function or class definitions
            if (stripped.startswith('def ') or stripped.startswith('class ') or
                    stripped.startswith('async def ')):

                # Save previous chunk if it exists and meets minimum size
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = []

                # TODO: Add support for indenting ?? current_indent was
                # declared but not used
                # current_indent = len(line) - len(line.lstrip())

            current_chunk.append(line)

            # Check if chunk is getting too large
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) > self.max_chunk_size:
                # Try to find a good breaking point
                if len(current_chunk) > 1:
                    # Keep the last line for the next chunk
                    last_line = current_chunk.pop()
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = [last_line]

        # Add the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def _chunk_js_code(self, text: str) -> List[str]:
        """Chunk JavaScript/TypeScript code by functions."""

        # TODO: why not use the function_pattern?
        # Simple function detection for JS/TS

        # function_pattern = re.compile(
        #     r'^\s*(function\s+\w+|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\
        # s*=|\w+\s*:\s*function|export\s+function)', re.MULTILINE)

        lines = text.split('\n')
        chunks = []
        current_chunk = []

        for line in lines:
            current_chunk.append(line)

            # Check if chunk is getting too large
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) > self.max_chunk_size:
                # Try to find a good breaking point
                if len(current_chunk) > 1:
                    # Keep the last line for the next chunk
                    last_line = current_chunk.pop()
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = [last_line]

        # Add the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks


class AdaptiveChunker(ChunkingStrategy):
    """
    Adaptive chunker that selects the best strategy based on content type.
    """

    def __init__(self,
                 default_chunk_size: int = 1000,
                 code_chunk_size: int = 1200,
                 markdown_chunk_size: int = 1500):
        """
        Initialize adaptive chunker.

        Args:
            default_chunk_size: Default chunk size for text files
            code_chunk_size: Chunk size for code files
            markdown_chunk_size: Chunk size for markdown files
        """
        self.default_chunk_size = default_chunk_size
        self.code_chunk_size = code_chunk_size
        self.markdown_chunk_size = markdown_chunk_size

    def chunk(self, text: str, document_metadata: Dict[str, Any]) -> List[str]:
        """Choose the best chunking strategy based on file type."""
        file_type = document_metadata.get('file_type', 'text')

        # Choose strategy based on file type
        if file_type in ['python', 'javascript', 'typescript']:
            chunker = CodeChunker(self.code_chunk_size)
        elif file_type == 'markdown':
            chunker = ParagraphChunker(self.markdown_chunk_size)
        elif file_type in ['json', 'yaml', 'toml']:
            # For structured data, use smaller chunks
            chunker = FixedSizeChunker(self.default_chunk_size // 2, 50)
        else:
            # Default to sentence-based chunking
            chunker = SentenceChunker(self.default_chunk_size)

        return chunker.chunk(text, document_metadata)


class DocumentChunker:
    """Main document chunking class that manages chunking strategies."""

    def __init__(self,
                 strategy: Optional[ChunkingStrategy] = None,
                 chunk_size: int = 1000,
                 overlap: int = 100):
        """
        Initialize document chunker.

        Args:
            strategy: Chunking strategy to use (defaults to AdaptiveChunker)
            chunk_size: Default chunk size
            overlap: Overlap between chunks for fixed-size chunking
        """
        if strategy is None:
            self.strategy = AdaptiveChunker(default_chunk_size=chunk_size)
        else:
            self.strategy = strategy

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Document object with content and metadata

        Returns:
            List of DocumentChunk objects
        """
        if not document.content.strip():
            return []

        # Get text chunks using the selected strategy
        text_chunks = self.strategy.chunk(document.content, document.metadata)

        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{document.id}_chunk_{i}"

            # Preserve and extend metadata
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                'original_document_id': document.id,
                # >>-->
                'original_document_path': document.path,
                'original_document_type': document.file_type,
                'chunk_size': len(chunk_text),
                'total_chunks': len(text_chunks),
                'chunking_strategy': self.strategy.__class__.__name__,
                'created_at': datetime.now().isoformat()
            })

            document_chunk = DocumentChunk(
                id=chunk_id,
                document_id=document.id,
                content=chunk_text,
                chunk_index=i,
                metadata=chunk_metadata
            )

            document_chunks.append(document_chunk)

        logger.debug(
            f"Chunked document {document.path} into "
            f"{len(document_chunks)} chunks")
        return document_chunks

    def get_optimal_chunk_size(self, text: str,
                               target_embedding_model: str = "default") -> int:
        """
        Calculate optimal chunk size based on text characteristics
        and embedding model.

        Args:
            text: Text to analyze
            target_embedding_model: Target embedding model name

        Returns:
            Recommended chunk size
        """
        text_length = len(text)

        # Model-specific optimizations
        if "openai" in target_embedding_model.lower():
            # OpenAI models work well with chunks up to 8192 tokens
            # (~6000 chars)
            max_size = 6000
        elif "sentence-transformers" in target_embedding_model.lower():
            # Sentence transformers typically have 512 token limits
            # (~400 chars)
            max_size = 400
        else:
            # Default conservative size
            max_size = 1000

        # Adjust based on text length
        if text_length < max_size:
            return text_length
        elif text_length < max_size * 2:
            return max_size // 2
        else:
            return max_size

    def set_strategy(self, strategy: ChunkingStrategy):
        """Change the chunking strategy."""
        self.strategy = strategy


# Convenience functions
def chunk_document(document, strategy: str = "adaptive", **kwargs
                   ) -> List[DocumentChunk]:
    """
    Convenience function to chunk a document.

    Args:
        document: Document to chunk
        strategy: Chunking strategy ("adaptive", "fixed", "sentence",
            "paragraph", "code")
        **kwargs: Additional arguments for the chunking strategy

    Returns:
        List of DocumentChunk objects
    """
    # Create strategy based on name
    if strategy == "fixed":
        chunking_strategy = FixedSizeChunker(**kwargs)
    elif strategy == "sentence":
        chunking_strategy = SentenceChunker(**kwargs)
    elif strategy == "paragraph":
        chunking_strategy = ParagraphChunker(**kwargs)
    elif strategy == "code":
        chunking_strategy = CodeChunker(**kwargs)
    else:  # adaptive
        chunking_strategy = AdaptiveChunker(**kwargs)

    chunker = DocumentChunker(strategy=chunking_strategy)
    return chunker.chunk_document(document)


def get_chunking_stats(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Get statistics about a list of chunks."""
    if not chunks:
        return {}

    chunk_sizes = [len(chunk.content) for chunk in chunks]

    return {
        'total_chunks': len(chunks),
        'total_characters': sum(chunk_sizes),
        'average_chunk_size': sum(chunk_sizes) / len(chunks),
        'min_chunk_size': min(chunk_sizes),
        'max_chunk_size': max(chunk_sizes),
        'strategies_used': list(set(chunk.metadata.get(
            'chunking_strategy', 'unknown')
            for chunk in chunks))
    }
