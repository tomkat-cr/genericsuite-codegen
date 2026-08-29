"""
Unit tests for document processing chunker module.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from genericsuite_codegen.document_processing.chunker import (
    DocumentChunk,
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    CodeChunker,
    AdaptiveChunker,
    DocumentChunker,
    chunk_document,
    get_chunking_stats
)


class TestDocumentChunk:
    """Test DocumentChunk data class."""

    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk."""
        chunk = DocumentChunk(
            id="test_chunk_1",
            document_id="doc_1",
            content="Test content",
            chunk_index=0,
            metadata={"source": "test.py"}
        )

        assert chunk.id == "test_chunk_1"
        assert chunk.document_id == "doc_1"
        assert chunk.content == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.metadata == {"source": "test.py"}


class TestFixedSizeChunker:
    """Test FixedSizeChunker class."""

    def test_fixed_size_chunker_basic(self):
        """Test basic fixed-size chunking."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        text = "This is a test text for chunking"
        metadata = {"file_type": "text"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) > 1
        # Allow for word boundaries
        assert all(len(chunk) <= 12 for chunk in chunks)

    def test_fixed_size_chunker_empty_text(self):
        """Test chunking empty text."""
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("", {})
        assert chunks == []

    def test_fixed_size_chunker_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("   \n\t  ", {})
        assert chunks == []

    def test_fixed_size_chunker_word_boundaries(self):
        """Test that chunker respects word boundaries."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        text = "word1 word2 word3 word4 word5"
        metadata = {"file_type": "text"}

        chunks = chunker.chunk(text, metadata)

        # Check that words are not split
        for chunk in chunks:
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")

    def test_fixed_size_chunker_no_overlap(self):
        """Test chunking without overlap."""
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        text = "12345678901234567890"
        metadata = {"file_type": "text"}

        chunks = chunker.chunk(text, metadata)

        # Reconstruct text (approximately, due to word boundaries)
        reconstructed_length = sum(len(chunk) for chunk in chunks)
        assert reconstructed_length <= len(text)


class TestSentenceChunker:
    """Test SentenceChunker class."""

    def test_sentence_chunker_basic(self):
        """Test basic sentence chunking."""
        chunker = SentenceChunker(max_chunk_size=50, min_chunk_size=10)
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        metadata = {"file_type": "text"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) >= 1
        assert all(len(chunk) >= 10 for chunk in chunks)
        assert all(len(chunk) <= 50 for chunk in chunks)

    def test_sentence_chunker_long_sentence(self):
        """Test handling of sentences longer than max_chunk_size."""
        chunker = SentenceChunker(max_chunk_size=20, min_chunk_size=5)
        text = "This is a very long sentence that exceeds the maximum chunk size limit."
        metadata = {"file_type": "text"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) >= 1
        # Should fall back to fixed-size chunking for long sentences

    def test_sentence_chunker_empty_text(self):
        """Test sentence chunking with empty text."""
        chunker = SentenceChunker()
        chunks = chunker.chunk("", {})
        assert chunks == []

    def test_sentence_chunker_no_sentences(self):
        """Test text without sentence boundaries."""
        chunker = SentenceChunker(max_chunk_size=50, min_chunk_size=10)
        text = "This is text without proper sentence endings"
        metadata = {"file_type": "text"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) == 1
        assert chunks[0] == text


class TestParagraphChunker:
    """Test ParagraphChunker class."""

    def test_paragraph_chunker_basic(self):
        """Test basic paragraph chunking."""
        chunker = ParagraphChunker(max_chunk_size=100, min_chunk_size=20)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        metadata = {"file_type": "text"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) >= 1
        assert all(len(chunk) >= 20 for chunk in chunks)
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_paragraph_chunker_single_newlines(self):
        """Test fallback to single newlines when no double newlines."""
        chunker = ParagraphChunker(max_chunk_size=50, min_chunk_size=10)
        text = "First line\nSecond line\nThird line"
        metadata = {"file_type": "text"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) >= 1

    def test_paragraph_chunker_long_paragraph(self):
        """Test handling of paragraphs longer than max_chunk_size."""
        chunker = ParagraphChunker(max_chunk_size=30, min_chunk_size=10)
        text = "This is a very long paragraph that exceeds the maximum chunk size and should be split."
        metadata = {"file_type": "text"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) >= 1
        # Should fall back to sentence chunking for long paragraphs


class TestCodeChunker:
    """Test CodeChunker class."""

    def test_code_chunker_python(self):
        """Test Python code chunking."""
        chunker = CodeChunker(max_chunk_size=200, min_chunk_size=50)
        python_code = """
def function1():
    # This is a longer function with more content
    # to ensure it meets the minimum chunk size requirement
    result = "test1"
    print(f"Function 1 result: {result}")
    return result

def function2():
    # This is another longer function with more content
    # to ensure it meets the minimum chunk size requirement
    result = "test2"
    print(f"Function 2 result: {result}")
    return result

class TestClass:
    def method1(self):
        # This method has enough content to meet minimum size
        result = "method1"
        print(f"Method 1 result: {result}")
        return result
"""
        metadata = {"file_type": "python"}

        chunks = chunker.chunk(python_code, metadata)

        assert len(chunks) >= 1
        # Should split at function/class boundaries

    def test_code_chunker_javascript(self):
        """Test JavaScript code chunking."""
        chunker = CodeChunker(max_chunk_size=200, min_chunk_size=50)
        js_code = """
function test1() {
    return "test1";
}

const test2 = () => {
    return "test2";
}
"""
        metadata = {"file_type": "javascript"}

        chunks = chunker.chunk(js_code, metadata)

        assert len(chunks) >= 1

    def test_code_chunker_other_language(self):
        """Test code chunking for unsupported language."""
        chunker = CodeChunker(max_chunk_size=50, min_chunk_size=10)
        code = "some code in unknown language"
        metadata = {"file_type": "unknown"}

        chunks = chunker.chunk(code, metadata)

        assert len(chunks) >= 1
        # Should fall back to paragraph chunking

    def test_code_chunker_empty_code(self):
        """Test code chunking with empty code."""
        chunker = CodeChunker()
        chunks = chunker.chunk("", {"file_type": "python"})
        assert chunks == []


class TestAdaptiveChunker:
    """Test AdaptiveChunker class."""

    def test_adaptive_chunker_python(self):
        """Test adaptive chunking for Python files."""
        chunker = AdaptiveChunker()
        code = """
def test_function():
    # This is a longer function with more content
    # to ensure it meets the minimum chunk size requirement
    result = "test"
    print(f"Test result: {result}")
    return result

def another_function():
    # Another function with sufficient content
    # to meet minimum chunk size requirements
    data = {"key": "value"}
    processed = process_data(data)
    return processed
"""
        metadata = {"file_type": "python"}

        chunks = chunker.chunk(code, metadata)

        assert len(chunks) >= 1

    def test_adaptive_chunker_markdown(self):
        """Test adaptive chunking for Markdown files."""
        chunker = AdaptiveChunker()
        text = """# Header

This is a longer paragraph with sufficient content to meet the minimum chunk size requirements for the adaptive chunker. It contains multiple sentences and provides enough text to be processed properly.

## Another Header

This is another paragraph with additional content to ensure that the chunking process works correctly with markdown files. The content is structured with headers and paragraphs as typical markdown would be.

### Subsection

More content here to make sure we have enough text for proper chunking behavior testing.
"""
        metadata = {"file_type": "markdown"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) >= 1

    def test_adaptive_chunker_json(self):
        """Test adaptive chunking for JSON files."""
        chunker = AdaptiveChunker()
        json_text = '{"key": "value", "array": [1, 2, 3]}'
        metadata = {"file_type": "json"}

        chunks = chunker.chunk(json_text, metadata)

        assert len(chunks) >= 1

    def test_adaptive_chunker_default(self):
        """Test adaptive chunking for unknown file types."""
        chunker = AdaptiveChunker()
        text = """Some text content with sufficient length to meet the minimum chunk size requirements. This text contains multiple sentences and provides enough content for the adaptive chunker to process properly. The chunker should handle unknown file types by falling back to sentence-based chunking, which requires a minimum amount of text to create valid chunks."""
        metadata = {"file_type": "unknown"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) >= 1


class TestDocumentChunker:
    """Test DocumentChunker class."""

    def test_document_chunker_basic(self):
        """Test basic document chunking."""
        mock_document = Mock()
        mock_document.id = "doc_1"
        mock_document.path = "test.py"
        mock_document.content = """
def test_function():
    # This is a longer function with more content
    # to ensure it meets the minimum chunk size requirement
    result = "test"
    print(f"Test result: {result}")
    return result

def another_function():
    # Another function with sufficient content
    data = {"key": "value"}
    return data
"""
        mock_document.file_type = "python"
        mock_document.metadata = {"file_type": "python"}

        chunker = DocumentChunker()
        chunks = chunker.chunk_document(mock_document)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_id == "doc_1" for chunk in chunks)

    def test_document_chunker_empty_document(self):
        """Test chunking empty document."""
        mock_document = Mock()
        mock_document.id = "doc_1"
        mock_document.path = "empty.txt"
        mock_document.content = ""
        mock_document.file_type = "text"
        mock_document.metadata = {"file_type": "text"}

        chunker = DocumentChunker()
        chunks = chunker.chunk_document(mock_document)

        assert chunks == []

    def test_document_chunker_metadata_preservation(self):
        """Test that metadata is preserved and extended."""
        mock_document = Mock()
        mock_document.id = "doc_1"
        mock_document.path = "test.py"
        mock_document.content = """
def test_function():
    # This is a longer function with more content
    # to ensure it meets the minimum chunk size requirement
    result = "test"
    print(f"Test result: {result}")
    return result

def another_function():
    # Another function with sufficient content
    data = {"key": "value"}
    return data
"""
        mock_document.file_type = "python"
        mock_document.metadata = {"file_type": "python", "custom": "value"}

        chunker = DocumentChunker()
        chunks = chunker.chunk_document(mock_document)

        assert len(chunks) >= 1
        chunk = chunks[0]

        # Check original metadata is preserved
        assert chunk.metadata["custom"] == "value"

        # Check extended metadata is added
        assert "original_document_id" in chunk.metadata
        assert "original_document_path" in chunk.metadata
        assert "chunk_size" in chunk.metadata
        assert "total_chunks" in chunk.metadata
        assert "chunking_strategy" in chunk.metadata
        assert "created_at" in chunk.metadata

    def test_document_chunker_custom_strategy(self):
        """Test document chunker with custom strategy."""
        mock_strategy = Mock()
        mock_strategy.chunk.return_value = ["chunk1", "chunk2"]

        mock_document = Mock()
        mock_document.id = "doc_1"
        mock_document.path = "test.txt"
        mock_document.content = "test content"
        mock_document.file_type = "text"
        mock_document.metadata = {"file_type": "text"}

        chunker = DocumentChunker(strategy=mock_strategy)
        chunks = chunker.chunk_document(mock_document)

        assert len(chunks) == 2
        mock_strategy.chunk.assert_called_once_with(
            "test content", {"file_type": "text"})

    def test_get_optimal_chunk_size(self):
        """Test optimal chunk size calculation."""
        chunker = DocumentChunker()

        # Test with OpenAI model
        size = chunker.get_optimal_chunk_size("short text", "openai-ada-002")
        assert size == len("short text")

        # Test with sentence transformers
        long_text = "a" * 1000
        size = chunker.get_optimal_chunk_size(
            long_text, "sentence-transformers/all-MiniLM-L6-v2")
        assert size == 400

        # Test with default model
        size = chunker.get_optimal_chunk_size(long_text, "default")
        assert size == 500  # max_size // 2 because text_length < max_size * 2

    def test_set_strategy(self):
        """Test changing chunking strategy."""
        chunker = DocumentChunker()
        original_strategy = chunker.strategy

        new_strategy = FixedSizeChunker()
        chunker.set_strategy(new_strategy)

        assert chunker.strategy is new_strategy
        assert chunker.strategy is not original_strategy


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_chunk_document_function(self):
        """Test chunk_document convenience function."""
        mock_document = Mock()
        mock_document.id = "doc_1"
        mock_document.path = "test.py"
        mock_document.content = "def test(): pass"
        mock_document.file_type = "python"
        mock_document.metadata = {"file_type": "python"}

        chunks = chunk_document(mock_document, strategy="fixed", chunk_size=50)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

    def test_chunk_document_strategies(self):
        """Test different strategies in chunk_document function."""
        mock_document = Mock()
        mock_document.id = "doc_1"
        mock_document.path = "test.txt"
        mock_document.content = """This is test content for chunking with sufficient length to meet minimum chunk size requirements. The content contains multiple sentences and provides enough text for the chunking algorithms to work properly. This ensures that the convenience functions can be tested effectively with realistic content that meets the minimum size thresholds."""
        mock_document.file_type = "text"
        mock_document.metadata = {"file_type": "text"}

        strategies = ["fixed", "sentence", "paragraph", "code", "adaptive"]

        for strategy in strategies:
            chunks = chunk_document(mock_document, strategy=strategy)
            assert len(chunks) >= 1

    def test_get_chunking_stats(self):
        """Test get_chunking_stats function."""
        chunks = [
            DocumentChunk("1", "doc_1", "content1", 0, {
                          "chunking_strategy": "FixedSizeChunker"}),
            DocumentChunk("2", "doc_1", "content22", 1, {
                          "chunking_strategy": "FixedSizeChunker"}),
            DocumentChunk("3", "doc_1", "content333", 2, {
                          "chunking_strategy": "SentenceChunker"})
        ]

        stats = get_chunking_stats(chunks)

        assert stats["total_chunks"] == 3
        assert stats["total_characters"] == 8 + 9 + 10  # len of contents
        assert stats["average_chunk_size"] == (8 + 9 + 10) / 3
        assert stats["min_chunk_size"] == 8
        assert stats["max_chunk_size"] == 10
        assert set(stats["strategies_used"]) == {
            "FixedSizeChunker", "SentenceChunker"}

    def test_get_chunking_stats_empty(self):
        """Test get_chunking_stats with empty list."""
        stats = get_chunking_stats([])
        assert stats == {}
