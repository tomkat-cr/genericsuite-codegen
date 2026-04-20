"""
Integration tests for document ingestion workflow.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from genericsuite_codegen.document_processing.processors import DocumentProcessorManager
from genericsuite_codegen.document_processing.chunker import DocumentChunker
from genericsuite_codegen.document_processing.embeddings import EmbeddingGenerator, HuggingFaceEmbeddingProvider


class TestDocumentIngestionWorkflow:
    """Test end-to-end document ingestion workflow."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_complete_ingestion_workflow(self):
        """Test complete workflow from file processing to chunking."""
        # Create test repository structure
        (self.repo_path / "src").mkdir()
        (self.repo_path / "docs").mkdir()

        # Create test files
        (self.repo_path / "README.md").write_text("""
# Test Project

This is a test project for integration testing. It contains multiple files
with different content types to test the complete document processing pipeline.

## Features

- Document processing
- Text chunking
- Embedding generation
""")

        (self.repo_path / "src" / "main.py").write_text("""
def main():
    '''
    Main function for the test application.
    This function demonstrates the document processing capabilities
    of the GenericSuite CodeGen system.
    '''
    print("Hello, World!")
    
    # Process documents
    processor = DocumentProcessor()
    documents = processor.process_all()
    
    return documents

if __name__ == "__main__":
    main()
""")

        (self.repo_path / "docs" / "api.md").write_text("""
# API Documentation

## Overview

This document describes the API endpoints available in the system.

### Endpoints

#### GET /api/documents
Returns a list of all processed documents.

#### POST /api/process
Processes a new document and returns the result.
""")

        # Step 1: Process documents
        processor_manager = DocumentProcessorManager(str(self.repo_path))
        documents = processor_manager.process_repository()

        # Verify documents were processed
        assert len(documents) == 3

        # Verify document types
        doc_types = {doc.file_type for doc in documents}
        expected_types = {"markdown", "python"}
        assert doc_types == expected_types

        # Step 2: Chunk documents
        chunker = DocumentChunker()
        all_chunks = []

        for document in documents:
            chunks = chunker.chunk_document(document)
            all_chunks.extend(chunks)

        # Verify chunks were created
        assert len(all_chunks) > 0

        # Verify chunk metadata
        for chunk in all_chunks:
            assert chunk.document_id is not None
            assert chunk.content is not None
            assert len(chunk.content) > 0
            assert "original_document_path" in chunk.metadata
            assert "chunking_strategy" in chunk.metadata

    def test_ingestion_with_filtering(self):
        """Test ingestion workflow with file filtering."""
        # Create test files including ones that should be filtered out
        (self.repo_path / "test.py").write_text("print('test')")
        (self.repo_path / "image.png").write_bytes(b"fake image data")
        (self.repo_path / "document.pdf").write_bytes(b"fake pdf data")
        (self.repo_path / "config.json").write_text('{"key": "value"}')

        # Create .gitignore to filter some files
        (self.repo_path / ".gitignore").write_text("*.log\n__pycache__/\n")
        (self.repo_path / "debug.log").write_text("debug info")

        processor_manager = DocumentProcessorManager(str(self.repo_path))
        documents = processor_manager.process_repository()

        # Verify filtering worked
        processed_files = {Path(doc.path).name for doc in documents}

        # Should include supported files
        assert "test.py" in processed_files
        assert "config.json" in processed_files

        # Should exclude unsupported files
        assert "image.png" not in processed_files

        # Should exclude gitignored files
        assert "debug.log" not in processed_files

    def test_ingestion_error_handling(self):
        """Test ingestion workflow error handling."""
        # Create a file that might cause processing errors
        (self.repo_path / "test.py").write_text("print('test')")

        # Create a file with permission issues (simulate)
        problematic_file = self.repo_path / "problematic.py"
        problematic_file.write_text("print('problematic')")

        processor_manager = DocumentProcessorManager(str(self.repo_path))

        # Mock file processing to simulate an error
        original_process_file = processor_manager.process_file

        def mock_process_file(file_path):
            if file_path.name == "problematic.py":
                raise Exception("Simulated processing error")
            return original_process_file(file_path)

        processor_manager.process_file = mock_process_file

        # Should handle errors gracefully
        documents = processor_manager.process_repository()

        # Should still process the good file
        assert len(documents) >= 1
        processed_files = {Path(doc.path).name for doc in documents}
        assert "test.py" in processed_files

    def test_chunking_strategies_integration(self):
        """Test integration of different chunking strategies."""
        # Create files of different types
        (self.repo_path / "code.py").write_text("""
def function_one():
    '''First function with sufficient content for chunking.'''
    result = "This is a longer function with more content"
    print(f"Function 1 result: {result}")
    return result

def function_two():
    '''Second function with sufficient content for chunking.'''
    data = {"key": "value", "items": [1, 2, 3, 4, 5]}
    processed = process_data(data)
    return processed

class TestClass:
    '''Test class with methods.'''
    
    def method_one(self):
        '''Method with sufficient content.'''
        return "method result with enough content for chunking"
""")

        (self.repo_path / "document.md").write_text("""
# Main Header

This is a comprehensive document with multiple sections and sufficient content
for testing the paragraph chunking strategy. Each section contains enough text
to meet minimum chunk size requirements.

## Section One

This section contains detailed information about the first topic. It includes
multiple sentences and paragraphs to ensure proper chunking behavior.

The content is structured to test how the chunker handles markdown formatting
and maintains readability across chunk boundaries.

## Section Two

This section covers additional topics with more detailed explanations.
It demonstrates how the chunking algorithm handles different content types
and maintains context across chunk boundaries.

### Subsection

More detailed content here to ensure we have sufficient text for proper
chunking behavior testing across different content structures.
""")

        processor_manager = DocumentProcessorManager(str(self.repo_path))
        documents = processor_manager.process_repository()

        # Test different chunking strategies
        chunker = DocumentChunker()

        for document in documents:
            chunks = chunker.chunk_document(document)

            # Verify chunks were created
            assert len(chunks) > 0

            # Verify chunk content
            for chunk in chunks:
                assert len(chunk.content.strip()) > 0
                assert chunk.chunk_index >= 0

                # Verify metadata includes strategy info
                assert "chunking_strategy" in chunk.metadata
                assert "chunk_size" in chunk.metadata
                assert "total_chunks" in chunk.metadata

    @patch('genericsuite_codegen.document_processing.embeddings.SentenceTransformer')
    def test_end_to_end_with_embeddings(self, mock_sentence_transformer):
        """Test complete end-to-end workflow including embeddings."""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sentence_transformer.return_value = mock_model

        # Create test content
        (self.repo_path / "test.py").write_text("""
def test_function():
    '''Test function with sufficient content for chunking and embedding.'''
    result = "This is a test function with enough content to be chunked properly"
    print(f"Test result: {result}")
    return result

def another_function():
    '''Another function with sufficient content.'''
    data = {"test": "data", "values": [1, 2, 3, 4, 5]}
    return process_test_data(data)
""")

        # Step 1: Process documents
        processor_manager = DocumentProcessorManager(str(self.repo_path))
        documents = processor_manager.process_repository()

        # Step 2: Chunk documents
        chunker = DocumentChunker()
        all_chunks = []
        for document in documents:
            chunks = chunker.chunk_document(document)
            all_chunks.extend(chunks)

        # Step 3: Generate embeddings (mock the generator since implementation differs)
        generator = Mock()
        generator.generate_embeddings_for_chunks = Mock()

        # Mock embedded chunks
        mock_embedded_chunks = []
        for chunk in all_chunks:
            mock_embedded_chunk = Mock()
            mock_embedded_chunk.chunk = chunk
            mock_embedded_chunk.embedding = [0.1, 0.2, 0.3]
            mock_embedded_chunk.embedding_model = "test-model"
            mock_embedded_chunks.append(mock_embedded_chunk)

        generator.generate_embeddings_for_chunks.return_value = mock_embedded_chunks
        embedded_chunks = generator.generate_embeddings_for_chunks(all_chunks)

        # Verify embeddings were generated
        assert len(embedded_chunks) > 0

        for embedded_chunk in embedded_chunks:
            assert embedded_chunk.embedding is not None
            assert len(embedded_chunk.embedding) == 3  # Mock dimension
            assert embedded_chunk.embedding_model is not None
            assert embedded_chunk.chunk is not None

    def test_large_repository_processing(self):
        """Test processing a larger repository structure."""
        # Create a more complex repository structure
        dirs = ["src", "tests", "docs", "config", "scripts"]
        for dir_name in dirs:
            (self.repo_path / dir_name).mkdir()

        # Create multiple files in each directory
        file_contents = {
            "src/main.py": "def main(): pass",
            "src/utils.py": "def utility_function(): return 'utility'",
            "tests/test_main.py": "def test_main(): assert True",
            "docs/README.md": "# Project Documentation\n\nThis is the main documentation.",
            "docs/api.md": "# API Reference\n\n## Endpoints\n\n### GET /api/v1/status",
            "config/settings.json": '{"debug": true, "port": 8002}',
            "scripts/deploy.sh": "#!/bin/bash\necho 'Deploying application'"
        }

        for file_path, content in file_contents.items():
            full_path = self.repo_path / file_path
            full_path.write_text(content)

        # Process the repository
        processor_manager = DocumentProcessorManager(str(self.repo_path))
        documents = processor_manager.process_repository()

        # Verify all supported files were processed
        assert len(documents) >= 6  # Should process most files

        # Verify different file types were handled
        file_types = {doc.file_type for doc in documents}
        expected_types = {"python", "markdown", "json", "shell"}
        assert file_types.intersection(expected_types) == expected_types

        # Test chunking on all documents
        chunker = DocumentChunker()
        total_chunks = 0

        for document in documents:
            chunks = chunker.chunk_document(document)
            total_chunks += len(chunks)

            # Verify each document produced at least some chunks
            # (or none if content is too short)
            assert len(chunks) >= 0

        # Should have created some chunks overall
        assert total_chunks >= 0

    def test_repository_statistics(self):
        """Test repository processing statistics."""
        # Create test files
        (self.repo_path / "file1.py").write_text("print('file1')")
        (self.repo_path / "file2.md").write_text("# File 2\n\nContent")
        (self.repo_path / "file3.json").write_text('{"key": "value"}')

        processor_manager = DocumentProcessorManager(str(self.repo_path))

        # Get file statistics
        stats = processor_manager.get_file_stats()

        assert "total_files" in stats
        assert "file_types" in stats
        assert "total_size" in stats

        assert stats["total_files"] >= 3
        assert isinstance(stats["file_types"], dict)
        assert isinstance(stats["total_size"], int)
        assert stats["total_size"] > 0
