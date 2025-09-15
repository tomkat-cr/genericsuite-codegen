#!/usr/bin/env python3
"""
Simple test script to verify the document processing pipeline works.
This is a basic integration test to ensure all components work together.
"""

import os
import tempfile
from pathlib import Path

from processors import DocumentProcessorManager, Document
from chunker import chunk_document
from embeddings import EmbeddingGenerator, get_available_providers


def create_test_files(test_dir: Path):
    """Create some test files for processing."""
    # Create a Python file
    python_file = test_dir / "test.py"
    python_file.write_text("""
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return "Hello, World!"

class TestClass:
    '''A test class for demonstration.'''
    
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"
""")
    
    # Create a markdown file
    md_file = test_dir / "README.md"
    md_file.write_text("""
# Test Project

This is a test project for the document processing pipeline.

## Features

- Document processing
- Text chunking
- Embedding generation

## Usage

Simply run the pipeline and it will process all files.
""")
    
    # Create a JSON file
    json_file = test_dir / "config.json"
    json_file.write_text("""
{
    "name": "test-project",
    "version": "1.0.0",
    "description": "A test project configuration",
    "dependencies": {
        "python": ">=3.8"
    }
}
""")


def test_document_processing():
    """Test the document processing pipeline."""
    print("Testing document processing pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test files
        print("Creating test files...")
        create_test_files(test_dir)
        
        # Test file processing
        print("Testing file processing...")
        processor_manager = DocumentProcessorManager(str(test_dir))
        
        # Get file statistics
        stats = processor_manager.get_file_stats()
        print(f"File statistics: {stats}")
        
        # Process all files
        documents = processor_manager.process_repository()
        print(f"Processed {len(documents)} documents")
        
        for doc in documents:
            print(f"  - {doc.path} ({doc.file_type}): {len(doc.content)} characters")
        
        # Test chunking
        print("\nTesting chunking...")
        all_chunks = []
        for doc in documents:
            chunks = chunk_document(doc, strategy="adaptive")
            all_chunks.extend(chunks)
            print(f"  - {doc.path}: {len(chunks)} chunks")
        
        print(f"Total chunks: {len(all_chunks)}")
        
        # Test embedding providers (without actually generating embeddings)
        print("\nTesting embedding providers...")
        available_providers = get_available_providers()
        print(f"Available providers: {available_providers}")
        
        # Try to initialize embedding generators (if dependencies are available)
        for provider, models in available_providers.items():
            try:
                if models:  # If there are models available
                    model = models[0]  # Use first available model
                    print(f"  Testing {provider} with model {model}...")
                    generator = EmbeddingGenerator(provider=provider, model=model)
                    info = generator.get_model_info()
                    print(f"    Model info: {info}")
            except Exception as e:
                print(f"    Could not initialize {provider}: {e}")
        
        print("\nDocument processing pipeline test completed successfully!")


if __name__ == "__main__":
    test_document_processing()