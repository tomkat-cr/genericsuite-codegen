#!/usr/bin/env python3
"""
Test script for the document retrieval tool.
"""

from genericsuite_codegen.agent.document_retrieval_tool import (
    DocumentRetrievalTool
)
from genericsuite_codegen.agent.enhanced_search_types import (
    DocumentRetrievalError
)
import os
import sys

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))


def test_document_retrieval_tool():
    """Test the document retrieval tool functionality."""
    print("Testing DocumentRetrievalTool...")

    # Initialize the tool
    tool = DocumentRetrievalTool()
    print(f"✓ Tool initialized with base path: {tool.base_path}")

    # Test cases
    test_cases = [
        # Valid document paths (adjust these based on your actual files)
        "genericsuite-basecamp/README.md",
        "genericsuite-basecamp/docs/index.md",
        "genericsuite-basecamp/docs/Configuration-Guide/index.md",
    ]

    print("\n--- Testing Single Document Retrieval ---")
    for test_path in test_cases:
        try:
            print(f"\nTesting: {test_path}")

            # Test metadata retrieval first
            metadata = tool.get_document_metadata(test_path)
            print(f"  Metadata - Exists: {metadata.exists}, "
                  f"Size: {metadata.size} bytes, Type: {metadata.file_type}")

            if metadata.exists and metadata.is_readable:
                # Test full document retrieval
                document = tool.retrieve_document(test_path)
                content_preview = (document.content[:200] + "..."
                                   if len(document.content) > 200
                                   else document.content)
                print(f"  ✓ Retrieved document: "
                      f"{len(document.content)} characters")
                print(f"  Content preview: {content_preview}")
                print(f"  Encoding: {document.encoding}")
                print(f"  Last modified: {document.last_modified}")
            else:
                print(f"  ⚠ Document not accessible: "
                      f"{metadata.error_message}")

        except DocumentRetrievalError as e:
            print(f"  ✗ Document retrieval error: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")

    print("\n--- Testing Batch Document Retrieval ---")
    try:
        # Test batch retrieval with a mix of valid and invalid paths
        batch_paths = test_cases + ["nonexistent/file.txt"]
        documents = tool.retrieve_multiple_documents(batch_paths)
        print(f"✓ Batch retrieval completed: {len(documents)} documents "
              f"retrieved out of {len(batch_paths)} requested")

        for doc in documents:
            print(f"  - {doc.path}: {len(doc.content)} characters, "
                  f"{doc.file_type}")

    except Exception as e:
        print(f"✗ Batch retrieval error: {e}")

    print("\n--- Testing Error Handling ---")
    # Test with invalid path
    try:
        # Should be blocked
        tool.retrieve_document("../../../etc/passwd")
        print("✗ Security test failed - path traversal not blocked")
    except DocumentRetrievalError as e:
        print(f"✓ Correctly caught path traversal attempt: {e}")

    # Test with non-existent file
    try:
        tool.retrieve_document("nonexistent/file.txt")
        print("✗ Error handling test failed - should have thrown error")
    except DocumentRetrievalError as e:
        print(f"✓ Correctly caught file not found error: {e}")

    print("\n✓ Document retrieval tool tests completed!")


if __name__ == "__main__":
    test_document_retrieval_tool()
