"""
Pytest configuration and fixtures for enhanced search tests.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from genericsuite_codegen.database.setup import SearchResult
from genericsuite_codegen.agent.enhanced_search_types import (
    CodeGenerationContext,
    DocumentContent,
    DocumentMetadata,
    SearchTemplate
)


@pytest.fixture
def mock_search_result():
    """Create a mock SearchResult for testing."""
    return SearchResult(
        content="Test content",
        metadata={"source": "test.py", "type": "python"},
        similarity_score=0.85,
        document_path="test.py"
    )


@pytest.fixture
def mock_search_results():
    """Create multiple mock SearchResults for testing."""
    return [
        SearchResult(
            content="JSON configuration example",
            metadata={"source": "config.json", "type": "json"},
            similarity_score=0.9,
            document_path="config.json"
        ),
        SearchResult(
            content="Python tool implementation",
            metadata={"source": "tool.py", "type": "python"},
            similarity_score=0.8,
            document_path="tool.py"
        ),
        SearchResult(
            content="GenericSuite pattern example",
            metadata={"source": "pattern.md", "type": "markdown"},
            similarity_score=0.75,
            document_path="pattern.md"
        )
    ]


@pytest.fixture
def mock_kb_tool():
    """Create a mock KnowledgeBaseTool for testing."""
    mock_tool = Mock()
    mock_tool.search_knowledge_base = AsyncMock()
    return mock_tool


@pytest.fixture
def sample_code_context():
    """Create a sample CodeGenerationContext for testing."""
    return CodeGenerationContext(
        code_type="json",
        framework="genericsuite",
        confidence=0.85,
        detected_patterns=["table_config", "json_schema"]
    )


@pytest.fixture
def sample_search_templates():
    """Create sample search templates for testing."""
    return {
        "json": SearchTemplate(
            code_type="json",
            template="examples of JSON table configuration in GenericSuite",
            file_type_filter="json",
            priority=1
        ),
        "python": SearchTemplate(
            code_type="python",
            template="examples of Python tools in GenericSuite",
            file_type_filter="py",
            priority=1
        ),
        "langchain": SearchTemplate(
            code_type="langchain",
            template="examples of LangChain tools in GenericSuite",
            file_type_filter="py",
            priority=2
        )
    }


@pytest.fixture
def temp_repo_dir():
    """Create a temporary directory structure for document retrieval tests."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir) / "local_repo_files"
    repo_path.mkdir(parents=True)

    # Create test files
    (repo_path / "test.py").write_text("# Test Python file\nprint('hello')")
    (repo_path / "config.json").write_text('{"test": "value"}')
    (repo_path / "README.md").write_text("# Test README")

    # Create subdirectory
    subdir = repo_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("Nested file content")

    # Create binary file
    (repo_path / "binary.bin").write_bytes(b'\x00\x01\x02\x03')

    yield str(repo_path)

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_document_content():
    """Create sample DocumentContent for testing."""
    return DocumentContent(
        path="test/sample.py",
        content="# Sample Python code\nprint('test')",
        file_type="py",
        size=35,
        last_modified=None,
        metadata={"encoding": "utf-8", "lines": 2}
    )


@pytest.fixture
def sample_document_metadata():
    """Create sample DocumentMetadata for testing."""
    return DocumentMetadata(
        path="test/sample.py",
        file_type="py",
        size=35,
        last_modified=None,
        exists=True
    )


@pytest.fixture
def mock_template_config_file(tmp_path):
    """Create a temporary template configuration file."""
    config_data = {
        "templates": {
            "json": {
                "template": "JSON configuration examples",
                "file_type_filter": "json",
                "priority": 1
            },
            "python": {
                "template": "Python tool examples",
                "file_type_filter": "py",
                "priority": 1
            }
        }
    }

    config_file = tmp_path / "test_templates.json"
    config_file.write_text(json.dumps(config_data, indent=2))
    return str(config_file)


@pytest.fixture
def mock_invalid_template_config_file(tmp_path):
    """Create an invalid template configuration file for error testing."""
    config_file = tmp_path / "invalid_templates.json"
    config_file.write_text('{"invalid": json}')  # Invalid JSON
    return str(config_file)
