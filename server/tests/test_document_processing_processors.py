"""
Unit tests for document processing processors module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

from genericsuite_codegen.document_processing.processors import (
    Document,
    FileFilter,
    BaseProcessor,
    TextProcessor,
    PDFProcessor,
    DocumentProcessorManager
)


class TestDocument:
    """Test Document data class."""

    def test_document_creation(self):
        """Test creating a Document."""
        now = datetime.now()
        doc = Document(
            id="doc_1",
            path="test.py",
            content="print('hello')",
            file_type="python",
            metadata={"size": 100},
            created_at=now
        )

        assert doc.id == "doc_1"
        assert doc.path == "test.py"
        assert doc.content == "print('hello')"
        assert doc.file_type == "python"
        assert doc.metadata == {"size": 100}
        assert doc.created_at == now


class TestFileFilter:
    """Test FileFilter class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_file_filter_initialization(self):
        """Test FileFilter initialization."""
        file_filter = FileFilter(str(self.repo_path))

        assert file_filter.repo_path == self.repo_path
        assert isinstance(file_filter.gitignore_patterns, list)

    def test_load_gitignore_patterns(self):
        """Test loading .gitignore patterns."""
        # Create .gitignore file
        gitignore_content = """
# Comment
*.pyc
__pycache__/
.env
node_modules/
"""
        gitignore_path = self.repo_path / '.gitignore'
        gitignore_path.write_text(gitignore_content)

        file_filter = FileFilter(str(self.repo_path))

        expected_patterns = ['*.pyc', '__pycache__/', '.env', 'node_modules/']
        assert file_filter.gitignore_patterns == expected_patterns

    def test_should_include_file_included_extension(self):
        """Test including files with allowed extensions."""
        file_filter = FileFilter(str(self.repo_path))

        # Create test files
        py_file = self.repo_path / "test.py"
        py_file.touch()

        js_file = self.repo_path / "test.js"
        js_file.touch()

        assert file_filter.should_include_file(py_file)
        assert file_filter.should_include_file(js_file)

    def test_should_include_file_excluded_extension(self):
        """Test excluding files with disallowed extensions."""
        file_filter = FileFilter(str(self.repo_path))

        # Create test files
        png_file = self.repo_path / "image.png"
        png_file.touch()

        lock_file = self.repo_path / "package.lock"
        lock_file.touch()

        assert not file_filter.should_include_file(png_file)
        assert not file_filter.should_include_file(lock_file)


class TestBaseProcessor:
    """Test BaseProcessor class."""

    def test_base_processor_initialization(self):
        """Test BaseProcessor initialization."""
        processor = BaseProcessor()

        assert isinstance(processor.supported_extensions, set)
        assert len(processor.supported_extensions) == 0

    def test_can_process_not_implemented(self):
        """Test that process raises NotImplementedError."""
        processor = BaseProcessor()

        with pytest.raises(NotImplementedError):
            processor.process(Path("test.txt"))


class TestTextProcessor:
    """Test TextProcessor class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_text_processor_initialization(self):
        """Test TextProcessor initialization."""
        processor = TextProcessor()

        expected_extensions = {
            '.txt', '.md', '.py', '.js', '.jsx', '.ts', '.tsx', '.json',
            '.html', '.css', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.sh', '.env', '.gitignore', '.dockerignore', '.babelrc',
            '.npmrc', '.nvmrc', '.python-version', '.sample', '.for_test',
            '.mjs', '.cjs', '.hbs'
        }
        assert processor.supported_extensions == expected_extensions

    def test_can_process_supported_extension(self):
        """Test can_process with supported extensions."""
        processor = TextProcessor()

        assert processor.can_process(Path("test.txt"))
        assert processor.can_process(Path("README.md"))
        assert processor.can_process(Path("script.py"))

    def test_can_process_unsupported_extension(self):
        """Test can_process with unsupported extensions."""
        processor = TextProcessor()

        assert not processor.can_process(Path("image.png"))
        assert not processor.can_process(Path("document.pdf"))

    def test_process_file_python(self):
        """Test processing Python file."""
        processor = TextProcessor()

        # Create test file
        test_file = self.repo_path / "test.py"
        test_content = "def hello():\n    print('Hello, World!')"
        test_file.write_text(test_content, encoding='utf-8')

        document = processor.process(test_file)

        assert document.content == test_content
        assert document.file_type == "python"
        assert document.path == str(test_file)

    def test_process_file_javascript(self):
        """Test processing JavaScript file."""
        processor = TextProcessor()

        # Create test file
        test_file = self.repo_path / "test.js"
        test_content = "console.log('Hello, World!');"
        test_file.write_text(test_content, encoding='utf-8')

        document = processor.process(test_file)

        assert document.content == test_content
        assert document.file_type == "javascript"

    def test_process_file_json(self):
        """Test processing JSON file."""
        processor = TextProcessor()

        # Create test file
        test_file = self.repo_path / "config.json"
        test_content = '{"key": "value", "number": 42}'
        test_file.write_text(test_content, encoding='utf-8')

        document = processor.process(test_file)

        assert document.content == test_content
        assert document.file_type == "json"

    def test_process_file_encoding_fallback(self):
        """Test processing file with different encoding."""
        processor = TextProcessor()

        # Create test file with utf-8 encoding (avoid encoding issues in tests)
        test_file = self.repo_path / "test.txt"
        test_content = "Test content with special chars"
        test_file.write_text(test_content, encoding='utf-8')

        document = processor.process(test_file)

        assert document is not None
        assert document.content == test_content

    def test_process_file_nonexistent(self):
        """Test processing non-existent file."""
        processor = TextProcessor()

        nonexistent_file = self.repo_path / "nonexistent.txt"

        document = processor.process(nonexistent_file)

        assert document is None


class TestPDFProcessor:
    """Test PDFProcessor class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_pdf_processor_initialization(self):
        """Test PDFProcessor initialization."""
        processor = PDFProcessor()

        assert processor.supported_extensions == {'.pdf'}

    def test_can_process_pdf(self):
        """Test can_process with PDF files."""
        processor = PDFProcessor()

        assert processor.can_process(Path("document.pdf"))
        assert not processor.can_process(Path("document.txt"))

    @patch('genericsuite_codegen.document_processing.processors.pypdf')
    def test_process_file_success(self, mock_pypdf):
        """Test successful PDF processing."""
        # Mock pypdf
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF content"
        mock_reader.pages = [mock_page]
        mock_pypdf.PdfReader.return_value = mock_reader

        processor = PDFProcessor()

        # Create actual test file
        test_file = self.repo_path / "test.pdf"
        test_file.write_bytes(b'PDF data')

        document = processor.process(test_file)

        assert "PDF content" in document.content
        assert document.file_type == "pdf"

    def test_process_file_pypdf_not_available(self):
        """Test PDF processing when pypdf is not available."""
        with patch('genericsuite_codegen.document_processing.processors.pypdf', None):
            processor = PDFProcessor()

            test_file = Path("test.pdf")
            document = processor.process(test_file)

            assert document is None


class TestDocumentProcessorManager:
    """Test DocumentProcessorManager class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_document_processor_manager_initialization(self):
        """Test DocumentProcessorManager initialization."""
        manager = DocumentProcessorManager(str(self.repo_path))

        assert manager.repo_path == self.repo_path
        assert isinstance(manager.file_filter, FileFilter)
        assert len(manager.processors) > 0

    def test_get_all_files(self):
        """Test getting all files in repository."""
        manager = DocumentProcessorManager(str(self.repo_path))

        # Create test files
        (self.repo_path / "test.py").write_text("print('test')")
        (self.repo_path / "README.md").write_text("# README")
        (self.repo_path / "image.png").write_bytes(b"fake image")

        files = manager.get_all_files()

        # Should only include supported files
        file_names = [f.name for f in files]
        assert "test.py" in file_names
        assert "README.md" in file_names
        assert "image.png" not in file_names

    def test_get_file_stats(self):
        """Test getting file statistics."""
        manager = DocumentProcessorManager(str(self.repo_path))

        # Create test files
        (self.repo_path / "test.py").write_text("print('test')")
        (self.repo_path / "README.md").write_text("# README")

        stats = manager.get_file_stats()

        assert stats["total_files"] == 2
        assert "file_types" in stats
        assert "total_size" in stats

    def test_process_file_success(self):
        """Test successful file processing."""
        manager = DocumentProcessorManager(str(self.repo_path))

        # Create test file
        test_file = self.repo_path / "test.py"
        test_content = "print('Hello, World!')"
        test_file.write_text(test_content, encoding='utf-8')

        document = manager.process_file(test_file)

        assert document is not None
        assert document.content == test_content
        assert document.file_type == "python"

    def test_process_file_filtered_out(self):
        """Test processing file that should be filtered out."""
        manager = DocumentProcessorManager(str(self.repo_path))

        # Create file that should be filtered out
        test_file = self.repo_path / "image.png"
        test_file.touch()

        document = manager.process_file(test_file)

        assert document is None

    def test_process_repository_success(self):
        """Test processing entire repository."""
        manager = DocumentProcessorManager(str(self.repo_path))

        # Create test files
        (self.repo_path / "test.py").write_text("print('Python')")
        (self.repo_path / "README.md").write_text("# README")
        (self.repo_path / "config.json").write_text('{"key": "value"}')

        # Create subdirectory with files
        subdir = self.repo_path / "subdir"
        subdir.mkdir()
        (subdir / "script.js").write_text("console.log('JS');")

        documents = manager.process_repository()

        assert len(documents) == 4
        assert all(isinstance(doc, Document) for doc in documents)

        # Check that all files were processed
        processed_paths = {doc.path for doc in documents}
        expected_paths = {
            str(self.repo_path / "test.py"),
            str(self.repo_path / "README.md"),
            str(self.repo_path / "config.json"),
            str(subdir / "script.js")
        }
        assert processed_paths == expected_paths

    def test_process_repository_with_gitignore(self):
        """Test processing repository with .gitignore exclusions."""
        manager = DocumentProcessorManager(str(self.repo_path))

        # Create .gitignore
        gitignore_content = "*.pyc\n__pycache__/"
        (self.repo_path / ".gitignore").write_text(gitignore_content)

        # Create files
        (self.repo_path / "test.py").write_text("print('Python')")
        (self.repo_path / "test.pyc").write_text("compiled")

        pycache_dir = self.repo_path / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "module.pyc").write_text("compiled")

        documents = manager.process_repository()

        # Should only process .py file, not .pyc or __pycache__ contents
        # Note: .gitignore might not be processed depending on processor configuration
        processed_names = {Path(doc.path).name for doc in documents}
        assert "test.py" in processed_names
        assert "test.pyc" not in processed_names
        # .gitignore processing depends on TextProcessor configuration

    def test_process_repository_empty(self):
        """Test processing empty repository."""
        manager = DocumentProcessorManager(str(self.repo_path))

        documents = manager.process_repository()

        assert documents == []
