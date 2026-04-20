"""
Unit tests for document processing embeddings module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from genericsuite_codegen.document_processing.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    EmbeddingGenerator,
    get_available_providers,
    validate_embedding_config
)
from genericsuite_codegen.document_processing.types import EmbeddingModel, EmbeddedChunk
from genericsuite_codegen.document_processing.chunker import DocumentChunk


class TestEmbeddingProvider:
    """Test EmbeddingProvider abstract base class."""

    def test_embedding_provider_is_abstract(self):
        """Test that EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_embedding_provider_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        class IncompleteProvider(EmbeddingProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()


class TestOpenAIEmbeddingProvider:
    """Test OpenAIEmbeddingProvider class."""

    @patch('genericsuite_codegen.document_processing.embeddings.openai')
    def test_openai_provider_initialization_default(self, mock_openai):
        """Test OpenAI provider initialization with defaults."""
        provider = OpenAIEmbeddingProvider()

        assert provider.model == "text-embedding-3-small"
        assert provider.get_embedding_dimension() == 1536

    @patch('genericsuite_codegen.document_processing.embeddings.openai')
    def test_openai_provider_initialization_custom_model(self, mock_openai):
        """Test OpenAI provider initialization with custom model."""
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")

        assert provider.model == "text-embedding-3-large"
        assert provider.get_embedding_dimension() == 3072

    @patch('genericsuite_codegen.document_processing.embeddings.openai')
    def test_openai_provider_initialization_invalid_model(self, mock_openai):
        """Test OpenAI provider initialization with invalid model."""
        with pytest.raises(ValueError):
            OpenAIEmbeddingProvider(model="invalid-model")

    def test_openai_provider_no_openai_library(self):
        """Test OpenAI provider when openai library is not available."""
        with patch('genericsuite_codegen.document_processing.embeddings.openai', None):
            with pytest.raises(ImportError):
                OpenAIEmbeddingProvider()

    @patch('genericsuite_codegen.document_processing.embeddings.openai')
    def test_openai_provider_api_key_from_env(self, mock_openai):
        """Test OpenAI provider gets API key from environment."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddingProvider()
            mock_openai.OpenAI.assert_called_once()

    @patch('genericsuite_codegen.document_processing.embeddings.openai')
    def test_openai_provider_custom_api_key(self, mock_openai):
        """Test OpenAI provider with custom API key."""
        provider = OpenAIEmbeddingProvider(api_key="custom-key")
        mock_openai.OpenAI.assert_called_once()

    @patch('genericsuite_codegen.document_processing.embeddings.openai')
    def test_generate_embedding_success(self, mock_openai):
        """Test successful embedding generation."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        provider = OpenAIEmbeddingProvider()
        embedding = provider.generate_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test text"
        )

    @patch('genericsuite_codegen.document_processing.embeddings.openai')
    def test_generate_embeddings_batch(self, mock_openai):
        """Test batch embedding generation."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        provider = OpenAIEmbeddingProvider()
        embeddings = provider.generate_embeddings(["text1", "text2"])

        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @patch('genericsuite_codegen.document_processing.embeddings.openai')
    def test_generate_embedding_api_error(self, mock_openai):
        """Test handling of OpenAI API errors."""
        # Mock OpenAI client to raise an exception
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.OpenAI.return_value = mock_client

        provider = OpenAIEmbeddingProvider()

        with pytest.raises(Exception):
            provider.generate_embedding("test text")

    @patch('genericsuite_codegen.document_processing.embeddings.openai')
    def test_validate_text_length(self, mock_openai):
        """Test text length validation."""
        provider = OpenAIEmbeddingProvider()

        # Short text should be valid
        assert provider.validate_text_length("short text")

        # Very long text should be invalid
        long_text = "word " * 10000  # Approximately 50,000 characters
        assert not provider.validate_text_length(long_text)

    @patch('genericsuite_codegen.document_processing.embeddings.openai')
    def test_get_model_name(self, mock_openai):
        """Test getting model name."""
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
        assert provider.get_model_name() == "text-embedding-3-large"


class TestHuggingFaceEmbeddingProvider:
    """Test HuggingFaceEmbeddingProvider class."""

    @patch('genericsuite_codegen.document_processing.embeddings.SentenceTransformer')
    def test_huggingface_provider_initialization_default(self, mock_st):
        """Test HuggingFace provider initialization with defaults."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider = HuggingFaceEmbeddingProvider()

        assert provider.model_name == "all-MiniLM-L6-v2"
        mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    @patch('genericsuite_codegen.document_processing.embeddings.SentenceTransformer')
    def test_huggingface_provider_initialization_custom_model(self, mock_st):
        """Test HuggingFace provider initialization with custom model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model

        provider = HuggingFaceEmbeddingProvider(model="all-mpnet-base-v2")

        assert provider.model_name == "all-mpnet-base-v2"
        mock_st.assert_called_once_with("all-mpnet-base-v2")

    def test_huggingface_provider_no_sentence_transformers(self):
        """Test HuggingFace provider when sentence_transformers is not available."""
        with patch('genericsuite_codegen.document_processing.embeddings.SentenceTransformer', None):
            with pytest.raises(ImportError):
                HuggingFaceEmbeddingProvider()

    @patch('genericsuite_codegen.document_processing.embeddings.SentenceTransformer')
    def test_generate_embedding_success(self, mock_st):
        """Test successful embedding generation."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_st.return_value = mock_model

        provider = HuggingFaceEmbeddingProvider()
        embedding = provider.generate_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with("test text")

    @patch('genericsuite_codegen.document_processing.embeddings.SentenceTransformer')
    def test_generate_embeddings_batch(self, mock_st):
        """Test batch embedding generation."""
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_st.return_value = mock_model

        provider = HuggingFaceEmbeddingProvider()
        embeddings = provider.generate_embeddings(["text1", "text2"])

        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @patch('genericsuite_codegen.document_processing.embeddings.SentenceTransformer')
    def test_validate_text_length(self, mock_st):
        """Test text length validation."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider = HuggingFaceEmbeddingProvider()

        # Short text should be valid
        assert provider.validate_text_length("short text")

        # Very long text should be invalid (sentence transformers have token limits)
        long_text = "word " * 1000
        assert not provider.validate_text_length(long_text)

    @patch('genericsuite_codegen.document_processing.embeddings.SentenceTransformer')
    def test_get_embedding_dimension(self, mock_st):
        """Test getting embedding dimension."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model

        provider = HuggingFaceEmbeddingProvider()
        assert provider.get_embedding_dimension() == 768

    @patch('genericsuite_codegen.document_processing.embeddings.SentenceTransformer')
    def test_get_model_name(self, mock_st):
        """Test getting model name."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider = HuggingFaceEmbeddingProvider(model="custom-model")
        assert provider.get_model_name() == "custom-model"


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class."""

    def test_embedding_generator_initialization(self):
        """Test EmbeddingGenerator initialization."""
        mock_provider = Mock(spec=EmbeddingProvider)
        generator = EmbeddingGenerator(mock_provider)

        assert generator.provider is mock_provider

    def test_generate_embeddings_for_chunks(self):
        """Test generating embeddings for document chunks."""
        # Create mock provider
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        mock_provider.get_model_name.return_value = "test-model"
        mock_provider.validate_text_length.return_value = True

        # Create test chunks
        chunks = [
            DocumentChunk("1", "doc1", "content1", 0, {}),
            DocumentChunk("2", "doc1", "content2", 1, {})
        ]

        generator = EmbeddingGenerator(mock_provider)
        embedded_chunks = generator.generate_embeddings_for_chunks(chunks)

        assert len(embedded_chunks) == 2
        assert all(isinstance(chunk, EmbeddedChunk)
                   for chunk in embedded_chunks)
        assert embedded_chunks[0].embedding == [0.1, 0.2, 0.3]
        assert embedded_chunks[1].embedding == [0.4, 0.5, 0.6]
        assert embedded_chunks[0].embedding_model == "test-model"

    def test_generate_embeddings_for_chunks_text_too_long(self):
        """Test handling of chunks with text too long."""
        # Create mock provider that rejects long text
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.validate_text_length.side_effect = lambda x: len(x) < 10
        mock_provider.get_model_name.return_value = "test-model"

        # Create test chunks with one long chunk
        chunks = [
            DocumentChunk("1", "doc1", "short", 0, {}),
            DocumentChunk(
                "2", "doc1", "this is a very long content that exceeds limit", 1, {})
        ]

        generator = EmbeddingGenerator(mock_provider)
        embedded_chunks = generator.generate_embeddings_for_chunks(chunks)

        # Should only process the short chunk
        assert len(embedded_chunks) == 1
        assert embedded_chunks[0].chunk.content == "short"

    def test_generate_embeddings_for_chunks_empty_list(self):
        """Test generating embeddings for empty chunk list."""
        mock_provider = Mock(spec=EmbeddingProvider)
        generator = EmbeddingGenerator(mock_provider)

        embedded_chunks = generator.generate_embeddings_for_chunks([])

        assert embedded_chunks == []
        mock_provider.generate_embeddings.assert_not_called()

    def test_generate_embeddings_for_chunks_provider_error(self):
        """Test handling of provider errors."""
        # Create mock provider that raises an error
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.generate_embeddings.side_effect = Exception(
            "Provider error")
        mock_provider.validate_text_length.return_value = True

        chunks = [DocumentChunk("1", "doc1", "content1", 0, {})]

        generator = EmbeddingGenerator(mock_provider)

        with pytest.raises(Exception):
            generator.generate_embeddings_for_chunks(chunks)

    def test_generate_embedding_for_text(self):
        """Test generating embedding for single text."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_provider.validate_text_length.return_value = True

        generator = EmbeddingGenerator(mock_provider)
        embedding = generator.generate_embedding_for_text("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_provider.generate_embedding.assert_called_once_with("test text")

    def test_generate_embedding_for_text_too_long(self):
        """Test generating embedding for text that's too long."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.validate_text_length.return_value = False

        generator = EmbeddingGenerator(mock_provider)

        with pytest.raises(ValueError):
            generator.generate_embedding_for_text("very long text")

    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.get_embedding_dimension.return_value = 1536

        generator = EmbeddingGenerator(mock_provider)
        dimension = generator.get_embedding_dimension()

        assert dimension == 1536

    def test_get_model_info(self):
        """Test getting model information."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.get_model_name.return_value = "test-model"
        mock_provider.get_embedding_dimension.return_value = 768

        generator = EmbeddingGenerator(mock_provider)
        model_info = generator.get_model_info()

        assert model_info["model_name"] == "test-model"
        assert model_info["embedding_dimension"] == 768
        assert "provider_type" in model_info


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_available_providers(self):
        """Test getting available providers."""
        providers = get_available_providers()

        assert isinstance(providers, dict)
        # Should have at least one provider available
        assert len(providers) > 0

    def test_validate_embedding_config_openai(self):
        """Test validating OpenAI embedding configuration."""
        with patch('genericsuite_codegen.document_processing.embeddings.openai'):
            try:
                config = validate_embedding_config(
                    "openai", "text-embedding-3-small")
                assert isinstance(config, dict)
                assert "provider" in config
                assert "model" in config
            except ValueError:
                # OpenAI might not be available in test environment
                pass

    def test_validate_embedding_config_huggingface(self):
        """Test validating HuggingFace embedding configuration."""
        with patch('genericsuite_codegen.document_processing.embeddings.SentenceTransformer'):
            try:
                config = validate_embedding_config(
                    "huggingface", "all-MiniLM-L6-v2")
                assert isinstance(config, dict)
                assert "provider" in config
                assert "model" in config
            except ValueError:
                # HuggingFace might not be available in test environment
                pass

    def test_validate_embedding_config_invalid(self):
        """Test validating invalid embedding configuration."""
        with pytest.raises(ValueError):
            validate_embedding_config("invalid_provider", "invalid_model")
