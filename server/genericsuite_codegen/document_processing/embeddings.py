"""
Embedding generation system with support for multiple providers.

This module provides embedding generation using OpenAI and HuggingFace models
with configurable model selection and dimension validation.
"""

import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
from abc import ABC, abstractmethod

try:
    import openai
except ImportError:
    openai = None

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """Represents a document chunk with its embedding."""
    chunk_id: str
    document_id: str
    content: str
    embedding: List[float]
    embedding_model: str
    chunk_index: int
    metadata: Dict[str, Any]
    created_at: datetime


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        pass
    
    @abstractmethod
    def validate_text_length(self, text: str) -> bool:
        """Validate if text length is within model limits."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    # Model configurations
    MODEL_CONFIGS = {
        'text-embedding-3-small': {
            'dimension': 1536,
            'max_tokens': 8192,
            'cost_per_1k_tokens': 0.00002
        },
        'text-embedding-3-large': {
            'dimension': 3072,
            'max_tokens': 8192,
            'cost_per_1k_tokens': 0.00013
        },
        'text-embedding-ada-002': {
            'dimension': 1536,
            'max_tokens': 8192,
            'cost_per_1k_tokens': 0.0001
        }
    }
    
    def __init__(self, 
                 model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL for OpenAI API
        """
        if openai is None:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        self.model = model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL')
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        client_kwargs = {'api_key': self.api_key}
        if self.base_url:
            client_kwargs['base_url'] = self.base_url
        
        self.client = openai.OpenAI(**client_kwargs)
        
        # Validate model
        if model not in self.MODEL_CONFIGS:
            logger.warning(f"Unknown OpenAI model {model}. Using default configuration.")
            self.model_config = {
                'dimension': 1536,
                'max_tokens': 8192,
                'cost_per_1k_tokens': 0.0001
            }
        else:
            self.model_config = self.MODEL_CONFIGS[model]
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self.validate_text_length(text):
            raise ValueError(f"Text too long for model {self.model}")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Validate all texts
        for text in texts:
            if not self.validate_text_length(text):
                raise ValueError(f"Text too long for model {self.model}")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model_config['dimension']
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model
    
    def validate_text_length(self, text: str) -> bool:
        """Validate text length against model limits."""
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = len(text) / 4
        return estimated_tokens <= self.model_config['max_tokens']


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider using sentence-transformers."""
    
    # Popular model configurations
    MODEL_CONFIGS = {
        'thenlper/gte-small': {
            'dimension': 384,
            'max_tokens': 512
        },
        'thenlper/gte-base': {
            'dimension': 768,
            'max_tokens': 512
        },
        'thenlper/gte-large': {
            'dimension': 1024,
            'max_tokens': 512
        },
        'sentence-transformers/all-MiniLM-L6-v2': {
            'dimension': 384,
            'max_tokens': 256
        },
        'sentence-transformers/all-mpnet-base-v2': {
            'dimension': 768,
            'max_tokens': 384
        },
        'BAAI/bge-small-en-v1.5': {
            'dimension': 384,
            'max_tokens': 512
        },
        'BAAI/bge-base-en-v1.5': {
            'dimension': 768,
            'max_tokens': 512
        },
        'BAAI/bge-large-en-v1.5': {
            'dimension': 1024,
            'max_tokens': 512
        }
    }
    
    def __init__(self, 
                 model: str = "thenlper/gte-small",
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None):
        """
        Initialize HuggingFace embedding provider.
        
        Args:
            model: HuggingFace model name
            device: Device to run model on ('cpu', 'cuda', 'mps', etc.)
            cache_folder: Custom cache folder for models
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers library not installed. Install with: pip install sentence-transformers")
        
        self.model_name = model
        
        # Determine device
        if device is None:
            if torch and torch.cuda.is_available():
                device = 'cuda'
            elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Load model
        try:
            model_kwargs = {}
            if cache_folder:
                model_kwargs['cache_folder'] = cache_folder
            
            self.model = SentenceTransformer(model, device=device, **model_kwargs)
            logger.info(f"Loaded HuggingFace model {model} on device {device}")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {model}: {e}")
            raise
        
        # Get model configuration
        if model in self.MODEL_CONFIGS:
            self.model_config = self.MODEL_CONFIGS[model]
        else:
            # Try to infer configuration
            logger.warning(f"Unknown HuggingFace model {model}. Inferring configuration.")
            try:
                # Generate a test embedding to get dimension
                test_embedding = self.model.encode("test")
                self.model_config = {
                    'dimension': len(test_embedding),
                    'max_tokens': 512  # Conservative default
                }
            except Exception as e:
                logger.error(f"Could not infer model configuration: {e}")
                self.model_config = {
                    'dimension': 384,
                    'max_tokens': 512
                }
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self.validate_text_length(text):
            logger.warning(f"Text may be too long for model {self.model_name}")
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating HuggingFace embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False, batch_size=32)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error generating HuggingFace embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model_config['dimension']
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name
    
    def validate_text_length(self, text: str) -> bool:
        """Validate text length against model limits."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        estimated_tokens = len(text) / 4
        return estimated_tokens <= self.model_config['max_tokens']


class EmbeddingGenerator:
    """Main embedding generator that manages providers and chunking."""
    
    def __init__(self, 
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 **provider_kwargs):
        """
        Initialize embedding generator.
        
        Args:
            provider: Provider name ('openai' or 'huggingface')
            model: Model name for the provider
            **provider_kwargs: Additional arguments for the provider
        """
        # Get configuration from environment if not provided
        if provider is None:
            provider = os.getenv('EMBEDDINGS_PROVIDER', 'huggingface').lower()
        
        if model is None:
            if provider == 'openai':
                model = os.getenv('EMBEDDINGS_MODEL', 'text-embedding-3-small')
            else:
                model = os.getenv('EMBEDDINGS_MODEL', 'thenlper/gte-small')
        
        self.provider_name = provider
        self.model_name = model
        
        # Initialize provider
        if provider == 'openai':
            self.provider = OpenAIEmbeddingProvider(model=model, **provider_kwargs)
        elif provider == 'huggingface':
            self.provider = HuggingFaceEmbeddingProvider(model=model, **provider_kwargs)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
        
        logger.info(f"Initialized {provider} embedding provider with model {model}")
    
    def generate_embeddings_for_chunks(self, chunks) -> List[EmbeddedChunk]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of EmbeddedChunk objects
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.provider_name}")
        
        # Extract texts from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batches for efficiency
        batch_size = 100 if self.provider_name == 'openai' else 32
        embedded_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Generate embeddings for this batch
                batch_embeddings = self.provider.generate_embeddings(batch_texts)
                
                # Create EmbeddedChunk objects
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    embedded_chunk = EmbeddedChunk(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        embedding=embedding,
                        embedding_model=self.provider.get_model_name(),
                        chunk_index=chunk.chunk_index,
                        metadata=chunk.metadata.copy(),
                        created_at=datetime.now()
                    )
                    
                    # Add embedding metadata
                    embedded_chunk.metadata.update({
                        'embedding_provider': self.provider_name,
                        'embedding_model': self.provider.get_model_name(),
                        'embedding_dimension': len(embedding),
                        'embedding_created_at': datetime.now().isoformat()
                    })
                    
                    embedded_chunks.append(embedded_chunk)
                
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch starting at index {i}: {e}")
                # Continue with next batch instead of failing completely
                continue
        
        logger.info(f"Successfully generated embeddings for {len(embedded_chunks)}/{len(chunks)} chunks")
        return embedded_chunks
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        return self.provider.generate_embedding(query)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.provider.get_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'provider': self.provider_name,
            'model': self.model_name,
            'dimension': self.provider.get_embedding_dimension(),
            'max_tokens': getattr(self.provider, 'model_config', {}).get('max_tokens', 'unknown')
        }
    
    def validate_compatibility(self, existing_dimension: int) -> bool:
        """Validate if current model is compatible with existing embeddings."""
        current_dimension = self.provider.get_embedding_dimension()
        return current_dimension == existing_dimension


# Convenience functions
def create_embedding_generator(provider: Optional[str] = None, 
                             model: Optional[str] = None,
                             **kwargs) -> EmbeddingGenerator:
    """Create an embedding generator with the specified configuration."""
    return EmbeddingGenerator(provider=provider, model=model, **kwargs)


def get_available_providers() -> Dict[str, List[str]]:
    """Get available embedding providers and their models."""
    providers = {}
    
    if openai is not None:
        providers['openai'] = list(OpenAIEmbeddingProvider.MODEL_CONFIGS.keys())
    
    if SentenceTransformer is not None:
        providers['huggingface'] = list(HuggingFaceEmbeddingProvider.MODEL_CONFIGS.keys())
    
    return providers


def validate_embedding_config(provider: str, model: str) -> Dict[str, Any]:
    """Validate embedding configuration and return model info."""
    try:
        generator = EmbeddingGenerator(provider=provider, model=model)
        return {
            'valid': True,
            'info': generator.get_model_info()
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


async def generate_embeddings_async(chunks, provider: Optional[str] = None, 
                                  model: Optional[str] = None) -> List[EmbeddedChunk]:
    """Asynchronously generate embeddings for chunks."""
    # For now, run synchronously in a thread pool
    # In the future, this could be enhanced with true async providers
    loop = asyncio.get_event_loop()
    
    def _generate():
        generator = EmbeddingGenerator(provider=provider, model=model)
        return generator.generate_embeddings_for_chunks(chunks)
    
    return await loop.run_in_executor(None, _generate)
