#!/usr/bin/env python3
"""
Debug script to test knowledge base search functionality.

This script will help diagnose why the system isn't finding things in the documentation.
"""

import os
import sys
import asyncio
import logging
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Add the server directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from genericsuite_codegen.database.setup import get_database_manager, health_check
from genericsuite_codegen.document_processing.embeddings import create_embedding_generator
from genericsuite_codegen.agent.tools import KnowledgeBaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_database_connection():
    """Test database connection and basic stats."""
    print("=" * 60)
    print("TESTING DATABASE CONNECTION")
    print("=" * 60)
    
    try:
        # Test health check
        health = await health_check()
        print(f"Database health: {health}")
        
        # Get database manager
        db_manager = get_database_manager()
        
        # Test connection
        if db_manager.database is None:
            db_manager.connect()
        
        # Get knowledge base stats
        stats = db_manager.get_knowledge_base_stats()
        print(f"Knowledge base stats: {stats}")
        
        # Get document count
        doc_count = db_manager.get_document_count()
        print(f"Total documents in knowledge base: {doc_count}")
        
        return db_manager, doc_count > 0
        
    except Exception as e:
        print(f"Database connection test failed: {e}")
        return None, False


def test_embedding_generation():
    """Test embedding generation."""
    print("\n" + "=" * 60)
    print("TESTING EMBEDDING GENERATION")
    print("=" * 60)
    
    try:
        # Create embedding generator
        embedding_gen = create_embedding_generator()
        print(f"Embedding provider: {embedding_gen.provider_name}")
        print(f"Model: {embedding_gen.model_name}")
        print(f"Dimension: {embedding_gen.get_embedding_dimension()}")
        
        # Test query embedding
        test_query = "How do I create a GenericSuite table configuration?"
        print(f"\nTesting query: '{test_query}'")
        
        query_embedding = embedding_gen.generate_query_embedding(test_query)
        print(f"Generated embedding dimension: {len(query_embedding)}")
        print(f"First 5 values: {query_embedding[:5]}")
        
        return embedding_gen, query_embedding
        
    except Exception as e:
        print(f"Embedding generation test failed: {e}")
        return None, None


async def test_vector_search(db_manager, embedding_gen, query_embedding):
    """Test vector search functionality."""
    print("\n" + "=" * 60)
    print("TESTING VECTOR SEARCH")
    print("=" * 60)
    
    try:
        # Test direct database search
        print("Testing direct database search...")
        search_results = db_manager.search_similar(
            query_embedding=query_embedding,
            limit=5
        )
        
        print(f"Direct search found {len(search_results)} results:")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. Score: {result.similarity_score:.4f}")
            print(f"      Path: {result.document_path}")
            print(f"      Content: {result.content[:100]}...")
            print()
        
        return len(search_results) > 0
        
    except Exception as e:
        print(f"Vector search test failed: {e}")
        return False


async def test_knowledge_base_tool():
    """Test the knowledge base tool."""
    print("\n" + "=" * 60)
    print("TESTING KNOWLEDGE BASE TOOL")
    print("=" * 60)
    
    try:
        # Create knowledge base tool
        kb_tool = KnowledgeBaseTool()
        
        # Test search
        test_query = "How do I create a GenericSuite table configuration?"
        print(f"Testing query: '{test_query}'")
        
        search_results = kb_tool.search(
            query=test_query,
            limit=5
        )
        
        print(f"Knowledge base tool found {len(search_results.results)} results:")
        print(f"Context summary: {search_results.context_summary}")
        
        for i, result in enumerate(search_results.results):
            print(f"  {i+1}. Score: {result.similarity_score:.4f}")
            print(f"      Path: {result.document_path}")
            print(f"      Type: {result.file_type}")
            print(f"      Content: {result.content[:100]}...")
            print()
        
        # Test context generation
        print("\nTesting context generation...")
        context, sources = kb_tool.get_context_for_generation(
            query=test_query,
            max_context_length=2000
        )
        
        print(f"Generated context length: {len(context)}")
        print(f"Sources: {sources}")
        print(f"Context preview: {context[:200]}...")
        
        return len(search_results.results) > 0
        
    except Exception as e:
        print(f"Knowledge base tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_sample_documents():
    """Check what documents are actually in the database."""
    print("\n" + "=" * 60)
    print("CHECKING SAMPLE DOCUMENTS")
    print("=" * 60)
    
    try:
        db_manager = get_database_manager()
        if db_manager.database is None:
            db_manager.connect()
        
        collection = db_manager.database[db_manager.knowledge_base_collection]
        
        # Get sample documents
        sample_docs = list(collection.find().limit(5))
        
        print(f"Found {len(sample_docs)} sample documents:")
        for i, doc in enumerate(sample_docs):
            print(f"  {i+1}. Path: {doc.get('path', 'unknown')}")
            print(f"      File type: {doc.get('file_type', 'unknown')}")
            print(f"      Content length: {len(doc.get('content', ''))}")
            print(f"      Embedding dimension: {len(doc.get('embedding', []))}")
            print(f"      Content preview: {doc.get('content', '')[:100]}...")
            print()
        
        # Check file types distribution
        pipeline = [
            {"$group": {"_id": "$file_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        file_types = list(collection.aggregate(pipeline))
        print(f"File types in database: {file_types}")
        
        return len(sample_docs) > 0
        
    except Exception as e:
        print(f"Sample documents check failed: {e}")
        return False


async def main():
    """Run all diagnostic tests."""
    print("GENERICSUITE CODEGEN - SEARCH DIAGNOSTIC")
    print("=" * 60)
    
    # Test 1: Database connection
    db_manager, has_docs = await test_database_connection()
    if not db_manager:
        print("❌ Database connection failed - cannot continue")
        return
    
    if not has_docs:
        print("⚠️  No documents found in knowledge base")
        await test_sample_documents()
        return
    
    # Test 2: Embedding generation
    embedding_gen, query_embedding = test_embedding_generation()
    if not embedding_gen or not query_embedding:
        print("❌ Embedding generation failed - cannot continue")
        return
    
    # Test 3: Vector search
    search_works = await test_vector_search(db_manager, embedding_gen, query_embedding)
    if not search_works:
        print("❌ Vector search failed")
    else:
        print("✅ Vector search working")
    
    # Test 4: Knowledge base tool
    kb_tool_works = await test_knowledge_base_tool()
    if not kb_tool_works:
        print("❌ Knowledge base tool failed")
    else:
        print("✅ Knowledge base tool working")
    
    # Test 5: Sample documents
    await test_sample_documents()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    if search_works and kb_tool_works:
        print("✅ All tests passed - search should be working")
    else:
        print("❌ Some tests failed - search may not work properly")


if __name__ == "__main__":
    asyncio.run(main())