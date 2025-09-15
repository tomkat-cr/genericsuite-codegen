#!/usr/bin/env python3
"""
Detailed vector search diagnostic to identify the search issue.
"""

import os
import sys
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Add the server directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from genericsuite_codegen.database.setup import get_database_manager
from genericsuite_codegen.document_processing.embeddings import create_embedding_generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vector_search_detailed():
    """Test vector search with detailed debugging."""
    print("=" * 60)
    print("DETAILED VECTOR SEARCH DIAGNOSTIC")
    print("=" * 60)
    
    try:
        # Get database manager and connect
        db_manager = get_database_manager()
        if db_manager.database is None:
            db_manager.connect()
        
        collection = db_manager.database[db_manager.knowledge_base_collection]
        
        # Create embedding generator
        embedding_gen = create_embedding_generator()
        
        # Generate query embedding
        test_query = "GenericSuite table configuration"
        print(f"Test query: '{test_query}'")
        
        query_embedding = embedding_gen.generate_query_embedding(test_query)
        print(f"Query embedding dimension: {len(query_embedding)}")
        print(f"Query embedding sample: {query_embedding[:5]}")
        
        # Check sample document embeddings
        print("\nChecking sample document embeddings...")
        sample_docs = list(collection.find().limit(3))
        
        for i, doc in enumerate(sample_docs):
            embedding = doc.get('embedding', [])
            print(f"  Doc {i+1}: {doc.get('path', 'unknown')}")
            print(f"    Embedding dimension: {len(embedding)}")
            print(f"    Embedding sample: {embedding[:5] if embedding else 'None'}")
            print(f"    Content preview: {doc.get('content', '')[:100]}...")
            print()
        
        # Test MongoDB Atlas vector search first
        print("Testing MongoDB Atlas vector search...")
        try:
            vector_search_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 50,
                        "limit": 5
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "path": 1,
                        "file_type": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            vector_results = list(collection.aggregate(vector_search_pipeline))
            print(f"Vector search results: {len(vector_results)}")
            
            for i, result in enumerate(vector_results):
                print(f"  {i+1}. Score: {result.get('score', 'N/A')}")
                print(f"      Path: {result.get('path', 'unknown')}")
                print(f"      Content: {result.get('content', '')[:100]}...")
                print()
                
        except Exception as e:
            print(f"Vector search failed: {e}")
            print("This is expected if MongoDB Atlas vector search is not available")
            
        # Test cosine similarity search (always run this)
        print("\nTesting cosine similarity search...")
        try:
                cosine_pipeline = [
                    {"$match": {}},
                    {
                        "$addFields": {
                            "similarity": {
                                "$let": {
                                    "vars": {
                                        "dot_product": {
                                            "$reduce": {
                                                "input": {"$range": [0, {"$size": "$embedding"}]},
                                                "initialValue": 0,
                                                "in": {
                                                    "$add": [
                                                        "$$value",
                                                        {
                                                            "$multiply": [
                                                                {"$arrayElemAt": ["$embedding", "$$this"]},
                                                                {"$arrayElemAt": [query_embedding, "$$this"]}
                                                            ]
                                                        }
                                                    ]
                                                }
                                            }
                                        },
                                        "query_magnitude": {
                                            "$sqrt": {
                                                "$reduce": {
                                                    "input": query_embedding,
                                                    "initialValue": 0,
                                                    "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                                }
                                            }
                                        },
                                        "doc_magnitude": {
                                            "$sqrt": {
                                                "$reduce": {
                                                    "input": "$embedding",
                                                    "initialValue": 0,
                                                    "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                                }
                                            }
                                        }
                                    },
                                    "in": {
                                        "$divide": [
                                            "$$dot_product",
                                            {"$multiply": ["$$query_magnitude", "$$doc_magnitude"]}
                                        ]
                                    }
                                }
                            }
                        }
                    },
                    {"$sort": {"similarity": -1}},
                    {"$limit": 5},
                    {
                        "$project": {
                            "content": 1,
                            "path": 1,
                            "file_type": 1,
                            "metadata": 1,
                            "similarity": 1
                        }
                    }
                ]
                
                cosine_results = list(collection.aggregate(cosine_pipeline))
                print(f"Cosine similarity results: {len(cosine_results)}")
                
                for i, result in enumerate(cosine_results):
                    print(f"  {i+1}. Similarity: {result.get('similarity', 'N/A'):.4f}")
                    print(f"      Path: {result.get('path', 'unknown')}")
                    print(f"      Content: {result.get('content', '')[:100]}...")
                    print()
                    
        except Exception as e:
            print(f"Cosine similarity search also failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test simple text search as fallback
        print("\nTesting simple text search...")
        try:
            text_results = list(collection.find(
                {"content": {"$regex": "table", "$options": "i"}}
            ).limit(5))
            
            print(f"Text search results: {len(text_results)}")
            for i, result in enumerate(text_results):
                print(f"  {i+1}. Path: {result.get('path', 'unknown')}")
                print(f"      Content: {result.get('content', '')[:100]}...")
                print()
                
        except Exception as e:
            print(f"Text search failed: {e}")
        
    except Exception as e:
        print(f"Detailed diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vector_search_detailed()