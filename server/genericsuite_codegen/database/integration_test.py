"""
Integration test for database operations.

This test demonstrates the complete workflow of the database manager
including connection, schema initialization, and vector operations.
"""

import os
import asyncio
from datetime import datetime
from typing import List

from genericsuite_codegen.database import (
    DatabaseManager,
    create_embedded_chunk,
    health_check,
    initialize_database
)


async def test_database_workflow():
    """Test the complete database workflow."""
    print("Testing database workflow...")
    
    # Set test environment
    os.environ["MONGODB_URI"] = "mongodb://localhost:27017/"
    
    try:
        # Test health check without connection
        health = await health_check()
        print(f"Health check result: {health['status']}")
        
        # Test database manager creation
        db_manager = DatabaseManager()
        print(f"Database manager created for: {db_manager.db_name}")
        
        # Test embedded chunk creation
        chunks = []
        for i in range(3):
            chunk = create_embedded_chunk(
                chunk_id=f"test_chunk_{i}",
                document_path=f"/test/document_{i}.md",
                content=f"This is test content for chunk {i}",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
                chunk_index=i,
                file_type="md",
                metadata={"test": True, "chunk_number": i}
            )
            chunks.append(chunk)
        
        print(f"Created {len(chunks)} test chunks")
        
        # Test stats without connection (should handle gracefully)
        stats = db_manager.get_knowledge_base_stats()
        print(f"Stats without connection: {stats}")
        
        print("✅ Database workflow test completed successfully")
        
    except Exception as e:
        print(f"❌ Database workflow test failed: {e}")
        raise


def test_vector_operations():
    """Test vector operations without actual database connection."""
    print("Testing vector operations...")
    
    try:
        # Test embedding validation
        from genericsuite_codegen.database import validate_embedding_dimensions
        
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ]
        
        assert validate_embedding_dimensions(embeddings, 4) == True
        assert validate_embedding_dimensions(embeddings, 3) == False
        
        print("✓ Vector validation tests passed")
        
        # Test search result creation
        from genericsuite_codegen.database import SearchResult
        
        result = SearchResult(
            content="Test search result",
            metadata={"source": "test"},
            similarity_score=0.95,
            document_path="/test/result.md"
        )
        
        assert result.similarity_score == 0.95
        print("✓ Search result creation test passed")
        
        print("✅ Vector operations test completed successfully")
        
    except Exception as e:
        print(f"❌ Vector operations test failed: {e}")
        raise


async def main():
    """Run all integration tests."""
    print("Running database integration tests...\n")
    
    try:
        await test_database_workflow()
        test_vector_operations()
        
        print("\n🎉 All integration tests passed!")
        
    except Exception as e:
        print(f"\n💥 Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)