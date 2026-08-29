#!/usr/bin/env python3
"""
Test runner for enhanced search components.

This script runs all unit tests for the enhanced search functionality
and provides a summary of test results.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_tests():
    """Run all enhanced search tests."""
    print("=" * 60)
    print("Running Enhanced Search Component Tests")
    print("=" * 60)

    # Change to server directory
    server_dir = Path(__file__).parent
    os.chdir(server_dir)

    # Test files to run
    test_files = [
        "tests/test_enhanced_search_types.py",
        "tests/test_document_retrieval_tool.py",
        "tests/test_search_templates.py",
        "tests/test_context_determination.py",
        "tests/test_enhanced_search.py",
        "tests/test_enhanced_search_integration.py"
    ]

    # Run each test file
    total_passed = 0
    total_failed = 0
    failed_files = []

    for test_file in test_files:
        print(f"\n{'='*40}")
        print(f"Running: {test_file}")
        print(f"{'='*40}")

        try:
            # Run pytest for this specific file
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--no-header"
            ], capture_output=True, text=True, timeout=120)

            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            if result.returncode == 0:
                print(f"✅ {test_file} - PASSED")
                # Count passed tests from output
                passed_count = result.stdout.count(" PASSED")
                total_passed += passed_count
            else:
                print(f"❌ {test_file} - FAILED")
                failed_files.append(test_file)
                # Count failed tests from output
                failed_count = result.stdout.count(" FAILED")
                total_failed += failed_count

        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} - TIMEOUT")
            failed_files.append(test_file)
            total_failed += 1
        except Exception as e:
            print(f"💥 {test_file} - ERROR: {e}")
            failed_files.append(test_file)
            total_failed += 1

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(
        ("Success Rate: " +
         f"{(total_passed / (total_passed + total_failed) * 100):.1f}%")
        if (total_passed + total_failed) > 0 else "No tests run"
    )

    if failed_files:
        print("\nFailed Test Files:")
        for file in failed_files:
            print(f"  - {file}")
        else:
            print("\n🎉 All tests passed!")

        return len(failed_files) == 0


def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")

    try:
        import pytest
        print(f"✅ pytest {pytest.__version__}")
    except ImportError:
        print("❌ pytest not found. Install with: pip install pytest")
        return False

    try:
        import pytest_asyncio
        print(
            "✅ pytest-asyncio available. Version: "
            f"{pytest_asyncio.__version__}")
    except ImportError:
        print("❌ pytest-asyncio not found. Install with: pip install"
              " pytest-asyncio")
        return False

    # Check if the enhanced search modules can be imported
    try:
        from genericsuite_codegen.agent.enhanced_search_types \
            import CodeGenerationContext
        print("✅ Enhanced search types module available: "
              f" {CodeGenerationContext.__version__}")
        print("✅ Enhanced search types module")
    except ImportError as e:
        print(f"❌ Cannot import enhanced search types: {e}")
        return False

    try:
        from genericsuite_codegen.agent.document_retrieval_tool \
            import DocumentRetrievalTool
        print("✅ Document retrieval tool module available: "
              f" {DocumentRetrievalTool.__version__}")
        print("✅ Document retrieval tool module")
    except ImportError as e:
        print(f"❌ Cannot import document retrieval tool: {e}")
        return False

    try:
        from genericsuite_codegen.agent.search_templates \
            import SearchTemplateManager
        print("✅ Search templates module available: "
              f" {SearchTemplateManager.__version__}")
        print("✅ Search templates module")
    except ImportError as e:
        print(f"❌ Cannot import search templates: {e}")
        return False

    try:
        from genericsuite_codegen.agent.context_determination \
            import ContextDeterminationService
        print("✅ Context determination module available: "
              f" {ContextDeterminationService.__version__}")
        print("✅ Context determination module")
    except ImportError as e:
        print(f"❌ Cannot import context determination: {e}")
        return False

    try:
        from genericsuite_codegen.agent.enhanced_search \
            import EnhancedVectorSearch
        print("✅ Enhanced search module available: "
              f" {EnhancedVectorSearch.__version__}")
    except ImportError as e:
        print(f"❌ Cannot import enhanced search: {e}")
        return False

    return True


def main():
    """Main test runner function."""
    print("Enhanced Search Component Test Runner")
    print("=" * 60)

    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing"
              " dependencies.")
        sys.exit(1)

    print("\n✅ All dependencies available.\n")

    # Run tests
    success = run_tests()

    if success:
        print("\n🎉 All enhanced search tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
