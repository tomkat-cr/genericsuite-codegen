#!/usr/bin/env python3
"""
Simple test for MCP server basic functionality.
"""

import sys
from pathlib import Path

from genericsuite_codegen.mcp_server import MCPConfig

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_basic_config():
    """Test basic configuration."""
    print("Testing basic MCP configuration...")

    config = MCPConfig()
    print(f"✓ Default server name: {config.server_name}")
    print(f"✓ Default port: {config.port}")
    print(f"✓ Debug mode: {config.debug}")

    # Test custom config
    custom_config = MCPConfig(
        server_name="test-server",
        port=9999,
        debug=True
    )
    print(f"✓ Custom server name: {custom_config.server_name}")
    print(f"✓ Custom port: {custom_config.port}")

    return True


def test_auth_config():
    """Test authentication configuration."""
    print("\nTesting authentication configuration...")

    from genericsuite_codegen.auth import AuthConfig, MCPAuthenticator

    auth_config = AuthConfig(
        require_auth=False,
        rate_limit_requests=50
    )

    authenticator = MCPAuthenticator(auth_config)
    print(f"✓ Auth required: {auth_config.require_auth}")
    print(f"✓ Rate limit: {auth_config.rate_limit_requests}")

    # Test API key validation (should pass when auth not required)
    is_valid = authenticator.validate_api_key(None)
    print(f"✓ API key validation (no auth): {is_valid}")

    return True


def main():
    """Run simple tests."""
    print("=" * 50)
    print("GenericSuite CodeGen MCP Server - Simple Test")
    print("=" * 50)

    tests = [
        ("Basic Configuration", test_basic_config),
        ("Authentication Configuration", test_auth_config)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name}: PASSED")
                passed += 1
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("All basic tests passed! ✅")
        return 0
    else:
        print("Some tests failed! ❌")
        return 1


if __name__ == "__main__":
    sys.exit(main())
