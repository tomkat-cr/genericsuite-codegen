#!/usr/bin/env python3
"""Integration test for configuration system."""

import os
import tempfile
import json
from pathlib import Path

from .config_loader import ConfigLoader, EnhancedSearchConfig
from .config_validator import ConfigValidator, ValidationError


def test_config_loading():
    """Test configuration loading functionality."""
    print("Testing configuration loading...")

    loader = ConfigLoader()

    # Test default configuration loading
    config = loader.load_enhanced_search_config()
    assert isinstance(config, EnhancedSearchConfig)
    assert config.enabled is True
    assert config.max_context_length > 0
    print("✓ Default configuration loaded successfully")

    # Test environment-specific configuration
    config_dev = loader.load_enhanced_search_config(environment="development")
    assert config_dev.max_context_length == 15000  # Development has higher limit
    print("✓ Development configuration loaded successfully")

    # Test search templates loading
    templates = loader.load_search_templates()
    assert "templates" in templates
    assert "json" in templates["templates"]
    assert "langchain" in templates["templates"]
    print("✓ Search templates loaded successfully")


def test_config_validation():
    """Test configuration validation functionality."""
    print("\nTesting configuration validation...")

    validator = ConfigValidator()

    # Test valid configuration
    valid_config = {
        "enhanced_search": {
            "enabled": True,
            "max_context_length": 10000
        },
        "local_storage": {
            "local_repo_path": "test_path",
            "max_file_size_mb": 5
        }
    }

    result = validator.validate_enhanced_search_config(valid_config)
    assert result is True
    print("✓ Valid configuration passed validation")

    # Test invalid configuration
    invalid_config = {
        "enhanced_search": {
            "enabled": "not_a_boolean",  # Should be boolean
            "max_context_length": -1     # Should be positive
        }
    }

    try:
        validator.validate_enhanced_search_config(invalid_config)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("✓ Invalid configuration correctly rejected")

    # Test valid templates
    valid_templates = {
        "templates": {
            "test": {
                "template": "test template",
                "file_type_filter": "py",
                "priority": 1
            }
        }
    }

    result = validator.validate_search_templates_config(valid_templates)
    assert result is True
    print("✓ Valid templates passed validation")


def test_environment_variable_override():
    """Test environment variable configuration override."""
    print("\nTesting environment variable override...")

    # Set test environment variables
    test_env_vars = {
        "ENHANCED_SEARCH_ENABLED": "false",
        "ENHANCED_SEARCH_MAX_CONTEXT_LENGTH": "5000",
        "LOCAL_REPO_DIR": "test_repo_path",
        "ENHANCED_SEARCH_LOG_LEVEL": "ERROR"
    }

    # Save original values
    original_values = {}
    for key in test_env_vars:
        original_values[key] = os.environ.get(key)
        os.environ[key] = test_env_vars[key]

    try:
        loader = ConfigLoader()
        config = loader.load_enhanced_search_config()

        # Check that environment variables override defaults
        assert config.enabled is False  # Overridden by env var
        assert config.max_context_length == 5000  # Overridden by env var
        assert config.local_repo_path == "test_repo_path"  # Overridden by env var
        assert config.log_level == "ERROR"  # Overridden by env var

        print("✓ Environment variables correctly override configuration")

    finally:
        # Restore original environment
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def test_config_file_creation():
    """Test configuration file creation and validation."""
    print("\nTesting configuration file creation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test configuration
        test_config = {
            "enhanced_search": {
                "enabled": True,
                "max_context_length": 8000
            },
            "logging": {
                "level": "INFO"
            }
        }

        config_file = temp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f, indent=2)

        # Validate the created file
        validator = ConfigValidator()
        result = validator.validate_config_file(config_file)
        assert result is True

        print("✓ Configuration file created and validated successfully")


def test_config_caching():
    """Test configuration caching functionality."""
    print("\nTesting configuration caching...")

    loader = ConfigLoader()

    # Load configuration twice
    config1 = loader.load_enhanced_search_config()
    config2 = loader.load_enhanced_search_config()

    # Should be the same (cached)
    assert config1.enabled == config2.enabled
    assert config1.max_context_length == config2.max_context_length

    # Clear cache and reload
    loader.reload_config()
    config3 = loader.load_enhanced_search_config()

    # Should still be the same values but freshly loaded
    assert config1.enabled == config3.enabled
    assert config1.max_context_length == config3.max_context_length

    print("✓ Configuration caching works correctly")


def main():
    """Run all integration tests."""
    print("Running Enhanced Search Configuration Integration Tests")
    print("=" * 60)

    try:
        test_config_loading()
        test_config_validation()
        test_environment_variable_override()
        test_config_file_creation()
        test_config_caching()

        print("\n" + "=" * 60)
        print("✓ All integration tests passed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
