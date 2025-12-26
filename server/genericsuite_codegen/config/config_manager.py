#!/usr/bin/env python3
"""Configuration management utility for enhanced search."""

import argparse
import json
import sys
from pathlib import Path

from .config_loader import ConfigLoader
from .config_validator import ConfigValidator, ValidationError


def validate_config_command(args) -> int:
    """Validate configuration files."""
    validator = ConfigValidator()

    try:
        if args.file:
            # Validate specific file
            config_path = Path(args.file)
            print(f"Validating configuration file: {config_path}")
            validator.validate_config_file(config_path)
            print("✓ Configuration file is valid")
        else:
            # Validate all configuration files
            config_dir = Path(__file__).parent
            config_files = [
                "enhanced_search_config.json",
                "enhanced_search_config.development.json",
                "enhanced_search_config.production.json",
                "enhanced_search_config.docker.json",
                "search_templates.json",
                "search_templates.extended.json"
            ]

            for config_file in config_files:
                config_path = config_dir / config_file
                if config_path.exists():
                    print(f"Validating: {config_file}")
                    try:
                        validator.validate_config_file(config_path)
                        print(f"✓ {config_file} is valid")
                    except ValidationError as e:
                        print(f"✗ {config_file} validation failed:")
                        print(f"  {e}")
                        return 1
                else:
                    print(f"⚠ {config_file} not found, skipping")

            print("✓ All configuration files are valid")

        return 0

    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1


def load_config_command(args) -> int:
    """Load and display configuration."""
    loader = ConfigLoader()

    try:
        if args.templates:
            # Load search templates
            print("Loading search templates configuration...")
            templates = loader.load_search_templates(args.file)
            print(json.dumps(templates, indent=2))
        else:
            # Load enhanced search config
            print(
                f"Loading enhanced search configuration "
                f"(environment: {args.environment})...")
            config = loader.load_enhanced_search_config(
                config_file=args.file,
                environment=args.environment
            )

            # Convert to dict for JSON serialization
            config_dict = {
                "enhanced_search": {
                    "enabled": config.enabled,
                    "max_context_length": config.max_context_length,
                    "fallback_enabled": config.fallback_enabled,
                    "dual_search_enabled": config.dual_search_enabled,
                    "context_determination_enabled":
                    config.context_determination_enabled,
                    "document_retrieval_enabled":
                    config.document_retrieval_enabled
                },
                "local_storage": {
                    "local_repo_path": config.local_repo_path,
                    "max_file_size_mb": config.max_file_size_mb,
                    "allowed_file_extensions": config.allowed_file_extensions,
                    "excluded_directories": config.excluded_directories
                },
                "search_performance": {
                    "max_concurrent_searches": config.max_concurrent_searches,
                    "search_timeout_seconds": config.search_timeout_seconds,
                    "cache_enabled": config.cache_enabled,
                    "cache_ttl_seconds": config.cache_ttl_seconds
                },
                "context_determination": {
                    "confidence_threshold": config.confidence_threshold,
                    "default_context": config.default_context,
                    "context_keywords": config.context_keywords
                },
                "logging": {
                    "level": config.log_level,
                    "log_search_queries": config.log_search_queries,
                    "log_document_retrieval": config.log_document_retrieval,
                    "log_context_determination":
                    config.log_context_determination,
                    "log_performance_metrics": config.log_performance_metrics
                }
            }

            print(json.dumps(config_dict, indent=2))

        return 0

    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return 1


def create_config_command(args) -> int:
    """Create new configuration file."""
    config_path = Path(args.output)

    if config_path.exists() and not args.force:
        print(f"✗ Configuration file already exists: {config_path}")
        print("Use --force to overwrite")
        return 1

    try:
        if args.templates:
            # Create templates configuration
            loader = ConfigLoader()
            default_templates = loader._get_default_templates()

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_templates, f, indent=2)

            print(f"✓ Created search templates configuration: {config_path}")
        else:
            # Create enhanced search configuration
            from .config_loader import EnhancedSearchConfig

            config = EnhancedSearchConfig()
            config_dict = {
                "enhanced_search": {
                    "enabled": config.enabled,
                    "max_context_length": config.max_context_length,
                    "fallback_enabled": config.fallback_enabled,
                    "dual_search_enabled": config.dual_search_enabled,
                    "context_determination_enabled":
                    config.context_determination_enabled,
                    "document_retrieval_enabled":
                    config.document_retrieval_enabled
                },
                "local_storage": {
                    "local_repo_path": config.local_repo_path,
                    "max_file_size_mb": config.max_file_size_mb,
                    "allowed_file_extensions": config.allowed_file_extensions,
                    "excluded_directories": config.excluded_directories
                },
                "search_performance": {
                    "max_concurrent_searches": config.max_concurrent_searches,
                    "search_timeout_seconds": config.search_timeout_seconds,
                    "cache_enabled": config.cache_enabled,
                    "cache_ttl_seconds": config.cache_ttl_seconds
                },
                "context_determination": {
                    "confidence_threshold": config.confidence_threshold,
                    "default_context": config.default_context,
                    "context_keywords": config.context_keywords
                },
                "logging": {
                    "level": config.log_level,
                    "log_search_queries": config.log_search_queries,
                    "log_document_retrieval": config.log_document_retrieval,
                    "log_context_determination":
                    config.log_context_determination,
                    "log_performance_metrics": config.log_performance_metrics
                }
            }

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)

            print(f"✓ Created enhanced search configuration: {config_path}")

        return 0

    except Exception as e:
        print(f"✗ Error creating configuration: {e}")
        return 1


def main():
    """Main configuration management CLI."""
    parser = argparse.ArgumentParser(
        description="Configuration management for GenericSuite "
        "CodeGen enhanced search"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate configuration files")
    validate_parser.add_argument(
        "--file", "-f",
        help="Specific configuration file to validate"
    )

    # Load command
    load_parser = subparsers.add_parser(
        "load", help="Load and display configuration")
    load_parser.add_argument(
        "--file", "-f",
        help="Specific configuration file to load"
    )
    load_parser.add_argument(
        "--environment", "-e",
        choices=["development", "production", "docker"],
        help="Environment-specific configuration"
    )
    load_parser.add_argument(
        "--templates", "-t",
        action="store_true",
        help="Load search templates instead of enhanced search config"
    )

    # Create command
    create_parser = subparsers.add_parser(
        "create", help="Create new configuration file")
    create_parser.add_argument(
        "output",
        help="Output file path"
    )
    create_parser.add_argument(
        "--templates", "-t",
        action="store_true",
        help="Create search templates configuration"
    )
    create_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing file"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "validate":
        return validate_config_command(args)
    elif args.command == "load":
        return load_config_command(args)
    elif args.command == "create":
        return create_config_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
