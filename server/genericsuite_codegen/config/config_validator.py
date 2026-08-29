"""Configuration validator for enhanced search functionality."""

import json
from pathlib import Path
from typing import Dict, Any, List, Union

from genericsuite_codegen.utilities.app_logger import (
    log_warning,
    log_error,
)

DEBUG = False


class ValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Validator for enhanced search configuration files."""

    def __init__(self):
        """Initialize configuration validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_enhanced_search_config(self, config_data: Dict[str, Any]
                                        ) -> bool:
        """Validate enhanced search configuration.

        Args:
            config_data: Configuration data to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValidationError: If critical validation errors are found
        """
        self.errors.clear()
        self.warnings.clear()

        try:
            self._validate_enhanced_search_section(
                config_data.get("enhanced_search", {}))
            self._validate_local_storage_section(
                config_data.get("local_storage", {}))
            self._validate_search_performance_section(
                config_data.get("search_performance", {}))
            self._validate_context_determination_section(
                config_data.get("context_determination", {}))
            self._validate_logging_section(config_data.get("logging", {}))

            if self.errors:
                error_msg = "Configuration validation failed:\n" + \
                    "\n".join(self.errors)
                raise ValidationError(error_msg)

            if self.warnings:
                log_warning(
                    "Configuration validation warnings:\n" + "\n".join(
                        self.warnings))

            return True

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            log_error(f"Unexpected error during validation: {e}")
            raise ValidationError(f"Validation failed: {e}")

    def validate_search_templates_config(self, templates_data: Dict[str, Any]
                                         ) -> bool:
        """Validate search templates configuration.

        Args:
            templates_data: Templates data to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValidationError: If critical validation errors are found
        """
        self.errors.clear()
        self.warnings.clear()

        try:
            if "templates" not in templates_data:
                self.errors.append("Missing 'templates' section")
                raise ValidationError(
                    "Templates configuration must contain 'templates' section")

            templates = templates_data["templates"]
            if not isinstance(templates, dict):
                self.errors.append("'templates' must be a dictionary")
                raise ValidationError("'templates' must be a dictionary")

            for template_name, template_config in templates.items():
                self._validate_template_config(template_name, template_config)

            # Validate template groups if present
            if "template_groups" in templates_data:
                self._validate_template_groups(
                    templates_data["template_groups"], templates)

            if self.errors:
                error_msg = "Templates validation failed:\n" + \
                    "\n".join(self.errors)
                raise ValidationError(error_msg)

            if self.warnings:
                log_warning(
                    "Templates validation warnings:\n" + "\n".join(
                        self.warnings))

            return True

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            log_error(f"Unexpected error during templates validation: {e}")
            raise ValidationError(f"Templates validation failed: {e}")

    def validate_config_file(self, config_path: Union[str, Path]) -> bool:
        """Validate configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            True if valid, False otherwise

        Raises:
            ValidationError: If file cannot be loaded or is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ValidationError(
                f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValidationError(f"Error reading configuration file: {e}")

        # Determine validation method based on file content
        if "templates" in config_data:
            return self.validate_search_templates_config(config_data)
        else:
            return self.validate_enhanced_search_config(config_data)

    def _validate_enhanced_search_section(self, config: Dict[str, Any]
                                          ) -> None:
        """Validate enhanced search section."""
        if not config:
            self.warnings.append(
                "Missing 'enhanced_search' section, using defaults")
            return

        # Validate boolean fields
        bool_fields = [
            "enabled", "fallback_enabled", "dual_search_enabled",
            "context_determination_enabled", "document_retrieval_enabled"
        ]
        for field in bool_fields:
            if field in config and not isinstance(config[field], bool):
                self.errors.append(
                    f"enhanced_search.{field} must be a boolean")

        # Validate numeric fields
        if "max_context_length" in config:
            max_length = config["max_context_length"]
            if not isinstance(max_length, int) or max_length < 1000:
                self.errors.append(
                    "enhanced_search.max_context_length must be an"
                    " integer >= 1000")

    def _validate_local_storage_section(self, config: Dict[str, Any]) -> None:
        """Validate local storage section."""
        if not config:
            self.warnings.append(
                "Missing 'local_storage' section, using defaults")
            return

        # Validate local_repo_path
        if "local_repo_path" in config:
            if not isinstance(config["local_repo_path"], str):
                self.errors.append(
                    "local_storage.local_repo_path must be a string")

        # Validate max_file_size_mb
        if "max_file_size_mb" in config:
            max_size = config["max_file_size_mb"]
            if not isinstance(max_size, (int, float)) or max_size <= 0:
                self.errors.append(
                    "local_storage.max_file_size_mb must be a positive number")

        # Validate allowed_file_extensions
        if "allowed_file_extensions" in config:
            extensions = config["allowed_file_extensions"]
            if not isinstance(extensions, list):
                self.errors.append(
                    "local_storage.allowed_file_extensions must be a list")
            elif not all(isinstance(ext, str) and ext.startswith('.')
                         for ext in extensions):
                self.errors.append(
                    "local_storage.allowed_file_extensions must contain"
                    " strings starting with '.'")

        # Validate excluded_directories
        if "excluded_directories" in config:
            directories = config["excluded_directories"]
            if not isinstance(directories, list):
                self.errors.append(
                    "local_storage.excluded_directories must be a list")
            elif not all(isinstance(dir_name, str)
                         for dir_name in directories):
                self.errors.append(
                    "local_storage.excluded_directories must contain strings")

    def _validate_search_performance_section(self, config: Dict[str, Any]
                                             ) -> None:
        """Validate search performance section."""
        if not config:
            self.warnings.append(
                "Missing 'search_performance' section, using defaults")
            return

        # Validate max_concurrent_searches
        if "max_concurrent_searches" in config:
            max_searches = config["max_concurrent_searches"]
            if not isinstance(max_searches, int) or max_searches < 1:
                self.errors.append(
                    "search_performance.max_concurrent_searches must be"
                    " a positive integer")

        # Validate search_timeout_seconds
        if "search_timeout_seconds" in config:
            timeout = config["search_timeout_seconds"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                self.errors.append(
                    "search_performance.search_timeout_seconds must be"
                    " a positive number")

        # Validate boolean fields
        bool_fields = ["cache_enabled"]
        for field in bool_fields:
            if field in config and not isinstance(config[field], bool):
                self.errors.append(
                    f"search_performance.{field} must be a boolean")

        # Validate cache_ttl_seconds
        if "cache_ttl_seconds" in config:
            ttl = config["cache_ttl_seconds"]
            if not isinstance(ttl, int) or ttl < 0:
                self.errors.append(
                    "search_performance.cache_ttl_seconds must be"
                    " a non-negative integer")

    def _validate_context_determination_section(self, config: Dict[str, Any]
                                                ) -> None:
        """Validate context determination section."""
        if not config:
            self.warnings.append(
                "Missing 'context_determination' section, using defaults")
            return

        # Validate confidence_threshold
        if "confidence_threshold" in config:
            threshold = config["confidence_threshold"]
            if not isinstance(threshold, (int, float)) or \
                    not (0.0 <= threshold <= 1.0):
                self.errors.append(
                    "context_determination.confidence_threshold must be"
                    " a number between 0.0 and 1.0")

        # Validate default_context
        if "default_context" in config:
            if not isinstance(config["default_context"], str):
                self.errors.append(
                    "context_determination.default_context must be a string")

        # Validate context_keywords
        if "context_keywords" in config:
            keywords = config["context_keywords"]
            if not isinstance(keywords, dict):
                self.errors.append(
                    "context_determination.context_keywords must be"
                    " a dictionary")
            else:
                for context_type, keyword_list in keywords.items():
                    if not isinstance(keyword_list, list):
                        self.errors.append(
                            "context_determination.context_keywords"
                            f".{context_type} must be a list")
                    elif not all(isinstance(keyword, str)
                                 for keyword in keyword_list):
                        self.errors.append(
                            "context_determination.context_keywords."
                            f"{context_type} must contain strings")

    def _validate_logging_section(self, config: Dict[str, Any]) -> None:
        """Validate logging section."""
        if not config:
            self.warnings.append("Missing 'logging' section, using defaults")
            return

        # Validate log level
        if "level" in config:
            level = config["level"]
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if not isinstance(level, str) or level.upper() not in valid_levels:
                self.errors.append(
                    f"logging.level must be one of: {', '.join(valid_levels)}")

        # Validate boolean fields
        bool_fields = [
            "log_search_queries", "log_document_retrieval",
            "log_context_determination", "log_performance_metrics"
        ]
        for field in bool_fields:
            if field in config and not isinstance(config[field], bool):
                self.errors.append(f"logging.{field} must be a boolean")

    def _validate_template_config(self, template_name: str,
                                  template_config: Dict[str, Any]) -> None:
        """Validate individual template configuration."""
        if not isinstance(template_config, dict):
            self.errors.append(
                f"Template '{template_name}' must be a dictionary")
            return

        # Validate required fields
        if "template" not in template_config:
            self.errors.append(
                f"Template '{template_name}' missing required"
                " 'template' field")
        elif not isinstance(template_config["template"], str):
            self.errors.append(
                f"Template '{template_name}' 'template' field must be"
                " a string")

        # Validate optional fields
        if "file_type_filter" in template_config:
            filter_value = template_config["file_type_filter"]
            if filter_value is not None and not isinstance(filter_value, str):
                self.errors.append(
                    f"Template '{template_name}' 'file_type_filter' must be"
                    " a string or null")

        if "priority" in template_config:
            priority = template_config["priority"]
            if not isinstance(priority, int):
                self.errors.append(
                    f"Template '{template_name}' 'priority' must be"
                    " an integer")

        if "description" in template_config:
            description = template_config["description"]
            if not isinstance(description, str):
                self.errors.append(
                    f"Template '{template_name}' 'description' must be"
                    " a string")

    def _validate_template_groups(self, groups: Dict[str, List[str]],
                                  templates: Dict[str, Any]) -> None:
        """Validate template groups."""
        if not isinstance(groups, dict):
            self.errors.append("'template_groups' must be a dictionary")
            return

        template_names = set(templates.keys())

        for group_name, template_list in groups.items():
            if not isinstance(template_list, list):
                self.errors.append(
                    f"Template group '{group_name}' must be a list")
                continue

            for template_name in template_list:
                if not isinstance(template_name, str):
                    self.errors.append(
                        f"Template group '{group_name}' must contain strings")
                elif template_name not in template_names:
                    self.warnings.append(
                        f"Template group '{group_name}' references"
                        " unknown template '{template_name}'")
