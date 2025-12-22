"""
Search Template Manager for Enhanced Vector Search

This module provides configurable search templates for different code
generation types, supporting both file-based configuration and hardcoded
fallback templates.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import asdict

from .enhanced_search_types import (
    SearchTemplate,
    TemplateLoadError,
)
from .enhanced_search_logging import (
    get_enhanced_search_logger,
    performance_tracking,
    log_performance,
    handle_enhanced_search_errors
)

logger = logging.getLogger(__name__)


class SearchTemplateManager:
    """Manager for configurable search templates."""

    # Default hardcoded templates as fallback
    DEFAULT_TEMPLATES = {
        "json": SearchTemplate(
            code_type="json",
            template="examples of how to create a JSON table configuration "
                     "files in Genericsuite",
            file_type_filter="json",
            priority=1
        ),
        "langchain": SearchTemplate(
            code_type="langchain",
            template="examples of how to create a Python Langchain Tool "
                     "in Genericsuite",
            file_type_filter="py",
            priority=1
        ),
        "mcp": SearchTemplate(
            code_type="mcp",
            template="examples of how to create a MCP server tool "
                     "in Genericsuite",
            file_type_filter="py",
            priority=1
        ),
        "frontend": SearchTemplate(
            code_type="frontend",
            template="examples of how to create frontend code in Genericsuite",
            file_type_filter="jsx",
            priority=1
        ),
        "backend": SearchTemplate(
            code_type="backend",
            template="examples of how to create backend code in Genericsuite",
            file_type_filter="py",
            priority=1
        ),
        "frontend_ai": SearchTemplate(
            code_type="frontend_ai",
            template="examples of how to create frontend with AI code "
                     "in Genericsuite",
            file_type_filter="jsx",
            priority=1
        ),
        "backend_ai": SearchTemplate(
            code_type="backend_ai",
            template="examples of how to create backend with AI code "
                     "in Genericsuite",
            file_type_filter="py",
            priority=1
        ),
        "generic": SearchTemplate(
            code_type="generic",
            template="examples and rules for creating code in Genericsuite",
            file_type_filter=None,
            priority=0
        )
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the SearchTemplateManager.

        Args:
            config_path: Optional path to configuration file. If None, uses
                        default templates.
        """
        self.config_path = config_path
        self.templates: Dict[str, SearchTemplate] = {}
        self._load_templates()

    @log_performance("template_load")
    @handle_enhanced_search_errors(fallback_enabled=True)
    def _load_templates(self) -> None:
        """Load templates from configuration file or use defaults."""
        with performance_tracking(
            "template_load_detailed",
            config_path=self.config_path,
            has_config_path=self.config_path is not None
        ):
            try:
                if self.config_path and Path(self.config_path).exists():
                    self._load_from_file()
                else:
                    self._load_default_templates()
                    if self.config_path:
                        logger.info(
                            f"Configuration file not found at"
                            f" {self.config_path}, "
                            f"using default templates")
            except TemplateLoadError:
                # Already handled, just use defaults
                self._load_default_templates()
            except Exception as e:
                logger.error(f"Unexpected error loading templates: {e}")
                # Log the error and use defaults
                search_logger = get_enhanced_search_logger()
                template_error = TemplateLoadError(
                    f"Failed to load templates: {e}",
                    config_path=self.config_path,
                    original_exception=e
                )
                search_logger.log_error(template_error)
                self._load_default_templates()

    def _load_from_file(self) -> None:
        """Load templates from configuration file."""
        try:
            # Validate file exists and is readable
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise TemplateLoadError(
                    f"Configuration file not found: {self.config_path}",
                    config_path=self.config_path,
                    error_code="FILE_NOT_FOUND"
                )

            if not config_file.is_file():
                raise TemplateLoadError(
                    f"Configuration path is not a file: {self.config_path}",
                    config_path=self.config_path,
                    error_code="NOT_A_FILE"
                )

            # Check file size (reasonable limit)
            file_size = config_file.stat().st_size
            max_size = 1024 * 1024  # 1MB
            if file_size > max_size:
                raise TemplateLoadError(
                    f"Configuration file too large: {file_size} bytes "
                    f"(max: {max_size})",
                    config_path=self.config_path,
                    error_code="FILE_TOO_LARGE"
                )

            # Read and parse JSON
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except json.JSONDecodeError as e:
                raise TemplateLoadError(
                    f"Invalid JSON in configuration file: {e}",
                    config_path=self.config_path,
                    error_code="INVALID_JSON",
                    original_exception=e
                )
            except UnicodeDecodeError as e:
                raise TemplateLoadError(
                    f"Encoding error reading configuration file: {e}",
                    config_path=self.config_path,
                    error_code="ENCODING_ERROR",
                    original_exception=e
                )

            # Validate configuration structure
            if not isinstance(config_data, dict):
                raise TemplateLoadError(
                    "Configuration must be a JSON object",
                    config_path=self.config_path,
                    error_code="INVALID_STRUCTURE"
                )

            if 'templates' not in config_data:
                raise TemplateLoadError(
                    "Configuration must contain 'templates' key",
                    config_path=self.config_path,
                    error_code="MISSING_TEMPLATES_KEY"
                )

            templates_data = config_data['templates']
            if not isinstance(templates_data, dict):
                raise TemplateLoadError(
                    "Templates must be a dictionary",
                    config_path=self.config_path,
                    error_code="INVALID_TEMPLATES_TYPE"
                )

            # Load templates from configuration
            loaded_templates = {}
            invalid_templates = []

            for code_type, template_data in templates_data.items():
                try:
                    template = self._create_template_from_dict(
                        code_type, template_data)

                    # Validate template
                    if self.validate_template(template):
                        loaded_templates[code_type] = template
                    else:
                        invalid_templates.append(code_type)
                        logger.warning(
                            f"Invalid template for {code_type}: "
                            "failed validation")

                except Exception as e:
                    invalid_templates.append(code_type)
                    logger.warning(
                        f"Failed to create template for {code_type}: {e}")
                    continue

            if not loaded_templates:
                raise TemplateLoadError(
                    "No valid templates found in configuration",
                    config_path=self.config_path,
                    error_code="NO_VALID_TEMPLATES",
                    details={"invalid_templates": invalid_templates}
                )

            self.templates = loaded_templates
            logger.info(
                f"Loaded {len(self.templates)} templates from"
                f" {self.config_path}")

            if invalid_templates:
                logger.warning(
                    f"Skipped {len(invalid_templates)} invalid templates: "
                    f"{', '.join(invalid_templates)}")

        except TemplateLoadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading templates from file: {e}")
            raise TemplateLoadError(
                f"Failed to load templates from file: {e}",
                config_path=self.config_path,
                error_code="LOAD_ERROR",
                original_exception=e
            )

    def _create_template_from_dict(
            self, code_type: str, template_data: Dict[str, Any]
    ) -> SearchTemplate:
        """Create SearchTemplate from dictionary data."""
        if not isinstance(template_data, dict):
            raise ValueError(
                f"Template data for {code_type} must be a dictionary")

        required_fields = ['template']
        for field in required_fields:
            if field not in template_data:
                raise ValueError(
                    f"Missing required field '{field}' for template "
                    f"{code_type}")

        return SearchTemplate(
            code_type=code_type,
            template=template_data['template'],
            file_type_filter=template_data.get('file_type_filter'),
            priority=template_data.get('priority', 1)
        )

    def _load_default_templates(self) -> None:
        """Load default hardcoded templates."""
        self.templates = self.DEFAULT_TEMPLATES.copy()
        logger.info(f"Loaded {len(self.templates)} default templates")

    def get_template(self, code_type: str) -> str:
        """
        Get search template for specific code type.

        Args:
            code_type: The type of code being generated

        Returns:
            Search template string
        """
        template = self.templates.get(code_type)
        if template:
            return template.template

        # Fallback to generic template
        generic_template = self.templates.get("generic")
        if generic_template:
            logger.info(
                f"No specific template for {code_type}, "
                f"using generic template")
            return generic_template.template

        # Ultimate fallback
        fallback_template = ("examples and rules for creating code "
                             "in Genericsuite")
        logger.warning(f"No template found for {code_type}, using fallback")
        return fallback_template

    def get_template_object(self, code_type: str) -> Optional[SearchTemplate]:
        """
        Get complete SearchTemplate object for specific code type.

        Args:
            code_type: The type of code being generated

        Returns:
            SearchTemplate object or None if not found
        """
        return self.templates.get(code_type)

    def get_file_type_filter(self, code_type: str) -> Optional[str]:
        """
        Get file type filter for specific code type.

        Args:
            code_type: The type of code being generated

        Returns:
            File type filter string or None
        """
        template = self.templates.get(code_type)
        return template.file_type_filter if template else None

    def get_all_templates(self) -> Dict[str, SearchTemplate]:
        """
        Get all available templates.

        Returns:
            Dictionary of all templates
        """
        return self.templates.copy()

    def get_supported_code_types(self) -> list[str]:
        """
        Get list of supported code types.

        Returns:
            List of supported code type strings
        """
        return list(self.templates.keys())

    @log_performance("template_reload")
    @handle_enhanced_search_errors(fallback_enabled=True)
    def reload_templates(self) -> None:
        """
        Reload templates from configuration.

        This method can be called to refresh templates without restarting
        the application.
        """
        with performance_tracking("template_reload_detailed"):
            try:
                old_count = len(self.templates)
                # Backup current templates
                old_templates = self.templates.copy()

                self._load_templates()
                new_count = len(self.templates)

                logger.info(f"Templates reloaded: {old_count} -> {new_count}")

            except Exception as e:
                logger.error(f"Failed to reload templates: {e}")
                # Restore previous templates on reload failure
                if 'old_templates' in locals():
                    self.templates = old_templates
                    logger.info(
                        "Restored previous templates after reload failure")

                # Log the error
                search_logger = get_enhanced_search_logger()
                template_error = TemplateLoadError(
                    f"Template reload failed: {e}",
                    config_path=self.config_path,
                    original_exception=e
                )
                search_logger.log_error(template_error)

    def validate_template(self, template: SearchTemplate) -> bool:
        """
        Validate a search template.

        Args:
            template: SearchTemplate to validate

        Returns:
            True if template is valid, False otherwise
        """
        try:
            # Check required fields
            if (not template.code_type or
                    not isinstance(template.code_type, str)):
                return False

            if not template.template or not isinstance(template.template, str):
                return False

            # Check template length (reasonable bounds)
            if len(template.template.strip()) < 10:
                return False

            if len(template.template) > 1000:
                return False

            # Check priority is valid
            if (not isinstance(template.priority, int) or
                    template.priority < 0):
                return False

            # Check file_type_filter if provided
            if template.file_type_filter is not None:
                if (not isinstance(template.file_type_filter, str) or
                        not template.file_type_filter.strip()):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating template: {e}")
            return False

    def add_template(self, template: SearchTemplate) -> bool:
        """
        Add or update a template.

        Args:
            template: SearchTemplate to add

        Returns:
            True if template was added successfully, False otherwise
        """
        try:
            if not self.validate_template(template):
                logger.error(f"Invalid template for {template.code_type}")
                return False

            self.templates[template.code_type] = template
            logger.info(f"Added/updated template for {template.code_type}")
            return True

        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return False

    def remove_template(self, code_type: str) -> bool:
        """
        Remove a template.

        Args:
            code_type: Code type to remove

        Returns:
            True if template was removed, False if not found
        """
        if code_type in self.templates:
            del self.templates[code_type]
            logger.info(f"Removed template for {code_type}")
            return True
        return False

    def export_config(self, output_path: str) -> bool:
        """
        Export current templates to configuration file.

        Args:
            output_path: Path to save configuration file

        Returns:
            True if export was successful, False otherwise
        """
        try:
            config_data = {
                "templates": {
                    code_type: asdict(template)
                    for code_type, template in self.templates.items()
                }
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Exported {len(self.templates)} templates to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
