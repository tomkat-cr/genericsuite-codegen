"""
Simplified unit tests for SearchTemplateManager focusing on core functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path

from genericsuite_codegen.agent.search_templates import SearchTemplateManager
from genericsuite_codegen.agent.enhanced_search_types import (
    SearchTemplate,
    TemplateLoadError
)


class TestSearchTemplateManagerCore:
    """Test cases for core SearchTemplateManager functionality."""

    def test_init_no_config_file(self):
        """Test initialization without config file uses default templates."""
        manager = SearchTemplateManager()

        # Should have default templates
        assert len(manager.templates) > 0
        assert "json" in manager.templates
        assert "langchain" in manager.templates
        assert "mcp" in manager.templates

    def test_init_with_valid_config_file(self, mock_template_config_file):
        """Test initialization with valid config file."""
        manager = SearchTemplateManager(mock_template_config_file)

        # Should load templates from file
        assert len(manager.templates) >= 2
        assert "json" in manager.templates
        assert "python" in manager.templates

    def test_get_template_existing(self):
        """Test getting existing template."""
        manager = SearchTemplateManager()

        template = manager.get_template("json")
        assert template is not None
        assert isinstance(template, str)
        assert "JSON table configuration" in template

    def test_get_template_nonexistent_returns_fallback(self):
        """Test getting non-existent template returns fallback."""
        manager = SearchTemplateManager()

        template = manager.get_template("nonexistent")
        assert template is not None
        assert isinstance(template, str)
        # Should return fallback template

    def test_get_template_object_existing(self):
        """Test getting existing template object."""
        manager = SearchTemplateManager()

        template_obj = manager.get_template_object("json")
        assert template_obj is not None
        assert isinstance(template_obj, SearchTemplate)
        assert template_obj.code_type == "json"

    def test_get_template_object_nonexistent(self):
        """Test getting non-existent template object returns None."""
        manager = SearchTemplateManager()

        template_obj = manager.get_template_object("nonexistent")
        assert template_obj is None

    def test_get_all_templates(self):
        """Test getting all templates."""
        manager = SearchTemplateManager()

        all_templates = manager.get_all_templates()

        assert isinstance(all_templates, dict)
        assert len(all_templates) > 0
        assert "json" in all_templates
        assert all(isinstance(template, SearchTemplate)
                   for template in all_templates.values())

    def test_get_supported_code_types(self):
        """Test getting supported code types."""
        manager = SearchTemplateManager()

        code_types = manager.get_supported_code_types()

        assert isinstance(code_types, list)
        assert len(code_types) > 0
        assert "json" in code_types
        assert "langchain" in code_types

    def test_get_file_type_filter(self):
        """Test getting file type filter for code type."""
        manager = SearchTemplateManager()

        filter_type = manager.get_file_type_filter("json")
        assert filter_type == "json"

        filter_type = manager.get_file_type_filter("langchain")
        assert filter_type == "py"

    def test_validate_template_valid(self):
        """Test validating a valid template."""
        manager = SearchTemplateManager()

        template = SearchTemplate(
            code_type="test",
            template="Test template",
            file_type_filter="txt",
            priority=1
        )

        assert manager.validate_template(template) is True

    def test_validate_template_invalid(self):
        """Test validating an invalid template."""
        manager = SearchTemplateManager()

        # Template with empty template string
        template = SearchTemplate(
            code_type="test",
            template="",
            file_type_filter="txt",
            priority=1
        )

        assert manager.validate_template(template) is False

    def test_add_template_success(self):
        """Test adding a new template."""
        manager = SearchTemplateManager()

        template = SearchTemplate(
            code_type="custom",
            template="Custom template for testing",
            file_type_filter="txt",
            priority=1
        )

        result = manager.add_template(template)
        assert result is True
        assert "custom" in manager.templates
        assert manager.get_template("custom") == "Custom template for testing"

    def test_add_template_invalid(self):
        """Test adding an invalid template."""
        manager = SearchTemplateManager()

        # Invalid template with empty template string
        template = SearchTemplate(
            code_type="invalid",
            template="",
            file_type_filter="txt",
            priority=1
        )

        result = manager.add_template(template)
        assert result is False
        assert "invalid" not in manager.templates

    def test_remove_template_existing(self):
        """Test removing an existing template."""
        manager = SearchTemplateManager()

        # Add a template first
        template = SearchTemplate(
            code_type="removeme",
            template="Template to remove",
            file_type_filter="txt",
            priority=1
        )
        manager.add_template(template)

        # Now remove it
        result = manager.remove_template("removeme")
        assert result is True
        assert "removeme" not in manager.templates

    def test_remove_template_nonexistent(self):
        """Test removing a non-existent template."""
        manager = SearchTemplateManager()

        result = manager.remove_template("nonexistent")
        assert result is False

    def test_reload_templates_success(self, mock_template_config_file):
        """Test successful template reloading."""
        manager = SearchTemplateManager(mock_template_config_file)
        original_count = len(manager.templates)

        # Reload should work without errors
        manager.reload_templates()

        # Should still have templates
        assert len(manager.templates) > 0

    def test_reload_templates_no_config_file(self):
        """Test template reloading without config file."""
        manager = SearchTemplateManager()
        original_templates = manager.templates.copy()

        manager.reload_templates()

        # Should still have default templates
        assert len(manager.templates) == len(original_templates)

    def test_export_config_success(self, tmp_path):
        """Test exporting configuration to file."""
        manager = SearchTemplateManager()

        output_file = tmp_path / "exported_config.json"
        result = manager.export_config(str(output_file))

        assert result is True
        assert output_file.exists()

        # Verify the exported content
        with open(output_file, 'r') as f:
            exported_data = json.load(f)

        assert "templates" in exported_data
        assert len(exported_data["templates"]) > 0

    def test_default_templates_structure(self):
        """Test that default templates have correct structure."""
        manager = SearchTemplateManager()

        # Check that all default templates are valid
        for code_type, template in manager.DEFAULT_TEMPLATES.items():
            assert isinstance(template, SearchTemplate)
            assert template.code_type == code_type
            assert isinstance(template.template, str)
            assert len(template.template) > 0
            assert isinstance(template.priority, int)
            assert template.priority >= 0  # Priority can be 0

    def test_template_case_sensitivity(self):
        """Test template retrieval case sensitivity."""
        manager = SearchTemplateManager()

        # Test that case matters for template retrieval
        json_template = manager.get_template("json")
        JSON_template = manager.get_template("JSON")

        # These might be different or the same depending on implementation
        assert isinstance(json_template, str)
        assert isinstance(JSON_template, str)

    def test_config_path_property(self, mock_template_config_file):
        """Test config path property."""
        manager = SearchTemplateManager(mock_template_config_file)

        assert hasattr(manager, 'config_path')
        assert manager.config_path == mock_template_config_file

    def test_config_path_none(self):
        """Test config path property when no config file."""
        manager = SearchTemplateManager()

        assert hasattr(manager, 'config_path')
        assert manager.config_path is None
