
# import pytest
from genericsuite_codegen.agent.tools import \
    JSONConfigGenerator, KnowledgeBaseTool
from unittest.mock import Mock


class TestJSONConfigGeneratorStructure:
    def setup_method(self):
        self.mock_kb_tool = Mock(spec=KnowledgeBaseTool)
        self.generator = JSONConfigGenerator(self.mock_kb_tool)

    def test_frontend_config_structure(self):
        """
        Verify that generated frontend config matches
        FrontendCrudEditorConfig interface.
        """
        requirements = "Create a user management table with name, email," \
            + " and role."

        # Mock the context retrieval to return something to avoid errors if
        # it's called
        self.mock_kb_tool.get_context_for_generation.return_value = (
            "Context", ["source"], [])

        result = self.generator.generate_frontend_config(
            requirements, "UserManagement")
        config = result.configuration

        # Check top-level keys
        assert "baseUrl" in config
        assert "title" in config
        assert "name" in config
        assert "component" in config
        assert "dbApiUrl" in config
        assert "fieldElements" in config
        assert isinstance(config["fieldElements"], list)

    def test_backend_config_structure(self):
        """
        Verify that generated backend config matches BackendCrudEditorConfig
        interface.
        """
        requirements = "Create a user management table."

        self.mock_kb_tool.get_context_for_generation.return_value = (
            "Context", ["source"], [])

        result = self.generator.generate_backend_config(requirements, "users")
        config = result.configuration

        # Check top-level keys
        assert "table_name" in config
        assert config["table_name"] == "users"
        # Optional but common fields
        # assert "mandatory_fields" in config

    def test_field_mapping(self):
        """Verify that requirements map to correct field types."""
        # This will rely on _parse_field_requirements logic which needs to
        # be updated or mapped
        fields = self.generator._parse_field_requirements(
            "name string required")

        # We need to ensure that what _parse_field_requirements returns can be
        # converted to FieldElement
        # Or that generate_frontend_config does the conversion

        self.mock_kb_tool.get_context_for_generation.return_value = (
            "Context", ["source"], [])
        result = self.generator.generate_frontend_config(
            "name string required", "Test")

        field_elements = result.configuration["fieldElements"]
        name_field = next(
            (f for f in field_elements if f["name"] == "name"), None)

        assert name_field is not None
        assert name_field["type"] == "text"  # 'string' should map to 'text'
        assert name_field["required"] is True
        assert fields is not None
