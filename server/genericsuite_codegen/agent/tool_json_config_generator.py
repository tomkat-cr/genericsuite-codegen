"""
JSON Configuration Generation Tools
"""

from typing import List, Dict, Any, Tuple

from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_error,
)

from genericsuite_codegen.agent.types import (
    JSONConfigResultRecord,
    JSONConfigResult,
)

from genericsuite_codegen.agent.tool_knowledge_base import KnowledgeBaseTool


DEBUG = True


class JSONConfigGenerator:
    """
    JSON configuration generator for GenericSuite table and form definitions.

    Generates valid JSON configurations following GenericSuite patterns
    with proper validation, examples, and documentation.
    """

    def __init__(self, kb_tool: KnowledgeBaseTool):
        """Initialize the JSON configuration generator."""
        self.kb_tool = kb_tool
        self._load_templates()

    def _load_templates(self) -> None:
        """Load configuration templates and patterns."""

        # Base table configuration template
        self.table_template = {
            "table_name": "",
            "table_config": {
                "id_field": "id",
                "fields": {},
                "validations": {},
                "relationships": {},
                "permissions": {
                    "create": ["admin", "user"],
                    "read": ["admin", "user"],
                    "update": ["admin", "user"],
                    "delete": ["admin"]
                },
                "ui_config": {
                    "list_view": {
                        "columns": [],
                        "searchable_fields": [],
                        "sortable_fields": []
                    },
                    "form_view": {
                        "field_order": [],
                        "required_fields": [],
                        "hidden_fields": []
                    }
                }
            }
        }

        # Common field types and their configurations
        self.field_types = {
            "string": {
                "type": "string",
                "max_length": 255,
                "required": False,
                "default": ""
            },
            "integer": {
                "type": "integer",
                "min_value": None,
                "max_value": None,
                "required": False,
                "default": 0
            },
            "float": {
                "type": "float",
                "min_value": None,
                "max_value": None,
                "precision": 2,
                "required": False,
                "default": 0.0
            },
            "boolean": {
                "type": "boolean",
                "required": False,
                "default": False
            },
            "date": {
                "type": "date",
                "format": "YYYY-MM-DD",
                "required": False,
                "default": None
            },
            "datetime": {
                "type": "datetime",
                "format": "YYYY-MM-DD HH:mm:ss",
                "required": False,
                "default": None
            },
            "email": {
                "type": "email",
                "max_length": 255,
                "required": False,
                "validation": "email_format"
            },
            "url": {
                "type": "url",
                "max_length": 500,
                "required": False,
                "validation": "url_format"
            },
            "text": {
                "type": "text",
                "max_length": 5000,
                "required": False,
                "default": ""
            },
            "select": {
                "type": "select",
                "options": [],
                "multiple": False,
                "required": False,
                "default": None
            },
            "file": {
                "type": "file",
                "allowed_types": ["image/*", "application/pdf"],
                "max_size": "10MB",
                "required": False
            }
        }

        # Form configuration templates

        # Frontend configuration template (based on FrontendCrudEditorConfig)
        self.frontend_template = {
            "baseUrl": "",
            "title": "",
            "name": "",
            "component": "",
            "dbApiUrl": "",
            "fieldElements": [],
            "mandatoryFilters": {},
            "userIdFilter": False,
            "type": "master_listing",
            "dbListPreRead": [],
            "validations": []
        }

        # Backend configuration template (based on BackendCrudEditorConfig)
        self.backend_template = {
            "table_name": "",
            "creation_pk_name": "id",
            "projection_exclusion": [],
            "email_verification": [],
            "passwords": [],
            "mandatory_fields": [],
            "additional_query_params": []
        }

        # # Base form configuration template
        # self.form_template = {
        #     "form_name": "",
        #     "form_config": {
        #         "fields": {},
        #         "validation_rules": {},
        #         "ui_layout": {
        #             "sections": [],
        #             "field_groups": {}
        #         },
        #         "submit_config": {
        #             "endpoint": "",
        #             "method": "POST",
        #             "success_message": "Form submitted successfully",
        #             "error_message": "Form submission failed"
        #         }
        #     }
        # }

    def generate_table_config(self, requirements: str, table_name: str,
                              include_validation: bool = True
                              ) -> JSONConfigResult:
        """
        Generate a GenericSuite table configuration.

        Args:
            requirements: Requirements for the table.
            table_name: Name of the table.
            include_validation: Include validation rules.

        Returns:
            JSONConfigResult: Generated table configuration.
        """
        _ = DEBUG and log_debug(f"Generating table config for {table_name}")
        try:
            # Get relevant context for table configurations
            context, sources, raw_results = \
                self.kb_tool.get_context_for_generation(
                    query=f"GenericSuite table configuration {requirements}",
                    max_context_length=None,
                    file_type_filter="json"
                )

            # Parse requirements to extract fields and specifications
            fields = self._parse_field_requirements(requirements)

            # Generate base configuration
            config = self.table_template.copy()
            config["table_name"] = table_name
            config["table_config"]["fields"] = fields

            # Add validation rules if requested
            if include_validation:
                config["table_config"]["validations"] = \
                    self._generate_validation_rules(fields)

            # Configure UI settings
            config["table_config"]["ui_config"] = self._generate_ui_config(
                fields, table_name)

            # Generate validation notes
            validation_notes = self._generate_validation_notes(
                config, requirements)

            # Create examples
            examples = self._generate_config_examples(config, "table")

            filename = f"backend/table_{table_name}.json"
            result = JSONConfigResult(
                records=[
                    JSONConfigResultRecord(
                        configuration=config,
                        config_type="table",
                        validation_notes=validation_notes,
                        examples=examples,
                        sources=sources,
                        filename=filename
                    )
                ]
            )
            _ = DEBUG and log_debug(
                f"Generated table config for {table_name} | result: {result}")
            return result

        except Exception as e:
            log_error(f"Failed to generate table configuration: {e}")
            raise RuntimeError(f"Table configuration generation failed: {e}")

    def generate_form_config(
        self,
        requirements: str,
        config_name: str
    ) -> JSONConfigResult:
        """
        Generate a GenericSuite form configuration.

        Args:
            requirements: Requirements for the form.
            config_name: Name of the form.

        Returns:
            JSONConfigResult: Generated form configuration.
        """
        _ = DEBUG and log_debug(f"Generating form config for {config_name}")
        include_validation = False
        frontend_config = self.generate_frontend_config(
            requirements=requirements,
            name=config_name,
            include_validation=include_validation
        )
        backend_config = self.generate_backend_config(
            requirements=requirements,
            name=config_name,
            include_validation=include_validation
        )
        result = JSONConfigResult(
            records=[frontend_config, backend_config])
        _ = DEBUG and log_debug(
            f"Generated form config for {config_name} | result: {result}")
        return result

    # def generate_form_config(self, requirements: str, form_name: str
    #                          ) -> JSONConfigResult:
    #     """
    #     Generate a GenericSuite form configuration.

    #     Args:
    #         requirements: Requirements for the form.
    #         form_name: Name of the form.

    #     Returns:
    #         JSONConfigResult: Generated form configuration.
    #     """
    #     try:
    #         # Get relevant context for form configurations
    #         context, sources, raw_results = \
    #             self.kb_tool.get_context_for_generation(
    #                 query=f"GenericSuite form configuration {requirements}",
    #                 max_context_length=None,
    #                 file_type_filter="json"
    #             )

    #         # Parse requirements to extract form fields
    #         fields = self._parse_field_requirements(requirements)

    #         # Generate base configuration
    #         config = self.form_template.copy()
    #         config["form_name"] = form_name
    #         config["form_config"]["fields"] = fields

    #         # Add validation rules
    #         config["form_config"]["validation_rules"] = \
    #             self._generate_form_validation_rules(
    #             fields)

    #         # Configure UI layout
    #         config["form_config"]["ui_layout"] = self._generate_form_layout(
    #             fields, form_name)

    #         # Generate validation notes
    #         validation_notes = self._generate_validation_notes(
    #             config, requirements)

    #         # Create examples
    #         examples = self._generate_config_examples(config, "form")

    #         return JSONConfigResult(
    #             configuration=config,
    #             config_type="form",
    #             validation_notes=validation_notes,
    #             examples=examples,
    #             sources=sources
    #         )

    #     except Exception as e:
    #         log_error(f"Failed to generate form configuration: {e}")
    #         raise RuntimeError(f"Form configuration generation failed: {e}")

    def generate_frontend_config(
        self,
        requirements: str,
        name: str,
        include_validation: bool = True,
    ) -> JSONConfigResultRecord:
        """
        Generate a GenericSuite Frontend CRUD Editor configuration.

        Args:
            requirements (str): Requirements for the editor.
            name (str): Name of the component/editor.

        Returns:
            JSONConfigResultRecord: Generated frontend configuration.
        """
        _ = DEBUG and log_debug(f"Generating frontend config for {name}")
        try:
            context, sources, raw_results = \
                self.kb_tool.get_context_for_generation(
                    query="GenericSuite frontend " +
                    f"configuration {requirements}",
                    max_context_length=None,
                    file_type_filter="json",
                    enable_dual_search=True
                )

            fields = self._parse_field_requirements(requirements)

            config = self.frontend_template.copy()
            # Basic defaults
            config["name"] = name
            config["title"] = f"{name} Management"
            config["component"] = f"{name}Editor"
            config["baseUrl"] = name.lower()
            config["dbApiUrl"] = name.lower()

            # Map fields to Frontend FieldElements
            field_elements = []
            for field_name, field_def in fields.items():
                field_elements.append(
                    self._map_to_field_element(field_name, field_def))

            config["fieldElements"] = field_elements

            validation_notes = self._generate_validation_notes(
                config, requirements)

            examples = self._generate_config_examples(config, "frontend")

            filename = f"frontend/{name}.json"

            result = JSONConfigResultRecord(
                configuration=config,
                config_type="frontend",
                validation_notes=validation_notes,
                examples=examples,
                sources=sources,
                filename=filename
            )
            _ = DEBUG and log_debug(
                f"Generated frontend configuration: {result}")
            return result

        except Exception as e:
            log_error(f"Failed to generate frontend configuration: {e}")
            raise RuntimeError(
                f"Frontend configuration generation failed: {e}")

    def generate_backend_config(
        self,
        requirements: str,
        name: str,
        include_validation: bool = True,
    ) -> JSONConfigResultRecord:
        """
        Generate a GenericSuite Backend CRUD Editor configuration.

        Args:
            requirements: Requirements for the table.
            name: Database table name.

        Returns:
            JSONConfigResultRecord: Generated backend configuration.
        """
        _ = DEBUG and log_debug(f"Generating backend config for {name}")
        try:
            context, sources, raw_results = \
                self.kb_tool.get_context_for_generation(
                    query=f"GenericSuite backend configuration {requirements}",
                    max_context_length=None,
                    file_type_filter="json",
                    enable_dual_search=True
                )

            fields = self._parse_field_requirements(requirements)

            config = self.backend_template.copy()
            config["table_name"] = name

            # Map fields to backend properties
            password_fields = []
            mandatory_fields = []
            email_verification = []

            for field_name, field_def in fields.items():
                if field_def.get("required"):
                    mandatory_fields.append(field_name)
                if field_def.get("type") == "email":
                    email_verification.append(field_name)
                if field_def.get("type") == "password":
                    password_fields.append(field_name)

            config["mandatory_fields"] = mandatory_fields
            if email_verification:
                config["email_verification"] = email_verification
            if password_fields:
                config["passwords"] = password_fields

            validation_notes = self._generate_validation_notes(
                config, requirements)

            examples = self._generate_config_examples(config, "backend")

            filename = f"backend/{name}.json"

            result = JSONConfigResultRecord(
                configuration=config,
                config_type="backend",
                validation_notes=validation_notes,
                examples=examples,
                sources=sources,
                filename=filename
            )
            _ = DEBUG and log_debug(
                f"Generated backend configuration: {result}")
            return result

        except Exception as e:
            log_error(f"Failed to generate backend configuration: {e}")
            raise RuntimeError(f"Backend configuration generation failed: {e}")

    def _map_to_field_element(self, field_name: str, field_def: Dict[str, Any]
                              ) -> Dict[str, Any]:
        """Map internal field definition to Frontend FieldElement."""

        field_type_map = {
            "string": "text",
            "text": "textarea",
            "integer": "integer",
            "float": "number",
            # Boolean often handled as select or special component, default to
            # integer or switch if available
            "boolean": "integer",
            "date": "date",
            "datetime": "datetime-local",
            "email": "email",
            "url": "text",
            "file": "text",  # File usually needs special handling
            "select": "select"
        }

        internal_type = field_def.get("type", "string")
        frontend_type = field_type_map.get(internal_type, "text")

        element = {
            "name": field_name,
            "label": field_name.replace("_", " ").title(),
            "type": frontend_type,
            "required": field_def.get("required", False),
            "listing": True  # Default to showing in listing
        }

        if internal_type == "select":
            # For select, we might need options. The interface implies
            # 'select_elements' might refer to a predefined list ID, or
            # 'options' array if supported directly.
            # The interface has 'select_elements?: string' (ID for predefined)
            # We'll just set a placeholder compatible with typical usage
            pass

        if field_def.get("default") is not None:
            element["default_value"] = field_def["default"]

        return element

    def _parse_field_requirements(self, requirements: str) -> Dict[str, Any]:
        """
        Parse requirements text to extract field definitions.

        Args:
            requirements: Requirements text.

        Returns:
            Dict[str, Any]: Parsed field definitions.
        """
        fields = {}

        # Common field patterns to look for
        field_patterns = {
            "name": "string",
            "title": "string",
            "description": "text",
            "email": "email",
            "phone": "string",
            "address": "text",
            "age": "integer",
            "price": "float",
            "cost": "float",
            "amount": "float",
            "date": "date",
            "created": "datetime",
            "updated": "datetime",
            "active": "boolean",
            "enabled": "boolean",
            "status": "select",
            "category": "select",
            "type": "select",
            "url": "url",
            "website": "url",
            "image": "file",
            "document": "file",
            "notes": "text",
            "comments": "text"
        }

        # Extract field names from requirements
        requirements_lower = requirements.lower()

        for field_name, field_type in field_patterns.items():
            if field_name in requirements_lower:
                field_config = self.field_types[field_type].copy()

                # Customize based on field name
                if field_name in ["name", "title"]:
                    field_config["required"] = True
                    field_config["max_length"] = 100
                elif field_name == "email":
                    field_config["required"] = True
                elif field_name in ["status", "category", "type"]:
                    field_config["options"] = self._get_default_options(
                        field_name)
                    field_config["required"] = True

                fields[field_name] = field_config

        # If no fields detected, create basic fields
        if not fields:
            fields = {
                "name": {
                    "type": "string",
                    "max_length": 100,
                    "required": True,
                    "default": ""
                },
                "description": {
                    "type": "text",
                    "max_length": 1000,
                    "required": False,
                    "default": ""
                },
                "created_at": {
                    "type": "datetime",
                    "required": False,
                    "auto_now_add": True
                },
                "updated_at": {
                    "type": "datetime",
                    "required": False,
                    "auto_now": True
                }
            }

        return fields

    def _get_default_options(self, field_name: str) -> List[str]:
        """Get default options for select fields."""
        options_map = {
            "status": ["active", "inactive", "pending", "archived"],
            "category": ["general", "important", "urgent", "low_priority"],
            "type": ["standard", "premium", "basic", "advanced"]
        }
        return options_map.get(field_name, ["option1", "option2", "option3"])

    def _generate_validation_rules(self, fields: Dict[str, Any]
                                   ) -> Dict[str, Any]:
        """Generate validation rules for table fields."""
        validations = {}

        for field_name, field_config in fields.items():
            field_validations = []

            if field_config.get("required"):
                field_validations.append("required")

            if field_config.get("max_length"):
                field_validations.append(
                    f"max_length:{field_config['max_length']}")

            if field_config.get("min_value") is not None:
                field_validations.append(
                    f"min_value:{field_config['min_value']}")

            if field_config.get("max_value") is not None:
                field_validations.append(
                    f"max_value:{field_config['max_value']}")

            if field_config.get("type") == "email":
                field_validations.append("email")

            if field_config.get("type") == "url":
                field_validations.append("url")

            if field_validations:
                validations[field_name] = field_validations

        return validations

    # def _generate_form_validation_rules(self, fields: Dict[str, Any]
    #                                     ) -> Dict[str, Any]:
    #     """Generate validation rules for form fields."""
    #     return self._generate_validation_rules(fields)  # Same logic for now

    def _generate_ui_config(self, fields: Dict[str, Any], table_name: str
                            ) -> Dict[str, Any]:
        """Generate UI configuration for table."""
        field_names = list(fields.keys())

        # Determine which fields to show in list view
        list_columns = []
        searchable_fields = []
        sortable_fields = []

        for field_name, field_config in fields.items():
            field_type = field_config.get("type", "string")

            # Add to list view if it's a basic display field
            if field_type in [
                "string", "integer", "date", "boolean", "select"] \
                    and len(list_columns) < 5:
                list_columns.append(field_name)

            # Add to searchable if it's text-based
            if field_type in ["string", "text", "email"]:
                searchable_fields.append(field_name)

            # Add to sortable if it's a simple type
            if field_type in ["string", "integer", "float", "date", "datetime",
                              "boolean"]:
                sortable_fields.append(field_name)

        return {
            "list_view": {
                "columns": list_columns,
                "searchable_fields": searchable_fields,
                "sortable_fields": sortable_fields
            },
            "form_view": {
                "field_order": field_names,
                "required_fields": [name for name, config in fields.items()
                                    if config.get("required")],
                "hidden_fields": [name for name in field_names
                                  if name.endswith("_at")]
            }
        }

    def _generate_form_layout(self, fields: Dict[str, Any], form_name: str
                              ) -> Dict[str, Any]:
        """Generate UI layout for form."""
        field_names = list(fields.keys())

        # Group fields into logical sections
        sections = []
        current_section = {
            "title": "Basic Information",
            "fields": []
        }

        for field_name in field_names:
            current_section["fields"].append(field_name)

            # Create new section after every 5 fields
            if len(current_section["fields"]) >= 5:
                sections.append(current_section)
                current_section = {
                    "title": "Additional Information",
                    "fields": []
                }

        # Add remaining fields
        if current_section["fields"]:
            sections.append(current_section)

        return {
            "sections": sections,
            "field_groups": {
                "basic": [name for name in field_names
                          if not name.endswith("_at")],
                "timestamps": [name for name in field_names
                               if name.endswith("_at")]
            }
        }

    def _generate_validation_notes(self, config: Dict[str, Any],
                                   requirements: str) -> List[str]:
        """Generate validation and usage notes for the configuration."""
        notes = [
            "This configuration follows GenericSuite patterns and"
            " conventions.",
            "Ensure all required fields are properly validated in your"
            " application.",
            "Review field types and constraints based on your specific"
            " requirements.",
            "Test the configuration in a development environment before"
            " production use."
        ]

        # Add specific notes based on configuration content
        if "table_config" in config:
            notes.append("Table configuration includes CRUD permissions and UI"
                         " settings.")
            if config["table_config"].get("relationships"):
                notes.append(
                    "Review relationship configurations for proper foreign key"
                    " constraints.")

        if "form_config" in config:
            notes.append(
                "Form configuration includes validation rules and UI layout.")
            notes.append(
                "Customize the submit endpoint and success/error messages as"
                " needed.")

        return notes

    def _generate_config_examples(self, config: Dict[str, Any],
                                  config_type: str) -> Dict[str, Any]:
        """Generate example configurations and usage patterns."""
        examples = {}

        if config_type == "table":
            examples["minimal_table"] = {
                "table_name": "simple_items",
                "table_config": {
                    "id_field": "id",
                    "fields": {
                        "name": {"type": "string", "required": True,
                                 "max_length": 100},
                        "active": {"type": "boolean", "default": True}
                    }
                }
            }

            examples["usage_example"] = {
                "description": "How to use this table configuration",
                "steps": [
                    "1. Save the configuration as a JSON file",
                    "2. Import it into your GenericSuite application",
                    "3. Run database migrations to create the table",
                    "4. Access the auto-generated CRUD endpoints"
                ]
            }

        elif config_type == "form":
            examples["minimal_form"] = {
                "form_name": "contact_form",
                "form_config": {
                    "fields": {
                        "name": {"type": "string", "required": True},
                        "email": {"type": "email", "required": True},
                        "message": {"type": "text", "required": True}
                    }
                }
            }

            examples["usage_example"] = {
                "description": "How to use this form configuration",
                "steps": [
                    "1. Save the configuration as a JSON file",
                    "2. Import it into your GenericSuite application",
                    "3. Create the corresponding React component",
                    "4. Configure the form submission endpoint"
                ]
            }

        return examples

    def validate_configuration(self, config: Dict[str, Any], config_type: str
                               ) -> Tuple[bool, List[str]]:
        """
        Validate a generated configuration against GenericSuite patterns.

        Args:
            config: Configuration to validate.
            config_type: Type of configuration.

        Returns:
            Tuple[bool, List[str]]: Validation result and error messages.
        """
        errors = []

        try:
            if config_type == "table":
                errors.extend(self._validate_table_config(config))
            elif config_type == "form":
                errors.extend(self._validate_form_config(config))
            else:
                errors.append(f"Unknown configuration type: {config_type}")

            return len(errors) == 0, errors

        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors

    def _validate_table_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate table configuration."""
        errors = []

        if "table_name" not in config:
            errors.append("Missing required field: table_name")

        if "table_config" not in config:
            errors.append("Missing required field: table_config")
            return errors

        table_config = config["table_config"]

        if "fields" not in table_config:
            errors.append("Missing required field: table_config.fields")
        else:
            fields = table_config["fields"]
            if not isinstance(fields, dict) or not fields:
                errors.append(
                    "table_config.fields must be a non-empty dictionary")

            # Validate individual fields
            for field_name, field_config in fields.items():
                if not isinstance(field_config, dict):
                    errors.append(
                        f"Field {field_name} configuration must be"
                        " a dictionary")
                    continue

                if "type" not in field_config:
                    errors.append(
                        f"Field {field_name} missing required 'type' property")

        return errors

    def _validate_form_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate form configuration."""
        errors = []

        if "form_name" not in config:
            errors.append("Missing required field: form_name")

        if "form_config" not in config:
            errors.append("Missing required field: form_config")
            return errors

        form_config = config["form_config"]

        if "fields" not in form_config:
            errors.append("Missing required field: form_config.fields")
        else:
            fields = form_config["fields"]
            if not isinstance(fields, dict) or not fields:
                errors.append(
                    "form_config.fields must be a non-empty dictionary")

        return errors
