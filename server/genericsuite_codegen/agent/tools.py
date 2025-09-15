"""
Knowledge base search tools for the GenericSuite CodeGen AI agent.

This module provides tools for vector similarity search, context retrieval,
and source attribution for the Pydantic AI agent.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

from pydantic_ai import Tool

from genericsuite_codegen.database.setup import (
    get_database_manager,
    # initialize_database,
    SearchResult,
    VectorSearchError,
    DatabaseConnectionError
)
from genericsuite_codegen.document_processing.embeddings import \
    create_embedding_generator

from genericsuite_codegen.agent.types import (
    KnowledgeBaseSearchResults,
    KnowledgeBaseQuery,
    SearchResultModel,
    ContextRanking,
    JSONConfigRequest,
    JSONConfigResult,
    ValidationRequest,
    ValidationResult,
    PythonCodeRequest,
    FrontendCodeRequest,
    BackendCodeRequest,
    PythonCodeResult,
    CodeGenerationResult,
    ContextQuery,
    ContextResult,
)

# Configure logging
logger = logging.getLogger(__name__)


# Knowledge Base Tools


class KnowledgeBaseTool:
    """
    Knowledge base search tool for vector similarity search and context
    retrieval.

    Provides methods for searching the knowledge base, ranking results,
    and generating context summaries with source attribution.
    """

    def __init__(self):
        """Initialize the knowledge base tool."""
        self.db_manager = get_database_manager()
        # self.db_manager = initialize_database()
        self.embedding_provider = None
        self._initialize_embedding_provider()

    def _initialize_embedding_provider(self) -> None:
        """Initialize the embedding provider for query vectorization."""
        try:
            self.embedding_provider = create_embedding_generator()
            logger.info(
                "Initialized embedding provider for knowledge base tool")
        except Exception as e:
            logger.error(
                f"Failed to initialize embedding provider: {e}")
            raise RuntimeError(
                f"Embedding provider initialization failed: {e}")

    def search(
        self, query: str, limit: int = 5,
        file_type_filter: Optional[str] = None
    ) -> KnowledgeBaseSearchResults:
        """
        Search the knowledge base using vector similarity.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.
            file_type_filter: Optional filter by file type.

        Returns:
            KnowledgeBaseSearchResults: Search results with context and
            attribution.

        Raises:
            VectorSearchError: If search operation fails.
            DatabaseConnectionError: If database connection fails.
        """
        try:
            logger.info(f"Searching knowledge base for query: '{query}' "
                        f"(limit: {limit})")

            # Generate query embedding
            query_embedding = self.embedding_provider \
                .generate_query_embedding(query)

            # Perform vector search
            search_results = self.db_manager.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                file_type_filter=file_type_filter
            )

            # Convert to response models
            result_models = []
            sources = set()

            for result in search_results:
                # Extract file type from metadata or path
                file_type = result.metadata.get(
                    'file_type',
                    result.document_path.split('.')[-1]
                    if '.' in result.document_path else 'unknown')

                result_model = SearchResultModel(
                    content=result.content,
                    document_path=result.document_path,
                    similarity_score=result.similarity_score,
                    file_type=file_type,
                    metadata=result.metadata
                )
                result_models.append(result_model)
                sources.add(result.document_path)

            # Generate context summary
            context_summary = self._generate_context_summary(
                search_results, query)

            # Create final results
            final_results = KnowledgeBaseSearchResults(
                results=result_models,
                total_results=len(result_models),
                query=query,
                context_summary=context_summary,
                sources=list(sources)
            )

            logger.info(
                f"Found {len(result_models)} results from "
                f"{len(sources)} sources")
            return final_results

        except VectorSearchError as e:
            logger.error(f"Vector search failed: {e}")
            raise
        except DatabaseConnectionError as e:
            logger.error(f"Database connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise VectorSearchError(f"Search operation failed: {e}")

    def _generate_context_summary(self, results: List[SearchResult],
                                  query: str) -> str:
        """
        Generate a summary of the retrieved context.

        Args:
            results: List of search results.
            query: Original search query.

        Returns:
            str: Context summary describing the retrieved information.
        """
        if not results:
            return f"No relevant context found for query: '{query}'"

        # Analyze file types and sources
        file_types = {}
        sources = set()
        total_content_length = 0

        for result in results:
            file_type = result.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            sources.add(result.document_path)
            total_content_length += len(result.content)

        # Generate summary
        summary_parts = [
            f"Retrieved {len(results)} relevant document chunks for query "
            f"'{query}'."
        ]

        if len(sources) > 1:
            summary_parts.append(
                f"Information sourced from {len(sources)} different "
                "documents.")

        if file_types:
            file_type_desc = ", ".join(
                [f"{count} {ftype}" for ftype, count in file_types.items()])
            summary_parts.append(f"Content types: {file_type_desc}.")

        # Add relevance information
        if results:
            avg_score = sum(r.similarity_score for r in results) / len(results)
            summary_parts.append(f"Average relevance score: {avg_score:.3f}.")

        return " ".join(summary_parts)

    def rank_context_by_relevance(
        self, results: List[SearchResult],
        query: str
    ) -> List[ContextRanking]:
        """
        Rank and score context results by relevance to the query.

        Args:
            results: List of search results to rank.
            query: Original search query for relevance scoring.

        Returns:
            List[ContextRanking]: Ranked context with relevance scores.
        """
        ranked_context = []
        query_lower = query.lower()

        for result in results:
            # Calculate enhanced relevance score
            relevance_score = self._calculate_relevance_score(
                result, query_lower
            )

            context_ranking = ContextRanking(
                content=result.content,
                relevance_score=relevance_score,
                source_path=result.document_path,
                file_type=result.metadata.get('file_type', 'unknown'),
                metadata=result.metadata
            )
            ranked_context.append(context_ranking)

        # Sort by relevance score (descending)
        ranked_context.sort(key=lambda x: x.relevance_score, reverse=True)

        return ranked_context

    def _calculate_relevance_score(self, result: SearchResult,
                                   query_lower: str) -> float:
        """
        Calculate enhanced relevance score combining similarity and text
        matching.

        Args:
            result: Search result to score.
            query_lower: Lowercase query for text matching.

        Returns:
            float: Enhanced relevance score.
        """
        # Base similarity score (0.0 to 1.0)
        base_score = result.similarity_score

        # Text matching bonus
        content_lower = result.content.lower()
        query_words = query_lower.split()

        # Exact phrase match bonus
        phrase_bonus = 0.1 if query_lower in content_lower else 0.0

        # Word match bonus
        word_matches = sum(1 for word in query_words if word in content_lower)
        word_bonus = (word_matches / len(query_words)) * \
            0.05 if query_words else 0.0

        # File type relevance bonus
        file_type = result.metadata.get('file_type', '')
        file_type_bonus = 0.0

        # Prioritize documentation and code files
        if file_type in ['md', 'rst', 'txt']:
            file_type_bonus = 0.02  # Documentation bonus
        elif file_type in ['py', 'js', 'ts', 'jsx', 'tsx']:
            file_type_bonus = 0.01  # Code file bonus

        # Content length penalty for very short or very long chunks
        content_length = len(result.content)
        length_penalty = 0.0

        if content_length < 50:  # Very short content
            length_penalty = -0.02
        elif content_length > 2000:  # Very long content
            length_penalty = -0.01

        # Calculate final score
        final_score = base_score + phrase_bonus + \
            word_bonus + file_type_bonus + length_penalty

        # Ensure score stays within reasonable bounds
        return max(0.0, min(1.0, final_score))

    def get_context_for_generation(
        self, query: str, max_context_length: int = 4000,
        file_type_filter: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Get formatted context for code generation with length limits.

        Args:
            query: Search query for context retrieval.
            max_context_length: Maximum total context length in characters.
            file_type_filter: Optional filter by file type.

        Returns:
            Tuple[str, List[str]]: Formatted context string and list of
            sources.
        """
        try:
            # Search for relevant context
            search_results = self.search(
                query=query,
                limit=10,  # Get more results for better selection
                file_type_filter=file_type_filter
            )

            if not search_results.results:
                return "No relevant context found.", []

            # Rank results by relevance
            raw_results = [
                SearchResult(
                    content=r.content,
                    metadata=r.metadata,
                    similarity_score=r.similarity_score,
                    document_path=r.document_path
                )
                for r in search_results.results
            ]

            ranked_context = self.rank_context_by_relevance(raw_results, query)

            # Build context string within length limits
            context_parts = []
            current_length = 0
            sources = []

            for context in ranked_context:
                # Format context entry
                source_info = f"Source: {context.source_path}"
                content_with_source = f"{source_info}\n{context.content}\n"

                # Check if adding this context would exceed the limit
                if current_length + len(content_with_source) \
                   > max_context_length:
                    # Try to fit a truncated version
                    remaining_space = max_context_length - current_length - \
                        len(source_info) - 20
                    if remaining_space > 100:
                        # Only add if we have reasonable space
                        truncated_content = \
                            context.content[:remaining_space] + "..."
                        context_parts.append(
                            f"{source_info}\n{truncated_content}\n")
                        sources.append(context.source_path)
                    break

                context_parts.append(content_with_source)
                sources.append(context.source_path)
                current_length += len(content_with_source)

            # Join all context parts
            formatted_context = "\n---\n".join(context_parts)

            # Add summary header
            header = f"Relevant context for: {query}\n" + "=" * 50 + "\n\n"
            final_context = header + formatted_context

            # Remove duplicate sources
            return final_context, list(set(sources))

        except Exception as e:
            logger.error(f"Failed to get context for generation: {e}")
            return f"Error retrieving context: {e}", []


# JSON Configuration Generation Tools


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

        # Base form configuration template
        self.form_template = {
            "form_name": "",
            "form_config": {
                "fields": {},
                "validation_rules": {},
                "ui_layout": {
                    "sections": [],
                    "field_groups": {}
                },
                "submit_config": {
                    "endpoint": "",
                    "method": "POST",
                    "success_message": "Form submitted successfully",
                    "error_message": "Form submission failed"
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
        try:
            # Get relevant context for table configurations
            context, sources = self.kb_tool.get_context_for_generation(
                query=f"GenericSuite table configuration {requirements}",
                max_context_length=3000,
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

            return JSONConfigResult(
                configuration=config,
                config_type="table",
                validation_notes=validation_notes,
                examples=examples,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Failed to generate table configuration: {e}")
            raise RuntimeError(f"Table configuration generation failed: {e}")

    def generate_form_config(self, requirements: str, form_name: str
                             ) -> JSONConfigResult:
        """
        Generate a GenericSuite form configuration.

        Args:
            requirements: Requirements for the form.
            form_name: Name of the form.

        Returns:
            JSONConfigResult: Generated form configuration.
        """
        try:
            # Get relevant context for form configurations
            context, sources = self.kb_tool.get_context_for_generation(
                query=f"GenericSuite form configuration {requirements}",
                max_context_length=3000,
                file_type_filter="json"
            )

            # Parse requirements to extract form fields
            fields = self._parse_field_requirements(requirements)

            # Generate base configuration
            config = self.form_template.copy()
            config["form_name"] = form_name
            config["form_config"]["fields"] = fields

            # Add validation rules
            config["form_config"]["validation_rules"] = \
                self._generate_form_validation_rules(
                fields)

            # Configure UI layout
            config["form_config"]["ui_layout"] = self._generate_form_layout(
                fields, form_name)

            # Generate validation notes
            validation_notes = self._generate_validation_notes(
                config, requirements)

            # Create examples
            examples = self._generate_config_examples(config, "form")

            return JSONConfigResult(
                configuration=config,
                config_type="form",
                validation_notes=validation_notes,
                examples=examples,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Failed to generate form configuration: {e}")
            raise RuntimeError(f"Form configuration generation failed: {e}")

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

    def _generate_form_validation_rules(self, fields: Dict[str, Any]
                                        ) -> Dict[str, Any]:
        """Generate validation rules for form fields."""
        return self._generate_validation_rules(fields)  # Same logic for now

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


# Pydantic AI Tool definitions

def create_knowledge_base_search_tool(kb_tool: KnowledgeBaseTool) -> Tool:
    """
    Create a Pydantic AI tool for knowledge base search.

    Returns:
        Tool: Pydantic AI tool for searching the knowledge base.
    """
    # kb_tool = KnowledgeBaseTool()

    def search_knowledge_base(query: KnowledgeBaseQuery
                              ) -> KnowledgeBaseSearchResults:
        """
        Search the GenericSuite knowledge base for relevant information.

        This tool searches through the ingested GenericSuite documentation,
        code examples, and configuration files to find relevant context
        for answering questions or generating code.

        Args:
            query: Search query with optional filters and limits.

        Returns:
            KnowledgeBaseSearchResults: Search results with source attribution.
        """
        return kb_tool.search(
            query=query.query,
            limit=query.limit,
            file_type_filter=query.file_type_filter
        )

    return Tool(search_knowledge_base, description=(
        "Search the GenericSuite knowledge base for relevant documentation, "
        "code examples, and configuration patterns. Use this tool to find "
        "context before generating code or answering questions about"
        " GenericSuite."
    ))


def create_context_retrieval_tool(kb_tool: KnowledgeBaseTool) -> Tool:
    """
    Create a Pydantic AI tool for context retrieval optimized for code
    generation.

    Returns:
        Tool: Pydantic AI tool for retrieving formatted context.
    """
    # kb_tool = KnowledgeBaseTool()

    def get_generation_context(query: ContextQuery) -> ContextResult:
        """
        Retrieve and format context optimized for code generation.

        This tool retrieves relevant context from the knowledge base
        and formats it for use in code generation prompts, with
        automatic length management and source attribution.

        Args:
            query: Context query with length and type constraints.

        Returns:
            ContextResult: Formatted context with sources.
        """
        context, sources = kb_tool.get_context_for_generation(
            query=query.query,
            max_context_length=query.max_length,
            file_type_filter=query.file_type
        )

        return ContextResult(
            context=context,
            sources=sources,
            query=query.query
        )

    return Tool(get_generation_context, description=(
        "Retrieve formatted context from the knowledge base optimized for "
        "code generation. This tool automatically manages context length "
        "and provides source attribution for generated content."
    ))


# Utility functions for tool integration

def create_json_config_generation_tool(kb_tool: KnowledgeBaseTool) -> Tool:
    """
    Create a Pydantic AI tool for JSON configuration generation.

    Returns:
        Tool: Pydantic AI tool for generating JSON configurations.
    """
    json_generator = JSONConfigGenerator(kb_tool)

    def generate_json_configuration(request: JSONConfigRequest
                                    ) -> JSONConfigResult:
        """
        Generate GenericSuite JSON configurations for tables, forms, and other\
        components.

        This tool creates valid JSON configurations following GenericSuite
        patterns,
        including proper field definitions, validation rules, and UI
        configurations.

        Args:
            request: Configuration generation request with type and
            requirements.

        Returns:
            JSONConfigResult: Generated configuration with validation notes
            and examples.
        """
        if request.config_type == "table":
            table_name = request.table_name or "generated_table"
            return json_generator.generate_table_config(
                requirements=request.requirements,
                table_name=table_name,
                include_validation=request.include_validation
            )
        elif request.config_type == "form":
            form_name = request.table_name or "generated_form"
            return json_generator.generate_form_config(
                requirements=request.requirements,
                form_name=form_name
            )
        else:
            # For other config types, use table as default with modifications
            config_name = request.table_name \
                or f"generated_{request.config_type}"
            result = json_generator.generate_table_config(
                requirements=request.requirements,
                table_name=config_name,
                include_validation=request.include_validation
            )
            result.config_type = request.config_type
            return result

    return Tool(generate_json_configuration, description=(
        "Generate GenericSuite JSON configurations for tables, forms, menus, "
        "and other components. Creates valid configurations with proper field "
        "definitions, validation rules, and UI settings following GenericSuite"
        " patterns."
    ))


def create_config_validation_tool(kb_tool: KnowledgeBaseTool) -> Tool:
    """
    Create a Pydantic AI tool for validating JSON configurations.

    Returns:
        Tool: Pydantic AI tool for validating configurations.
    """
    json_generator = JSONConfigGenerator(kb_tool)

    def validate_json_configuration(request: ValidationRequest
                                    ) -> ValidationResult:
        """
        Validate GenericSuite JSON configurations against patterns and
        requirements.

        This tool checks JSON configurations for compliance with GenericSuite
        standards, proper field definitions, and structural correctness.

        Args:
            request: Validation request with configuration and type.

        Returns:
            ValidationResult: Validation results with errors and suggestions.
        """
        is_valid, errors = json_generator.validate_configuration(
            config=request.configuration,
            config_type=request.config_type
        )

        # Generate suggestions based on errors
        suggestions = []
        for error in errors:
            if "missing" in error.lower():
                suggestions.append(
                    "Add the missing required fields to complete the"
                    " configuration")
            elif "type" in error.lower():
                suggestions.append(
                    "Ensure all fields have valid type definitions")
            elif "empty" in error.lower():
                suggestions.append(
                    "Provide at least one field definition in the"
                    " configuration")

        if not suggestions and is_valid:
            suggestions.append(
                "Configuration looks good! Consider adding more detailed"
                " validation rules.")

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            suggestions=suggestions
        )

    return Tool(validate_json_configuration, description=(
        "Validate GenericSuite JSON configurations for correctness and "
        "compliance with framework patterns. Provides detailed error "
        "messages and suggestions for improvement."
    ))


def get_all_knowledge_base_tools(kb_tool: KnowledgeBaseTool) -> List[Tool]:
    """
    Get all knowledge base tools for agent integration.

    Returns:
        List[Tool]: List of all knowledge base tools.
    """
    return [
        create_knowledge_base_search_tool(kb_tool),
        create_context_retrieval_tool(kb_tool)
    ]


# Python Code Generation Tools


class PythonCodeGenerator:
    """
    Python code generator for GenericSuite applications.

    Generates Langchain Tools, MCP Tools, and other Python code
    following GenericSuite patterns and best practices.
    """

    def __init__(self, kb_tool: KnowledgeBaseTool):
        """Initialize the Python code generator."""
        self.kb_tool = kb_tool
        self._load_templates()

    def _load_templates(self) -> None:
        """Load code templates and patterns."""
        # Langchain Tool template
        self.langchain_tool_template = '''"""
{tool_name} - Langchain Tool for GenericSuite

{description}
"""

from typing import Dict, Any, Optional, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class {tool_class_name}Input(BaseModel):
    """Input schema for {tool_name}."""
    {input_fields}


class {tool_class_name}(BaseTool):
    """
    {tool_description}

    This tool integrates with GenericSuite applications to provide
    {functionality_description}
    """

    name: str = "{tool_name}"
    description: str = "{tool_description}"
    args_schema: type[BaseModel] = {tool_class_name}Input

    def _run(self, {run_parameters}) -> str:
        """
        Execute the {tool_name} operation.

        Args:
            {run_args_docs}

        Returns:
            str: Result of the operation.
        """
        try:
            logger.info(f"Executing {tool_name} with parameters: "
                        f"{{{run_parameters}}}")

            # Implementation logic
            {implementation_code}

            return result

        except Exception as e:
            logger.error(f"{tool_name} execution failed: {{e}}")
            raise RuntimeError(f"{tool_name} failed: {{e}}")

    async def _arun(self, {run_parameters}) -> str:
        """Async version of _run."""
        return self._run({run_parameters})


# Tool instance for registration
{tool_instance_name} = {tool_class_name}()
'''

        # MCP Tool template
        self.mcp_tool_template = '''"""
{tool_name} - MCP Tool for GenericSuite

{description}
"""

from typing import Dict, Any, Optional, List, Sequence
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class {tool_class_name}Args(BaseModel):
    """Arguments schema for {tool_name}."""
    {args_fields}


# Initialize FastMCP server
mcp = FastMCP("{tool_name}")


@mcp.tool()
def {tool_function_name}({function_parameters}) -> Dict[str, Any]:
    """
    {tool_description}

    Args:
        {function_args_docs}

    Returns:
        Dict[str, Any]: Result of the operation.
    """
    try:
        logger.info(f"Executing {tool_name} MCP tool")

        # Validate arguments
        args = {tool_class_name}Args({args_validation})

        # Implementation logic
        {implementation_code}

        return {{
            "success": True,
            "result": result,
            "tool": "{tool_name}",
            "timestamp": datetime.utcnow().isoformat()
        }}

    except Exception as e:
        logger.error(f"{tool_name} MCP tool failed: {{e}}")
        return {{
            "success": False,
            "error": str(e),
            "tool": "{tool_name}",
            "timestamp": datetime.utcnow().isoformat()
        }}


@mcp.resource("urn:{tool_name}:info")
def get_{tool_function_name}_info() -> str:
    """Get information about the {tool_name} tool."""
    return f"""
{tool_name} MCP Tool

{tool_description}

Usage:
- Call the {tool_function_name} function with appropriate parameters
- Returns structured results with success/error status
- Integrates with GenericSuite applications

Generated by GenericSuite CodeGen
"""


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
'''

        # Utility function template
        self.utility_template = '''"""
{function_name} - GenericSuite Utility Function

{description}
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def {function_name}({function_parameters}) -> {return_type}:
    """
    {function_description}

    Args:
        {function_args_docs}

    Returns:
        {return_type}: {return_description}

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If operation fails.
    """
    try:
        logger.debug(f"Executing {function_name} with parameters: "
                     f"{{{function_parameters}}}")

        # Input validation
        {validation_code}

        # Implementation logic
        {implementation_code}

        logger.debug(f"{function_name} completed successfully")
        return result

    except ValueError as e:
        logger.error(f"Invalid parameters for {function_name}: {{e}}")
        raise
    except Exception as e:
        logger.error(f"{function_name} execution failed: {{e}}")
        raise RuntimeError(f"{function_name} failed: {{e}}")


# Helper functions
{helper_functions}
'''

    def generate_langchain_tool(self, requirements: str, tool_name: str
                                ) -> PythonCodeResult:
        """
        Generate a Langchain Tool following ExampleApp patterns.

        Args:
            requirements: Requirements for the tool.
            tool_name: Name of the tool.

        Returns:
            PythonCodeResult: Generated Langchain tool code.
        """
        try:
            # Get relevant context for Langchain tools
            context, sources = self.kb_tool.get_context_for_generation(
                query=f"GenericSuite Langchain tool {requirements}",
                max_context_length=3000,
                file_type_filter="py"
            )

            # Parse requirements and generate code components
            tool_class_name = self._to_class_name(tool_name)
            tool_instance_name = self._to_instance_name(tool_name)

            # Generate input fields
            input_fields = self._generate_input_fields(requirements)

            # Generate implementation code
            implementation_code = self._generate_implementation_code(
                requirements, "langchain")

            # Generate run parameters
            run_parameters = self._generate_run_parameters(input_fields)

            # Format the template
            code = self.langchain_tool_template.format(
                tool_name=tool_name,
                description=f"Langchain tool for {requirements}",
                tool_class_name=tool_class_name,
                tool_instance_name=tool_instance_name,
                tool_description="GenericSuite Langchain tool that "
                                 f"{requirements}",
                functionality_description=requirements.lower(),
                input_fields=input_fields,
                run_parameters=run_parameters,
                run_args_docs=self._generate_args_docs(input_fields),
                implementation_code=implementation_code
            )

            # Generate imports
            imports = [
                "from langchain.tools import BaseTool",
                "from pydantic import BaseModel, Field",
                "from typing import Dict, Any, Optional, List",
                "import logging"
            ]

            # Generate usage example
            usage_example = self._generate_langchain_usage_example(
                tool_class_name, tool_name)

            # Generate test code
            test_code = self._generate_test_code(
                tool_name, "langchain") if True else None

            # Generate documentation
            documentation = self._generate_documentation(
                tool_name, "langchain", requirements)

            return PythonCodeResult(
                code=code,
                code_type="langchain_tool",
                imports=imports,
                usage_example=usage_example,
                test_code=test_code,
                documentation=documentation,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Failed to generate Langchain tool: {e}")
            raise RuntimeError(f"Langchain tool generation failed: {e}")

    def generate_mcp_tool(self, requirements: str, tool_name: str
                          ) -> PythonCodeResult:
        """
        Generate an MCP Tool compatible with FastMCP framework.

        Args:
            requirements: Requirements for the tool.
            tool_name: Name of the tool.

        Returns:
            PythonCodeResult: Generated MCP tool code.
        """
        try:
            # Get relevant context for MCP tools
            context, sources = self.kb_tool.get_context_for_generation(
                query=f"GenericSuite MCP tool FastMCP {requirements}",
                max_context_length=3000,
                file_type_filter="py"
            )

            # Parse requirements and generate code components
            tool_class_name = self._to_class_name(tool_name)
            tool_function_name = self._to_function_name(tool_name)

            # Generate arguments fields
            args_fields = self._generate_args_fields(requirements)

            # Generate implementation code
            implementation_code = self._generate_implementation_code(
                requirements, "mcp")

            # Generate function parameters
            function_parameters = self._generate_function_parameters(
                args_fields)

            # Format the template
            code = self.mcp_tool_template.format(
                tool_name=tool_name,
                description=f"MCP tool for {requirements}",
                tool_class_name=tool_class_name,
                tool_function_name=tool_function_name,
                tool_description=f"GenericSuite MCP tool that {requirements}",
                args_fields=args_fields,
                function_parameters=function_parameters,
                function_args_docs=self._generate_function_args_docs(
                    args_fields),
                args_validation=self._generate_args_validation(args_fields),
                implementation_code=implementation_code
            )

            # Generate imports
            imports = [
                "from fastmcp import FastMCP",
                "from pydantic import BaseModel, Field",
                "from typing import Dict, Any, Optional, List, Sequence",
                "from datetime import datetime",
                "import logging"
            ]

            # Generate usage example
            usage_example = self._generate_mcp_usage_example(
                tool_function_name, tool_name)

            # Generate test code
            test_code = self._generate_test_code(
                tool_name, "mcp") if True else None

            # Generate documentation
            documentation = self._generate_documentation(
                tool_name, "mcp", requirements)

            return PythonCodeResult(
                code=code,
                code_type="mcp_tool",
                imports=imports,
                usage_example=usage_example,
                test_code=test_code,
                documentation=documentation,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Failed to generate MCP tool: {e}")
            raise RuntimeError(f"MCP tool generation failed: {e}")

    def generate_utility_function(self, requirements: str, function_name: str
                                  ) -> PythonCodeResult:
        """
        Generate a utility function for GenericSuite applications.

        Args:
            requirements: Requirements for the function.
            function_name: Name of the function.

        Returns:
            PythonCodeResult: Generated utility function code.
        """
        try:
            # Get relevant context
            context, sources = self.kb_tool.get_context_for_generation(
                query=f"GenericSuite utility function {requirements}",
                max_context_length=2000,
                file_type_filter="py"
            )

            # Generate function components
            function_parameters = self._generate_utility_parameters(
                requirements)
            return_type = self._determine_return_type(requirements)

            # Generate implementation code
            implementation_code = self._generate_implementation_code(
                requirements, "utility")
            validation_code = self._generate_validation_code(requirements)
            helper_functions = self._generate_helper_functions(requirements)

            # Format the template
            code = self.utility_template.format(
                function_name=function_name,
                description=f"Utility function for {requirements}",
                function_parameters=function_parameters,
                return_type=return_type,
                function_description="GenericSuite utility function that "
                                     f"{requirements}",
                function_args_docs=self._generate_utility_args_docs(
                    function_parameters),
                return_description=f"Result of {requirements}",
                validation_code=validation_code,
                implementation_code=implementation_code,
                helper_functions=helper_functions
            )

            # Generate imports
            imports = [
                "from typing import Dict, Any, Optional, List, Union",
                "from datetime import datetime",
                "import logging"
            ]

            # Generate usage example
            usage_example = self._generate_utility_usage_example(function_name)

            # Generate test code
            test_code = self._generate_test_code(
                function_name, "utility") if True else None

            # Generate documentation
            documentation = self._generate_documentation(
                function_name, "utility", requirements)

            return PythonCodeResult(
                code=code,
                code_type="utility",
                imports=imports,
                usage_example=usage_example,
                test_code=test_code,
                documentation=documentation,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Failed to generate utility function: {e}")
            raise RuntimeError(f"Utility function generation failed: {e}")

    # Helper methods for code generation

    def _to_class_name(self, name: str) -> str:
        """Convert name to PascalCase class name."""
        return ''.join(word.capitalize() for word in
                       name.replace('_', ' ').replace('-', ' ').split())

    def _to_instance_name(self, name: str) -> str:
        """Convert name to snake_case instance name."""
        return name.lower().replace(' ', '_').replace('-', '_')

    def _to_function_name(self, name: str) -> str:
        """Convert name to snake_case function name."""
        return name.lower().replace(' ', '_').replace('-', '_')

    def _generate_input_fields(self, requirements: str) -> str:
        """Generate input fields for Langchain tools."""
        # Basic field generation based on requirements
        fields = []

        if "query" in requirements.lower() or "search" in requirements.lower():
            fields.append(
                'query: str = Field(description="Search query or input text")')

        if "limit" in requirements.lower() or "count" in requirements.lower():
            fields.append(
                'limit: int = Field(default=10, description="Maximum number'
                ' of results")')

        if "filter" in requirements.lower():
            fields.append(
                'filter_type: Optional[str] = Field(default=None, '
                'description="Optional filter criteria")')

        if not fields:
            fields.append(
                'input_data: str = Field(description="Input data for'
                ' processing")')

        return '\n    '.join(fields)

    def _generate_args_fields(self, requirements: str) -> str:
        """Generate argument fields for MCP tools."""
        return self._generate_input_fields(requirements)  # Same logic for now

    def _generate_implementation_code(self, requirements: str, code_type: str
                                      ) -> str:
        """Generate implementation code based on requirements."""
        if "search" in requirements.lower():
            return '''# Perform search operation
            search_results = perform_search(query, limit)
            result = format_search_results(search_results)'''
        elif "process" in requirements.lower():
            return '''# Process input data
            processed_data = process_input(input_data)
            result = format_output(processed_data)'''
        elif "generate" in requirements.lower():
            return '''# Generate output based on input
            generated_content = generate_content(input_data)
            result = validate_and_format(generated_content)'''
        else:
            return '''# Implement the main functionality
            # TODO: Add specific implementation based on requirements
            result = f"Processed: {input_data}"'''

    def _generate_run_parameters(self, input_fields: str) -> str:
        """Generate run method parameters from input fields."""
        # Extract field names from input_fields
        lines = input_fields.split('\n')
        params = []
        for line in lines:
            if ':' in line:
                field_name = line.strip().split(':')[0]
                params.append(field_name)
        return ', '.join(params)

    def _generate_function_parameters(self, args_fields: str) -> str:
        """Generate function parameters from args fields."""
        return self._generate_run_parameters(args_fields)

    def _generate_utility_parameters(self, requirements: str) -> str:
        """Generate parameters for utility functions."""
        if "data" in requirements.lower():
            return "data: Dict[str, Any], options: Optional[Dict[str, Any]]"
            " = None"
        elif "text" in requirements.lower():
            return "text: str, config: Optional[Dict[str, Any]] = None"
        else:
            return "input_value: Any, **kwargs"

    def _generate_args_docs(self, input_fields: str) -> str:
        """Generate argument documentation."""
        return "Parameters extracted from input schema"

    def _generate_function_args_docs(self, args_fields: str) -> str:
        """Generate function argument documentation."""
        return self._generate_args_docs(args_fields)

    def _generate_utility_args_docs(self, parameters: str) -> str:
        """Generate utility function argument documentation."""
        return "Function parameters as specified"

    def _generate_args_validation(self, args_fields: str) -> str:
        """Generate argument validation code."""
        # Extract field names and create validation dict
        lines = args_fields.split('\n')
        validations = []
        for line in lines:
            if ':' in line:
                field_name = line.strip().split(':')[0]
                validations.append(f"{field_name}={field_name}")
        return ', '.join(validations)

    def _determine_return_type(self, requirements: str) -> str:
        """Determine return type based on requirements."""
        if "list" in requirements.lower() or "results" in requirements.lower():
            return "List[Dict[str, Any]]"
        elif "dict" in requirements.lower() or \
                "object" in requirements.lower():
            return "Dict[str, Any]"
        elif "bool" in requirements.lower() or "check" in requirements.lower():
            return "bool"
        else:
            return "str"

    def _generate_validation_code(self, requirements: str) -> str:
        """Generate input validation code."""
        return '''if not input_value:
            raise ValueError("Input value cannot be empty")'''

    def _generate_helper_functions(self, requirements: str) -> str:
        """Generate helper functions."""
        return '''
def format_output(data: Any) -> str:
    """Format output data for return."""
    return str(data)


def validate_input(data: Any) -> bool:
    """Validate input data."""
    return data is not None
'''

    def _generate_langchain_usage_example(self, class_name: str,
                                          tool_name: str) -> str:
        """Generate usage example for Langchain tool."""
        return f'''# Usage example for {tool_name}
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI

# Initialize the tool
tool = {class_name}()

# Use with an agent
llm = OpenAI(temperature=0)
agent = initialize_agent([tool], llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Execute
result = agent.run("Use {tool_name} to process some data")
print(result)
'''

    def _generate_mcp_usage_example(self, function_name: str, tool_name: str
                                    ) -> str:
        """Generate usage example for MCP tool."""
        return f'''# Usage example for {tool_name} MCP tool

# Run the MCP server
python {function_name}.py

# Or integrate with MCP client
from mcp_client import MCPClient

client = MCPClient("http://localhost:8000")
result = client.call_tool("{function_name}", {{"input_data": "test"}})
print(result)
'''

    def _generate_utility_usage_example(self, function_name: str) -> str:
        """Generate usage example for utility function."""
        return f'''# Usage example for {function_name}

from your_module import {function_name}

# Basic usage
result = {function_name}("input data")
print(result)

# With options
result = {function_name}("input data", {{"option": "value"}})
print(result)
'''

    def _generate_test_code(self, name: str, code_type: str) -> str:
        """Generate unit test code."""
        test_name = self._to_function_name(name)

        return f'''"""
Unit tests for {name} {code_type}
"""

import unittest
from unittest.mock import Mock, patch
import pytest

from your_module import {name}


class Test{self._to_class_name(name)}(unittest.TestCase):
    """Test cases for {name}."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_input = "test data"

    def test_{test_name}_success(self):
        """Test successful execution."""
        # Arrange
        expected_result = "expected output"

        # Act
        result = {name}(self.test_input)

        # Assert
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_{test_name}_invalid_input(self):
        """Test handling of invalid input."""
        # Act & Assert
        with self.assertRaises(ValueError):
            {name}("")

    def test_{test_name}_error_handling(self):
        """Test error handling."""
        # This test would depend on specific implementation
        pass


if __name__ == "__main__":
    unittest.main()
'''

    def _generate_documentation(self, name: str, code_type: str,
                                requirements: str) -> str:
        """Generate comprehensive documentation."""
        return f'''# {name} - {code_type.title()}

## Overview
{name} is a GenericSuite {code_type} that {requirements}.

## Features
- Follows GenericSuite patterns and conventions
- Includes comprehensive error handling
- Provides detailed logging and monitoring
- Compatible with GenericSuite applications

## Installation
1. Copy the generated code to your project
2. Install required dependencies
3. Configure as needed for your application

## Usage
See the usage example in the generated code for implementation details.

## Testing
Run the included unit tests to verify functionality:
```bash
python -m pytest test_{self._to_function_name(name)}.py
```

## Integration
This {code_type} integrates with GenericSuite applications and follows
the framework's patterns for consistency and maintainability.

Generated by GenericSuite CodeGen
'''


def get_all_json_generation_tools(kb_tool: KnowledgeBaseTool) -> List[Tool]:
    """
    Get all JSON configuration generation tools.

    Returns:
        List[Tool]: List of JSON generation tools.
    """
    return [
        create_json_config_generation_tool(kb_tool),
        create_config_validation_tool(kb_tool)
    ]


def create_python_code_generation_tool(kb_tool: KnowledgeBaseTool) -> Tool:
    """
    Create a Pydantic AI tool for Python code generation.

    Returns:
        Tool: Pydantic AI tool for generating Python code.
    """
    code_generator = PythonCodeGenerator(kb_tool)

    def generate_python_code(request: PythonCodeRequest) -> PythonCodeResult:
        """
        Generate Python code for GenericSuite applications including Langchain
        Tools and MCP Tools.

        This tool creates well-structured Python code following GenericSuite
        patterns, including proper error handling, documentation, and test
        code.

        Args:
            request: Code generation request with type, requirements, and
            options.

        Returns:
            PythonCodeResult: Generated code with documentation and examples.
        """
        if request.code_type == "langchain_tool":
            return code_generator.generate_langchain_tool(
                requirements=request.requirements,
                tool_name=request.tool_name
            )
        elif request.code_type == "mcp_tool":
            return code_generator.generate_mcp_tool(
                requirements=request.requirements,
                tool_name=request.tool_name
            )
        elif request.code_type == "utility":
            return code_generator.generate_utility_function(
                requirements=request.requirements,
                function_name=request.tool_name
            )
        else:
            # Default to utility function
            return code_generator.generate_utility_function(
                requirements=request.requirements,
                function_name=request.tool_name
            )

    return Tool(generate_python_code, description=(
        "Generate Python code for GenericSuite applications including "
        "Langchain Tools, MCP Tools, utility functions, and API endpoints. "
        "Creates well-documented, tested code following GenericSuite patterns."
    ))


# Frontend and Backend Code Generation Tools


class FrontendCodeGenerator:
    """
    Frontend code generator for GenericSuite applications.

    Generates ReactJS components following GenericSuite UI patterns
    and ExampleApp structure.
    """

    def __init__(self, kb_tool: KnowledgeBaseTool):
        """Initialize the frontend code generator."""
        self.kb_tool = kb_tool
        self._load_templates()

    def _load_templates(self) -> None:
        """Load frontend code templates."""
        # React component template
        self.react_component_template = '''
import React, {{ useState, useEffect }} from 'react';
import {{ {imports} }} from '@/components/ui';
import {{ {api_imports} }} from '@/lib/api';
import {{ {type_imports} }} from '@/types';

interface {component_name}Props {{
  {props_interface}
}}

interface {component_name}State {{
  {state_interface}
}}

const {component_name}: React.FC<{component_name}Props> = ({{
  {props_destructuring}
}}) => {{
  // State management
  const [{state_variables}] = useState<{component_name}State>({{
    {initial_state}
  }});

  // Effects
  useEffect(() => {{
    {use_effect_code}
  }}, []);

  // Event handlers
  {event_handlers}

  // Render helpers
  {render_helpers}

  return (
    <div className="{container_classes}">
      {component_jsx}
    </div>
  );
}};

export default {component_name};
'''

        # React form component template
        self.react_form_template = '''
import React, {{ useState }} from 'react';
import {{ useForm }} from 'react-hook-form';
import {{ zodResolver }} from '@hookform/resolvers/zod';
import * as z from 'zod';
import {{
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
}} from '@/components/ui/form';
import {{ Button }} from '@/components/ui/button';
import {{ Input }} from '@/components/ui/input';
import {{ Textarea }} from '@/components/ui/textarea';
import {{ {additional_imports} }} from '@/components/ui';

// Validation schema
const {form_name}Schema = z.object({{
  {validation_schema}
}});

type {form_name}Values = z.infer<typeof {form_name}Schema>;

interface {form_name}Props {{
  onSubmit: (values: {form_name}Values) => void;
  initialValues?: Partial<{form_name}Values>;
  isLoading?: boolean;
}}

const {form_name}: React.FC<{form_name}Props> = ({{
  onSubmit,
  initialValues,
  isLoading = false
}}) => {{
  const form = useForm<{form_name}Values>({{
    resolver: zodResolver({form_name}Schema),
    defaultValues: {{
      {default_values}
    }}
  }});

  const handleSubmit = (values: {form_name}Values) => {{
    onSubmit(values);
  }};

  return (
    <Form {{...form}}>
      <form onSubmit={{form.handleSubmit(handleSubmit)}} className="space-y-6">
        {form_fields}

        <Button type="submit" disabled={{isLoading}}>
          {{isLoading ? 'Submitting...' : 'Submit'}}
        </Button>
      </form>
    </Form>
  );
}};

export default {form_name};
'''

    def generate_react_component(self, requirements: str, component_name: str,
                                 component_type: str = "component"
                                 ) -> CodeGenerationResult:
        """
        Generate a React component following GenericSuite patterns.

        Args:
            requirements: Requirements for the component.
            component_name: Name of the component.
            component_type: Type of component (form, table, page, component).

        Returns:
            CodeGenerationResult: Generated React component code.
        """
        try:
            # Get relevant context for React components
            context, sources = self.kb_tool.get_context_for_generation(
                query=f"GenericSuite React component {requirements} "
                f"{component_type}",
                max_context_length=3000,
                file_type_filter="jsx"
            )

            if component_type == "form":
                return self._generate_form_component(requirements,
                                                     component_name, sources)
            else:
                return self._generate_generic_component(
                    requirements,
                    component_name,
                    component_type,
                    sources)

        except Exception as e:
            logger.error(f"Failed to generate React component: {e}")
            raise RuntimeError(f"React component generation failed: {e}")

    def _generate_form_component(self, requirements: str, component_name: str,
                                 sources: List[str]) -> CodeGenerationResult:
        """Generate a form component."""
        # Parse requirements to extract form fields
        fields = self._parse_form_fields(requirements)

        # Generate validation schema
        validation_schema = self._generate_validation_schema(fields)

        # Generate form fields JSX
        form_fields = self._generate_form_fields_jsx(fields)

        # Generate default values
        default_values = self._generate_default_values(fields)

        # Format the template
        code = self.react_form_template.format(
            form_name=component_name,
            validation_schema=validation_schema,
            default_values=default_values,
            form_fields=form_fields,
            additional_imports=self._get_additional_imports(fields)
        )

        # Generate additional files
        files = {
            f"{component_name}.test.tsx":
                self._generate_component_test(component_name, "form"),
            f"{component_name}.stories.tsx":
                self._generate_storybook_story(component_name, "form")
        }

        return CodeGenerationResult(
            code=code,
            code_type="react_form",
            framework="react",
            files=files,
            imports=self._get_form_imports(),
            usage_instructions=self._generate_form_usage_instructions(
                component_name),
            integration_notes=self._generate_integration_notes(
                "react", "form"),
            sources=sources
        )

    def _generate_generic_component(
            self, requirements: str, component_name: str,
            component_type: str, sources: List[str]) -> CodeGenerationResult:
        """Generate a generic React component."""
        # Generate component parts
        props_interface = self._generate_props_interface(requirements)
        state_interface = self._generate_state_interface(requirements)
        component_jsx = self._generate_component_jsx(
            requirements, component_type)

        # Format the template
        code = self.react_component_template.format(
            component_name=component_name,
            imports=self._get_component_imports(component_type),
            api_imports=self._get_api_imports(requirements),
            type_imports=self._get_type_imports(requirements),
            props_interface=props_interface,
            state_interface=state_interface,
            props_destructuring=self._generate_props_destructuring(
                props_interface),
            state_variables=self._generate_state_variables(state_interface),
            initial_state=self._generate_initial_state(state_interface),
            use_effect_code=self._generate_use_effect_code(requirements),
            event_handlers=self._generate_event_handlers(requirements),
            render_helpers=self._generate_render_helpers(requirements),
            container_classes=self._get_container_classes(component_type),
            component_jsx=component_jsx
        )

        # Generate additional files
        files = {
            f"{component_name}.test.tsx":
                self._generate_component_test(component_name, component_type),
            f"{component_name}.module.css":
                self._generate_component_styles(component_name, component_type)
        }

        return CodeGenerationResult(
            code=code,
            code_type=f"react_{component_type}",
            framework="react",
            files=files,
            imports=self._get_component_imports_list(component_type),
            usage_instructions=self._generate_component_usage_instructions(
                component_name, component_type),
            integration_notes=self._generate_integration_notes(
                "react", component_type),
            sources=sources
        )

    # Helper methods for React component generation
    def _parse_form_fields(self, requirements: str) -> List[Dict[str, Any]]:
        """Parse requirements to extract form fields."""
        fields = []

        # Common field patterns
        field_patterns = {
            "name": {"type": "text", "required": True},
            "email": {"type": "email", "required": True},
            "password": {"type": "password", "required": True},
            "description": {"type": "textarea", "required": False},
            "phone": {"type": "tel", "required": False},
            "age": {"type": "number", "required": False},
            "date": {"type": "date", "required": False},
            "status": {"type": "select", "required": False, "options": [
                "active", "inactive"]},
            "category": {"type": "select", "required": False, "options": [
                "general", "important"]}
        }

        requirements_lower = requirements.lower()
        for field_name, field_config in field_patterns.items():
            if field_name in requirements_lower:
                fields.append({"name": field_name, **field_config})

        # If no fields detected, add basic fields
        if not fields:
            fields = [
                {"name": "name", "type": "text", "required": True},
                {"name": "description", "type": "textarea", "required": False}
            ]

        return fields

    def _generate_validation_schema(self, fields: List[Dict[str, Any]]) -> str:
        """Generate Zod validation schema."""
        schema_parts = []

        for field in fields:
            field_name = field["name"]
            field_type = field["type"]
            required = field.get("required", False)

            if field_type == "email":
                validation = "z.string().email('Invalid email address')"
            elif field_type == "number":
                validation = "z.number().min(0, 'Must be positive')"
            elif field_type == "textarea":
                validation = "z.string().min(10, 'Must be at least 10 "
                "characters')"
            else:
                validation = "z.string().min(1, 'This field is required')"

            if not required:
                validation += ".optional()"

            schema_parts.append(f"  {field_name}: {validation}")

        return ",\n".join(schema_parts)

    def _generate_form_fields_jsx(self, fields: List[Dict[str, Any]]) -> str:
        """Generate JSX for form fields."""
        jsx_parts = []

        for field in fields:
            field_name = field["name"]
            field_type = field["type"]
            label = field_name.replace("_", " ").title()

            if field_type == "textarea":
                control = "Textarea"
            elif field_type == "select":
                control = "Select"
            else:
                control = "Input"

            jsx = f'''        <FormField
          control={{form.control}}
          name="{field_name}"
          render={{({{ field }}) => (
            <FormItem>
              <FormLabel>{label}</FormLabel>
              <FormControl>
                <{control} placeholder="Enter {label.lower()}" {{...field}} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}}
        />'''

            jsx_parts.append(jsx)

        return "\n\n".join(jsx_parts)

    def _generate_default_values(self, fields: List[Dict[str, Any]]) -> str:
        """Generate default values for form fields."""
        defaults = []

        for field in fields:
            field_name = field["name"]
            field_type = field["type"]

            if field_type == "number":
                default_value = "0"
            elif field_type == "boolean":
                default_value = "false"
            else:
                default_value = '""'

            defaults.append(f"      {field_name}: {default_value}")

        return ",\n".join(defaults)

    def _get_additional_imports(self, fields: List[Dict[str, Any]]) -> str:
        """Get additional UI component imports based on fields."""
        imports = set()

        for field in fields:
            field_type = field["type"]
            if field_type == "textarea":
                imports.add("Textarea")
            elif field_type == "select":
                imports.add("Select")
            elif field_type == "checkbox":
                imports.add("Checkbox")

        return ", ".join(sorted(imports))

    # More helper methods would continue here...
    def _generate_props_interface(self, requirements: str) -> str:
        """Generate TypeScript props interface."""
        return "// Props interface based on requirements"

    def _generate_state_interface(self, requirements: str) -> str:
        """Generate TypeScript state interface."""
        return "// State interface based on requirements"

    def _generate_component_jsx(self, requirements: str, component_type: str
                                ) -> str:
        """Generate component JSX."""
        return \
            f"{{/* {component_type} component JSX based on requirements */}}"

    def _get_component_imports(self, component_type: str) -> str:
        """Get component imports."""
        return "Button, Card, CardContent, CardHeader, CardTitle"

    def _get_api_imports(self, requirements: str) -> str:
        """Get API imports."""
        return "useApi, ApiResponse"

    def _get_type_imports(self, requirements: str) -> str:
        """Get type imports."""
        return "ComponentProps, ApiData"

    def _generate_props_destructuring(self, props_interface: str) -> str:
        """Generate props destructuring."""
        return "// Props destructuring"

    def _generate_state_variables(self, state_interface: str) -> str:
        """Generate state variables."""
        return "data, setData, loading, setLoading"

    def _generate_initial_state(self, state_interface: str) -> str:
        """Generate initial state."""
        return "data: null, loading: false"

    def _generate_use_effect_code(self, requirements: str) -> str:
        """Generate useEffect code."""
        return "// Initialize component data"

    def _generate_event_handlers(self, requirements: str) -> str:
        """Generate event handlers."""
        return "// Event handlers based on requirements"

    def _generate_render_helpers(self, requirements: str) -> str:
        """Generate render helper functions."""
        return "// Render helpers based on requirements"

    def _get_container_classes(self, component_type: str) -> str:
        """Get container CSS classes."""
        return f"container mx-auto p-4 {component_type}-container"

    def _get_form_imports(self) -> List[str]:
        """Get form component imports."""
        return [
            "react-hook-form",
            "@hookform/resolvers/zod",
            "zod",
            "@/components/ui/form",
            "@/components/ui/button",
            "@/components/ui/input"
        ]

    def _get_component_imports_list(self, component_type: str) -> List[str]:
        """Get component imports list."""
        return [
            "react",
            "@/components/ui/button",
            "@/components/ui/card",
            "@/lib/api"
        ]

    def _generate_component_test(self, component_name: str,
                                 component_type: str) -> str:
        """Generate component test file."""
        return f'''import {{ render, screen }} from '@testing-library/react';
import {component_name} from './{component_name}';

describe('{component_name}', () => {{
  it('renders without crashing', () => {{
    render(<{component_name} />);
    expect(screen.getByRole('main')).toBeInTheDocument();
  }});

  // Add more tests based on component functionality
}});
'''

    def _generate_storybook_story(self, component_name: str,
                                  component_type: str) -> str:
        """Generate Storybook story."""
        return f'''import type {{ Meta, StoryObj }} from '@storybook/react';
import {component_name} from './{component_name}';

const meta: Meta<typeof {component_name}> = {{
  title: 'Components/{component_name}',
  component: {component_name},
  parameters: {{
    layout: 'centered',
  }},
}};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {{
  args: {{
    // Default props
  }},
}};
'''

    def _generate_component_styles(self, component_name: str,
                                   component_type: str) -> str:
        """Generate component CSS module."""
        return f'''.{component_name.lower()} {{
  /* Component styles */
}}

.{component_name.lower()}Container {{
  /* Container styles */
}}
'''

    def _generate_form_usage_instructions(self, component_name: str) -> str:
        """Generate form usage instructions."""
        return f'''Usage Instructions for {component_name}:

1. Import the component:
   import {component_name} from '@/components/{component_name}';

2. Use in your page/component:
   <{component_name}
     onSubmit={{handleSubmit}}
     initialValues={{initialData}}
     isLoading={{isSubmitting}}
   />

3. Handle form submission:
   const handleSubmit = (values) => {{
     // Process form data
     console.log(values);
   }};
'''

    def _generate_component_usage_instructions(self, component_name: str,
                                               component_type: str) -> str:
        """Generate component usage instructions."""
        return f'''Usage Instructions for {component_name}:

1. Import the component:
   import {component_name} from '@/components/{component_name}';

2. Use in your application:
   <{component_name} />

3. Customize props as needed based on your requirements.
'''

    def _generate_integration_notes(self, framework: str, component_type: str
                                    ) -> str:
        """Generate integration notes."""
        return f'''Integration Notes for {framework} {component_type}:

- Follows GenericSuite UI patterns and conventions
- Uses ShadCn/UI components for consistency
- Includes proper TypeScript types and interfaces
- Implements responsive design principles
- Compatible with GenericSuite authentication and API patterns

Make sure to:
1. Install required dependencies
2. Configure your build system for the imports
3. Set up proper routing if this is a page component
4. Test the component in your specific environment
'''


class BackendCodeGenerator:
    """
    Backend code generator for GenericSuite applications.

    Generates backend code for FastAPI, Flask, and Chalice frameworks
    following GenericSuite patterns and best practices.
    """

    def __init__(self, kb_tool: KnowledgeBaseTool):
        """Initialize the backend code generator."""
        self.kb_tool = kb_tool
        self._load_templates()

    def _load_templates(self) -> None:
        """Load backend code templates."""
        # FastAPI endpoint template
        self.fastapi_endpoint_template = '''"""
{endpoint_name} API endpoint for GenericSuite application.

{description}
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..auth import get_current_user, require_permissions
from ..database import get_db_session
from ..models import {model_imports}
from ..schemas import {schema_imports}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/{endpoint_prefix}", tags=["{endpoint_tag}"])


# Request/Response models
{request_response_models}


# Endpoints
{endpoints}


# Helper functions
{helper_functions}
'''

        # Flask endpoint template
        self.flask_endpoint_template = '''"""
{endpoint_name} Flask blueprint for GenericSuite application.

{description}
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, ValidationError
from datetime import datetime
import logging

from ..auth import require_permissions
from ..database import db
from ..models import {model_imports}

logger = logging.getLogger(__name__)

{endpoint_name}_bp = Blueprint('{endpoint_name}', __name__,
    url_prefix='/{endpoint_prefix}')


# Schemas
{schemas}


# Routes
{routes}


# Helper functions
{helper_functions}
'''

        # Chalice endpoint template
        self.chalice_endpoint_template = '''"""
{endpoint_name} Chalice routes for GenericSuite application.

{description}
"""

from chalice import Blueprint, Response
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

from ..auth import require_auth, get_current_user
from ..database import get_db_connection
from ..models import {model_imports}
from ..utils import validate_request, format_response

logger = logging.getLogger(__name__)

{endpoint_name}_routes = Blueprint(__name__)


# Request validation schemas
{validation_schemas}


# Routes
{routes}


# Helper functions
{helper_functions}
'''

    def generate_backend_code(
        self, requirements: str, module_name: str,
        framework: str, code_type: str = "api_endpoint"
    ) -> CodeGenerationResult:
        """
        Generate backend code for the specified framework.

        Args:
            requirements: Requirements for the backend code.
            module_name: Name of the module/endpoint.
            framework: Backend framework (fastapi, flask, chalice).
            code_type: Type of code to generate.

        Returns:
            CodeGenerationResult: Generated backend code.
        """
        try:
            # Get relevant context for backend code
            context, sources = self.kb_tool.get_context_for_generation(
                query=f"GenericSuite {framework} {code_type} {requirements}",
                max_context_length=3000,
                file_type_filter="py"
            )

            if framework == "fastapi":
                return self._generate_fastapi_code(requirements, module_name,
                                                   code_type, sources)
            elif framework == "flask":
                return self._generate_flask_code(requirements, module_name,
                                                 code_type, sources)
            elif framework == "chalice":
                return self._generate_chalice_code(requirements, module_name,
                                                   code_type, sources)
            else:
                raise ValueError(f"Unsupported framework: {framework}")

        except Exception as e:
            logger.error(f"Failed to generate {framework} code: {e}")
            raise RuntimeError(f"{framework} code generation failed: {e}")

    def _generate_fastapi_code(
        self, requirements: str, module_name: str,
        code_type: str, sources: List[str]
    ) -> CodeGenerationResult:
        """Generate FastAPI code."""
        # Generate code components
        endpoints = self._generate_fastapi_endpoints(requirements, module_name)
        request_response_models = self._generate_pydantic_models(
            requirements, module_name)
        helper_functions = self._generate_helper_functions(
            requirements, "fastapi")

        # Format the template
        code = self.fastapi_endpoint_template.format(
            endpoint_name=module_name,
            description=f"FastAPI endpoint for {requirements}",
            endpoint_prefix=module_name.lower().replace("_", "-"),
            endpoint_tag=module_name.replace("_", " ").title(),
            model_imports=self._get_model_imports(requirements),
            schema_imports=self._get_schema_imports(requirements),
            request_response_models=request_response_models,
            endpoints=endpoints,
            helper_functions=helper_functions
        )

        # Generate additional files
        files = {
            f"test_{module_name}.py": self._generate_backend_test(
                module_name, "fastapi"),
            f"{module_name}_models.py": self._generate_models_file(
                module_name, "fastapi"),
            f"{module_name}_schemas.py": self._generate_schemas_file(
                module_name, "fastapi")
        }

        return CodeGenerationResult(
            code=code,
            code_type="fastapi_endpoint",
            framework="fastapi",
            files=files,
            imports=self._get_fastapi_imports(),
            usage_instructions=self._generate_backend_usage_instructions(
                module_name, "fastapi"),
            integration_notes=self._generate_backend_integration_notes(
                "fastapi"),
            sources=sources
        )

    def _generate_flask_code(
        self, requirements: str, module_name: str,
        code_type: str, sources: List[str]
    ) -> CodeGenerationResult:
        """Generate Flask code."""
        # Generate code components
        routes = self._generate_flask_routes(requirements, module_name)
        schemas = self._generate_marshmallow_schemas(requirements, module_name)
        helper_functions = self._generate_helper_functions(
            requirements, "flask")

        # Format the template
        code = self.flask_endpoint_template.format(
            endpoint_name=module_name,
            description=f"Flask blueprint for {requirements}",
            endpoint_prefix=module_name.lower().replace("_", "-"),
            model_imports=self._get_model_imports(requirements),
            schemas=schemas,
            routes=routes,
            helper_functions=helper_functions
        )

        # Generate additional files
        files = {
            f"test_{module_name}.py": self._generate_backend_test(
                module_name, "flask"),
            f"{module_name}_models.py": self._generate_models_file(
                module_name, "flask")
        }

        return CodeGenerationResult(
            code=code,
            code_type="flask_blueprint",
            framework="flask",
            files=files,
            imports=self._get_flask_imports(),
            usage_instructions=self._generate_backend_usage_instructions(
                module_name, "flask"),
            integration_notes=self._generate_backend_integration_notes(
                "flask"),
            sources=sources
        )

    def _generate_chalice_code(
        self, requirements: str, module_name: str,
        code_type: str, sources: List[str]
    ) -> CodeGenerationResult:
        """Generate Chalice code."""
        # Generate code components
        routes = self._generate_chalice_routes(requirements, module_name)
        validation_schemas = self._generate_chalice_schemas(
            requirements, module_name)
        helper_functions = self._generate_helper_functions(
            requirements, "chalice")

        # Format the template
        code = self.chalice_endpoint_template.format(
            endpoint_name=module_name,
            description=f"Chalice routes for {requirements}",
            model_imports=self._get_model_imports(requirements),
            validation_schemas=validation_schemas,
            routes=routes,
            helper_functions=helper_functions
        )

        # Generate additional files
        files = {
            f"test_{module_name}.py": self._generate_backend_test(
                module_name, "chalice"),
            f"{module_name}_models.py": self._generate_models_file(
                module_name, "chalice")
        }

        return CodeGenerationResult(
            code=code,
            code_type="chalice_routes",
            framework="chalice",
            files=files,
            imports=self._get_chalice_imports(),
            usage_instructions=self._generate_backend_usage_instructions(
                module_name, "chalice"),
            integration_notes=self._generate_backend_integration_notes(
                "chalice"),
            sources=sources
        )

    # Helper methods for backend code generation
    def _generate_fastapi_endpoints(self, requirements: str, module_name: str
                                    ) -> str:
        """Generate FastAPI endpoints."""
        return f'''@router.get("/")
async def list_{module_name}(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """List {module_name} items."""
    # Implementation based on requirements
    return {{"items": [], "total": 0}}


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_{module_name}(
    item: {module_name.title()}Create,
    current_user = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Create a new {module_name} item."""
    # Implementation based on requirements
    return {{"id": 1, "message": "Created successfully"}}


@router.get("/{{item_id}}")
async def get_{module_name}(
    item_id: int,
    current_user = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Get a specific {module_name} item."""
    # Implementation based on requirements
    return {{"id": item_id}}


@router.put("/{{item_id}}")
async def update_{module_name}(
    item_id: int,
    item: {module_name.title()}Update,
    current_user = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Update a {module_name} item."""
    # Implementation based on requirements
    return {{"id": item_id, "message": "Updated successfully"}}


@router.delete("/{{item_id}}")
async def delete_{module_name}(
    item_id: int,
    current_user = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Delete a {module_name} item."""
    # Implementation based on requirements
    return {{"message": "Deleted successfully"}}'''

    def _generate_flask_routes(self, requirements: str, module_name: str
                               ) -> str:
        """Generate Flask routes."""
        return f'''@{module_name}_bp.route('/', methods=['GET'])
@jwt_required()
def list_{module_name}():
    """List {module_name} items."""
    try:
        # Implementation based on requirements
        return jsonify({{"items": [], "total": 0}})
    except Exception as e:
        logger.error(f"Error listing {module_name}: {{e}}")
        return jsonify({{"error": "Internal server error"}}), 500


@{module_name}_bp.route('/', methods=['POST'])
@jwt_required()
def create_{module_name}():
    """Create a new {module_name} item."""
    try:
        data = request.get_json()
        # Validate and process data
        return jsonify({{"id": 1, "message": "Created successfully"}}), 201
    except ValidationError as e:
        return jsonify({{"error": e.messages}}), 400
    except Exception as e:
        logger.error(f"Error creating {module_name}: {{e}}")
        return jsonify({{"error": "Internal server error"}}), 500


@{module_name}_bp.route('/<int:item_id>', methods=['GET'])
@jwt_required()
def get_{module_name}(item_id):
    """Get a specific {module_name} item."""
    try:
        # Implementation based on requirements
        return jsonify({{"id": item_id}})
    except Exception as e:
        logger.error(f"Error getting {module_name}: {{e}}")
        return jsonify({{"error": "Internal server error"}}), 500'''

    def _generate_chalice_routes(self, requirements: str, module_name: str
                                 ) -> str:
        """Generate Chalice routes."""
        return f'''@{module_name}_routes.route('/', methods=['GET'])
@require_auth
def list_{module_name}():
    """List {module_name} items."""
    try:
        # Implementation based on requirements
        return format_response({{"items": [], "total": 0}})
    except Exception as e:
        logger.error(f"Error listing {module_name}: {{e}}")
        return Response(
            body=json.dumps({{"error": "Internal server error"}}),
            status_code=500,
            headers={{"Content-Type": "application/json"}}
        )


@{module_name}_routes.route('/', methods=['POST'])
@require_auth
def create_{module_name}():
    """Create a new {module_name} item."""
    try:
        request_data = {module_name}_routes.current_request.json_body
        # Validate and process data
        return format_response({{"id": 1, "message": "Created successfully"}},
            201)
    except Exception as e:
        logger.error(f"Error creating {module_name}: {{e}}")
        return Response(
            body=json.dumps({{"error": "Internal server error"}}),
            status_code=500,
            headers={{"Content-Type": "application/json"}}
        )'''

    # More helper methods would continue here...
    def _generate_pydantic_models(self, requirements: str, module_name: str
                                  ) -> str:
        """Generate Pydantic models for FastAPI."""
        return f'''class {module_name.title()}Base(BaseModel):
    """Base {module_name} model."""
    name: str = Field(..., description="Name of the {module_name}")
    description: Optional[str] = Field(None, description="Description")


class {module_name.title()}Create({module_name.title()}Base):
    """Model for creating {module_name}."""
    pass


class {module_name.title()}Update(BaseModel):
    """Model for updating {module_name}."""
    name: Optional[str] = Field(None, description="Name of the {module_name}")
    description: Optional[str] = Field(None, description="Description")


class {module_name.title()}Response({module_name.title()}Base):
    """Model for {module_name} response."""
    id: int = Field(..., description="Unique identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True'''

    def _generate_marshmallow_schemas(self, requirements: str,
                                      module_name: str) -> str:
        """Generate Marshmallow schemas for Flask."""
        return f'''class {module_name.title()}Schema(Schema):
    """Schema for {module_name} validation."""
    name = fields.Str(required=True, validate=fields.Length(min=1, max=100))
    description = fields.Str(missing=None, validate=fields.Length(max=500))


class {module_name.title()}UpdateSchema(Schema):
    """Schema for {module_name} updates."""
    name = fields.Str(validate=fields.Length(min=1, max=100))
    description = fields.Str(validate=fields.Length(max=500))


# Schema instances
{module_name}_schema = {module_name.title()}Schema()
{module_name}_update_schema = {module_name.title()}UpdateSchema()'''

    def _generate_chalice_schemas(self, requirements: str, module_name: str
                                  ) -> str:
        """Generate validation schemas for Chalice."""
        return f'''def validate_{module_name}_create(data):
    """Validate {module_name} creation data."""
    required_fields = ['name']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {{field}}")

    if len(data.get('name', '')) < 1:
        raise ValueError("Name cannot be empty")

    return True


def validate_{module_name}_update(data):
    """Validate {module_name} update data."""
    if 'name' in data and len(data['name']) < 1:
        raise ValueError("Name cannot be empty")

    return True'''

    def _get_model_imports(self, requirements: str) -> str:
        """Get model imports."""
        return "User, BaseModel"

    def _get_schema_imports(self, requirements: str) -> str:
        """Get schema imports."""
        return "UserSchema, BaseSchema"

    def _generate_helper_functions(self, requirements: str, framework: str
                                   ) -> str:
        """Generate helper functions."""
        return '''def format_response(data, status_code=200):
    """Format API response."""
    return {{
        "data": data,
        "status": "success",
        "timestamp": datetime.utcnow().isoformat()
    }}


def handle_error(error, status_code=500):
    """Handle API errors."""
    logger.error(f"API error: {{error}}")
    return {{
        "error": str(error),
        "status": "error",
        "timestamp": datetime.utcnow().isoformat()
    }}'''

    def _get_fastapi_imports(self) -> List[str]:
        """Get FastAPI imports."""
        return [
            "fastapi",
            "pydantic",
            "sqlalchemy",
            "python-jose[cryptography]",
            "passlib[bcrypt]"
        ]

    def _get_flask_imports(self) -> List[str]:
        """Get Flask imports."""
        return [
            "flask",
            "flask-jwt-extended",
            "marshmallow",
            "sqlalchemy",
            "flask-sqlalchemy"
        ]

    def _get_chalice_imports(self) -> List[str]:
        """Get Chalice imports."""
        return [
            "chalice",
            "boto3",
            "pydantic"
        ]

    def _generate_backend_test(self, module_name: str, framework: str) -> str:
        """Generate backend test file."""
        return f'''"""
Tests for {module_name} {framework} endpoints.
"""

import pytest
from unittest.mock import Mock, patch

# Framework-specific test imports would go here

class Test{module_name.title()}Endpoints:
    """Test cases for {module_name} endpoints."""

    def test_list_{module_name}(self):
        """Test listing {module_name} items."""
        # Test implementation
        pass

    def test_create_{module_name}(self):
        """Test creating {module_name} item."""
        # Test implementation
        pass

    def test_get_{module_name}(self):
        """Test getting {module_name} item."""
        # Test implementation
        pass

    def test_update_{module_name}(self):
        """Test updating {module_name} item."""
        # Test implementation
        pass

    def test_delete_{module_name}(self):
        """Test deleting {module_name} item."""
        # Test implementation
        pass
'''

    def _generate_models_file(self, module_name: str, framework: str) -> str:
        """Generate models file."""
        return f'''"""
Database models for {module_name}.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class {module_name.title()}(Base):
    """Database model for {module_name}."""

    __tablename__ = '{module_name.lower()}'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
        onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<{module_name.title()}(id={{self.id}}, name='{{self.name}}')>"
'''

    def _generate_schemas_file(self, module_name: str, framework: str) -> str:
        """Generate schemas file for FastAPI."""
        return f'''"""
Pydantic schemas for {module_name}.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class {module_name.title()}Base(BaseModel):
    """Base schema for {module_name}."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class {module_name.title()}Create({module_name.title()}Base):
    """Schema for creating {module_name}."""
    pass


class {module_name.title()}Update(BaseModel):
    """Schema for updating {module_name}."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class {module_name.title()}InDB({module_name.title()}Base):
    """Schema for {module_name} in database."""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class {module_name.title()}Response({module_name.title()}InDB):
    """Schema for {module_name} API response."""
    pass
'''

    def _generate_backend_usage_instructions(self, module_name: str,
                                             framework: str) -> str:
        """Generate backend usage instructions."""
        return f'''Usage Instructions for {module_name} {framework} endpoint:

1. Install dependencies:
   pip install -r requirements.txt

2. Set up database:
   - Configure database connection
   - Run migrations if needed

3. Register the endpoint:
   {self._get_registration_instructions(framework, module_name)}

4. Test the endpoints:
   - Use the included test file
   - Test with API client (Postman, curl, etc.)

5. Deploy:
   - Configure environment variables
   - Set up production database
   - Deploy using your preferred method
'''

    def _get_registration_instructions(self, framework: str, module_name: str
                                       ) -> str:
        """Get framework-specific registration instructions."""
        if framework == "fastapi":
            return f"app.include_router({module_name}_router)"
        elif framework == "flask":
            return f"app.register_blueprint({module_name}_bp)"
        elif framework == "chalice":
            return f"app.register_blueprint({module_name}_routes)"
        else:
            return "Register according to framework documentation"

    def _generate_backend_integration_notes(self, framework: str) -> str:
        """Generate backend integration notes."""
        return f'''Integration Notes for {framework}:

- Follows GenericSuite backend patterns and conventions
- Includes proper authentication and authorization
- Implements comprehensive error handling and logging
- Uses framework-specific best practices
- Compatible with GenericSuite database models and schemas

Make sure to:
1. Configure authentication middleware
2. Set up database connections
3. Configure logging and monitoring
4. Test all endpoints thoroughly
5. Set up proper deployment pipeline

Framework-specific considerations:
{self._get_framework_considerations(framework)}
'''

    def _get_framework_considerations(self, framework: str) -> str:
        """Get framework-specific considerations."""
        considerations = {
            "fastapi": "- Use dependency injection for database sessions\n"
            "- Leverage automatic API documentation\n"
            "- Implement proper async/await patterns",
            "flask": "- Use blueprints for modular organization\n"
            "- Configure Flask-JWT-Extended for authentication\n"
            "- Set up proper error handlers",
            "chalice": "- Optimize for serverless deployment\n"
            "- Configure proper CORS settings\n"
            "- Use Chalice's built-in authentication features"
        }
        return considerations.get(framework, "Follow framework best practices")


def get_all_python_generation_tools(kb_tool: KnowledgeBaseTool) -> List[Tool]:
    """
    Get all Python code generation tools.

    Returns:
        List[Tool]: List of Python generation tools.
    """
    return [
        create_python_code_generation_tool(kb_tool)
    ]


def create_frontend_code_generation_tool(kb_tool: KnowledgeBaseTool) -> Tool:
    """
    Create a Pydantic AI tool for frontend code generation.

    Returns:
        Tool: Pydantic AI tool for generating frontend code.
    """
    frontend_generator = FrontendCodeGenerator(kb_tool)

    def generate_frontend_code(request: FrontendCodeRequest
                               ) -> CodeGenerationResult:
        """
        Generate ReactJS frontend code following GenericSuite UI patterns.

        This tool creates modern React components with TypeScript, proper
        styling, and integration with GenericSuite backend APIs.

        Args:
            request: Frontend code generation request with component
            specifications.

        Returns:
            CodeGenerationResult: Generated frontend code with tests and
                styles.
        """
        return frontend_generator.generate_react_component(
            requirements=request.requirements,
            component_name=request.component_name,
            component_type=request.component_type
        )

    return Tool(generate_frontend_code, description=(
        "Generate ReactJS frontend components for GenericSuite applications. "
        "Creates modern, responsive components with TypeScript, proper "
        "styling, and integration with GenericSuite patterns and APIs."
    ))


def create_backend_code_generation_tool(kb_tool: KnowledgeBaseTool) -> Tool:
    """
    Create a Pydantic AI tool for backend code generation.

    Returns:
        Tool: Pydantic AI tool for generating backend code.
    """
    backend_generator = BackendCodeGenerator(kb_tool)

    def generate_backend_code(request: BackendCodeRequest
                              ) -> CodeGenerationResult:
        """
        Generate backend code for FastAPI, Flask, or Chalice frameworks.

        This tool creates robust backend code following GenericSuite patterns,
        including proper authentication, validation, and error handling.

        Args:
            request: Backend code generation request with framework and
                specifications.

        Returns:
            CodeGenerationResult: Generated backend code with tests and
                documentation.
        """
        return backend_generator.generate_backend_code(
            requirements=request.requirements,
            module_name=request.module_name,
            framework=request.framework,
            code_type=request.code_type
        )

    return Tool(generate_backend_code, description=(
        "Generate backend code for GenericSuite applications using FastAPI, "
        "Flask, or Chalice frameworks. Creates secure, scalable endpoints "
        "with proper authentication, validation, and GenericSuite integration."
    ))


def get_all_frontend_backend_tools(kb_tool: KnowledgeBaseTool) -> List[Tool]:
    """
    Get all frontend and backend code generation tools.

    Returns:
        List[Tool]: List of frontend and backend generation tools.
    """
    return [
        create_frontend_code_generation_tool(kb_tool),
        create_backend_code_generation_tool(kb_tool)
    ]


def get_all_agent_tools(kb_tool: KnowledgeBaseTool) -> List[Tool]:
    """
    Get all tools for the GenericSuite AI agent.

    Returns:
        List[Tool]: Complete list of agent tools.
    """
    tools = []
    tools.extend(get_all_knowledge_base_tools(kb_tool))
    tools.extend(get_all_json_generation_tools(kb_tool))
    tools.extend(get_all_python_generation_tools(kb_tool))
    tools.extend(get_all_frontend_backend_tools(kb_tool))
    return tools


def validate_search_query(query: str) -> bool:
    """
    Validate a search query for basic requirements.

    Args:
        query: Search query to validate.

    Returns:
        bool: True if query is valid, False otherwise.
    """
    if not query or not isinstance(query, str):
        return False

    # Check minimum length
    if len(query.strip()) < 3:
        return False

    # Check maximum length
    if len(query) > 1000:
        return False

    return True


def format_sources_for_attribution(sources: List[str]) -> str:
    """
    Format source paths for attribution in responses.

    Args:
        sources: List of source document paths.

    Returns:
        str: Formatted source attribution string.
    """
    if not sources:
        return "No sources available."

    if len(sources) == 1:
        return f"Source: {sources[0]}"

    formatted_sources = "\n".join([f"- {source}" for source in sources])
    return f"Sources:\n{formatted_sources}"


if __name__ == "__main__":
    # Example usage and testing
    import asyncio

    async def test_knowledge_base_tool():
        """Test the knowledge base tool functionality."""
        try:
            # Initialize tool
            kb_tool = KnowledgeBaseTool()

            # Test search
            results = kb_tool.search(
                "GenericSuite table configuration", limit=3)
            print(f"Search results: {results}")

            # Test context retrieval
            context, sources = kb_tool.get_context_for_generation(
                "How to create a GenericSuite table", max_context_length=2000
            )
            print(f"Context length: {len(context)}")
            print(f"Sources: {sources}")

        except Exception as e:
            print(f"Test error: {e}")

    asyncio.run(test_knowledge_base_tool())
