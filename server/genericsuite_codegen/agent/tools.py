"""
Knowledge base search and code generation tools for the
GenericSuite CodeGen AI agent.

This module provides tools for vector similarity search, context retrieval,
source attribution and Python/React code generation.
"""
from typing import List

from pydantic_ai import Tool

from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_error,
)

from genericsuite_codegen.agent.types import (
    KnowledgeBaseSearchResults,
    KnowledgeBaseQuery,
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
from genericsuite_codegen.agent.agent_super import \
    get_tool_context_default_max_length
from genericsuite_codegen.agent.document_retrieval_tool import \
    DocumentRetrievalTool
from genericsuite_codegen.agent.enhanced_search_types import (
    DocumentRetrievalRequest,
    DocumentRetrievalResponse,
    BatchDocumentRetrievalRequest,
    BatchDocumentRetrievalResponse,
    DocumentRetrievalError,
)
from genericsuite_codegen.agent.tool_knowledge_base import KnowledgeBaseTool

from genericsuite_codegen.agent.tool_backend_code_generator import \
    BackendCodeGenerator
from genericsuite_codegen.agent.tool_frontend_code_generator import \
    FrontendCodeGenerator
from genericsuite_codegen.agent.tool_python_code_generator import \
    PythonCodeGenerator
from genericsuite_codegen.agent.tool_json_config_generator import \
    JSONConfigGenerator

DEBUG = True

TOOL_CONTEXT_DEFAULT_MAX_LENGTH = get_tool_context_default_max_length()

# Utility functions for tool integration


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
        context, sources, raw_results = kb_tool.get_context_for_generation(
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
        _ = DEBUG and log_debug(
            f"Generating JSON configuration for {request.config_type}")

        table_name = request.table_name or "generated_table"
        tables = json_generator.generate_table_config(
            requirements=request.requirements,
            table_name=table_name,
            include_validation=request.include_validation
        )
        config_name = request.table_name or "generated_form"
        forms = json_generator.generate_form_config(
            requirements=request.requirements,
            config_name=config_name
        )
        return JSONConfigResult(
            records=(
                tables.records + forms.records
            )
        )

        # if request.config_type == "table":
        #     table_name = request.table_name or "generated_table"
        #     return json_generator.generate_table_config(
        #         requirements=request.requirements,
        #         table_name=table_name,
        #         include_validation=request.include_validation
        #     )
        # elif request.config_type == "form":
        #     config_name = request.table_name or "generated_form"
        #     return json_generator.generate_form_config(
        #         requirements=request.requirements,
        #         config_name=config_name
        #     )
        # else:
        #     # For other config types, use table as default with modifications
        #     config_name = request.table_name \
        #         or f"generated_{request.config_type}"
        #     result = json_generator.generate_table_config(
        #         requirements=request.requirements,
        #         table_name=config_name,
        #         include_validation=request.include_validation
        #     )
        #     result.config_type = request.config_type
        #     return result

    return Tool(generate_json_configuration, description=(
        "Generate GenericSuite JSON configurations for tables, forms, menus, "
        "endpoints, and other components. Creates valid configurations with "
        "proper field definitions, validation rules, and UI settings "
        "following GenericSuite patterns."
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


def create_document_retrieval_tool() -> Tool:
    """
    Create a Pydantic AI tool for document retrieval from local storage.

    Returns:
        Tool: Pydantic AI tool for retrieving documents.
    """
    doc_retrieval_tool = DocumentRetrievalTool()

    def retrieve_document_from_local_storage(
        request: DocumentRetrievalRequest
    ) -> DocumentRetrievalResponse:
        """
        Retrieve a complete document from local storage.

        This tool allows the agent to access full document content from the
        local_repo_files directory, providing complete GenericSuite knowledge
        base articles for accurate code generation.

        Args:
            request: Document retrieval request with path and options.

        Returns:
            DocumentRetrievalResponse: Retrieved document or error information.
        """
        try:
            # Validate request
            if not request.document_path:
                return DocumentRetrievalResponse(
                    success=False,
                    error_message="Document path is required",
                    metadata={"error_code": "MISSING_PATH"}
                )

            # Check file size limit (convert MB to bytes)
            max_size_bytes = request.max_size_mb * 1024 * 1024

            # Get document metadata first to check size
            metadata = doc_retrieval_tool.get_document_metadata(
                request.document_path)

            if not metadata.exists:
                return DocumentRetrievalResponse(
                    success=False,
                    error_message="Document not found: "
                    f"{request.document_path}",
                    metadata={
                        "error_code": "FILE_NOT_FOUND",
                        "requested_path": request.document_path
                    }
                )

            if not metadata.is_readable:
                error_msg = (metadata.error_message or
                             "Document is not readable (possibly binary)")
                return DocumentRetrievalResponse(
                    success=False,
                    error_message=error_msg,
                    metadata={
                        "error_code": "NOT_READABLE",
                        "is_binary": metadata.is_binary,
                        "file_type": metadata.file_type
                    }
                )

            if metadata.size > max_size_bytes:
                return DocumentRetrievalResponse(
                    success=False,
                    error_message=(f"Document too large: {metadata.size} "
                                   f"bytes (limit: {max_size_bytes} bytes)"),
                    metadata={
                        "error_code": "FILE_TOO_LARGE",
                        "file_size": metadata.size,
                        "size_limit": max_size_bytes
                    }
                )

            # Retrieve the document
            document = doc_retrieval_tool.retrieve_document(
                request.document_path)

            # Prepare response data
            document_data = {
                "path": document.path,
                "content": document.content,
                "file_type": document.file_type,
                "size": document.size,
                "last_modified": document.last_modified.isoformat(),
                "encoding": document.encoding,
                "is_binary": document.is_binary
            }

            if request.include_metadata:
                document_data["metadata"] = document.metadata

            return DocumentRetrievalResponse(
                success=True,
                document=document_data,
                metadata={
                    "retrieval_successful": True,
                    "content_length": len(document.content),
                    "encoding_used": document.encoding
                }
            )

        except DocumentRetrievalError as e:
            log_error(f"Document retrieval error: {e}")
            return DocumentRetrievalResponse(
                success=False,
                error_message=str(e),
                metadata={
                    "error_code": getattr(e, 'error_code', 'RETRIEVAL_ERROR'),
                    "error_type": "DocumentRetrievalError"
                }
            )
        except Exception as e:
            log_error(f"Unexpected error in document retrieval: {e}")
            return DocumentRetrievalResponse(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                metadata={
                    "error_code": "UNEXPECTED_ERROR",
                    "error_type": type(e).__name__
                }
            )

    return Tool(retrieve_document_from_local_storage, description=(
        "Retrieve complete document content from local storage. Use this tool "
        "to access full GenericSuite knowledge base articles, configuration "
        "examples, and code samples from the local_repo_files directory. "
        "Provide the document path returned by knowledge base search results "
        "to get the complete content for accurate code generation."
    ))


def create_batch_document_retrieval_tool() -> Tool:
    """
    Create a Pydantic AI tool for batch document retrieval from local storage.

    Returns:
        Tool: Pydantic AI tool for retrieving multiple documents.
    """
    doc_retrieval_tool = DocumentRetrievalTool()

    def retrieve_multiple_documents_from_local_storage(
        request: BatchDocumentRetrievalRequest
    ) -> BatchDocumentRetrievalResponse:
        """
        Retrieve multiple documents from local storage in a single operation.

        This tool allows the agent to efficiently retrieve multiple documents
        at once, useful when the knowledge base search returns several relevant
        documents that need to be accessed for comprehensive code generation.

        Args:
            request: Batch document retrieval request with paths and options.

        Returns:
            BatchDocumentRetrievalResponse: Results of batch retrieval
            operation.
        """
        try:
            successful_retrievals = []
            failed_retrievals = []
            max_size_bytes = request.max_size_mb * 1024 * 1024

            _ = DEBUG and log_debug(
                "Starting batch retrieval of "
                f"{len(request.document_paths)} documents")

            for document_path in request.document_paths:
                try:
                    # Get metadata first to check size and readability
                    metadata = doc_retrieval_tool.get_document_metadata(
                        document_path)

                    if not metadata.exists:
                        failed_retrievals.append({
                            "path": document_path,
                            "error_message": "Document not found",
                            "error_code": "FILE_NOT_FOUND"
                        })
                        continue

                    if not metadata.is_readable:
                        error_msg = metadata.error_message or \
                            "Document is not readable"
                        failed_retrievals.append({
                            "path": document_path,
                            "error_message": error_msg,
                            "error_code": "NOT_READABLE"
                        })
                        continue

                    if metadata.size > max_size_bytes:
                        failed_retrievals.append({
                            "path": document_path,
                            "error_message":
                            f"Document too large: {metadata.size} bytes",
                            "error_code": "FILE_TOO_LARGE"
                        })
                        continue

                    # Retrieve the document
                    document = doc_retrieval_tool.retrieve_document(
                        document_path)

                    # Prepare document data
                    document_data = {
                        "path": document.path,
                        "content": document.content,
                        "file_type": document.file_type,
                        "size": document.size,
                        "last_modified": document.last_modified.isoformat(),
                        "encoding": document.encoding,
                        "is_binary": document.is_binary
                    }

                    if request.include_metadata:
                        document_data["metadata"] = document.metadata

                    successful_retrievals.append(document_data)

                except DocumentRetrievalError as e:
                    failed_retrievals.append({
                        "path": document_path,
                        "error_message": str(e),
                        "error_code": getattr(e, 'error_code',
                                              'RETRIEVAL_ERROR')
                    })

                    if not request.continue_on_error:
                        break

                except Exception as e:
                    failed_retrievals.append({
                        "path": document_path,
                        "error_message": f"Unexpected error: {str(e)}",
                        "error_code": "UNEXPECTED_ERROR"
                    })

                    if not request.continue_on_error:
                        break

            return BatchDocumentRetrievalResponse(
                successful_retrievals=successful_retrievals,
                failed_retrievals=failed_retrievals,
                total_requested=len(request.document_paths),
                total_successful=len(successful_retrievals),
                total_failed=len(failed_retrievals),
                metadata={
                    "batch_completed": True,
                    "continue_on_error": request.continue_on_error,
                    "max_size_mb": request.max_size_mb
                }
            )

        except Exception as e:
            log_error(f"Unexpected error in batch document retrieval: {e}")
            return BatchDocumentRetrievalResponse(
                successful_retrievals=[],
                failed_retrievals=[{
                    "path": "batch_operation",
                    "error_message": f"Batch operation failed: {str(e)}",
                    "error_code": "BATCH_OPERATION_ERROR"
                }],
                total_requested=len(request.document_paths),
                total_successful=0,
                total_failed=len(request.document_paths),
                metadata={
                    "batch_completed": False,
                    "error_type": type(e).__name__
                }
            )

    return Tool(retrieve_multiple_documents_from_local_storage, description=(
        "Retrieve multiple documents from local storage in a single operation."
        " Use this tool when you need to access several documents at once for "
        "comprehensive code generation. Provide a list of document paths "
        "returned by knowledge base search results to get all the complete "
        "content efficiently."
    ))


def get_all_document_retrieval_tools() -> List[Tool]:
    """
    Get all document retrieval tools for agent integration.

    Returns:
        List[Tool]: List of document retrieval tools.
    """
    return [
        create_document_retrieval_tool(),
        create_batch_document_retrieval_tool()
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
    tools.extend(get_all_document_retrieval_tools())
    return tools


def validate_search_query(
        query: str,
        max_length: int = TOOL_CONTEXT_DEFAULT_MAX_LENGTH) -> bool:
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
    if len(query) > max_length:
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
            context, sources, raw_results = \
                kb_tool.get_context_for_generation(
                    "How to create a GenericSuite table",
                    max_context_length=None
                )
            print(f"Context length: {len(context)}")
            print(f"Sources: {sources}")

        except Exception as e:
            print(f"Test error: {e}")

    asyncio.run(test_knowledge_base_tool())
