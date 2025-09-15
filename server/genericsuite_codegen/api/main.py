"""
FastAPI application for GenericSuite CodeGen.

This module sets up the main FastAPI application with CORS, middleware,
and all API endpoints for the GenericSuite CodeGen RAG system.
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, \
    UploadFile, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import StreamingResponse, Response

# from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from .types import (
    HealthResponse,
    ErrorResponse,
    AppInfo,
    QueryRequest,
    QueryResponse,
    ConversationCreate,
    ConversationUpdate,
    Conversation,
    ConversationList,
    KnowledgeBaseUpdate,
    KnowledgeBaseStatus,
    DocumentInfo,
    ProgressUpdate,
    Statistics,
    SearchQuery,
    SearchResponse,
    FileGenerationRequest,
    GeneratedFile,
    GeneratedFilesResponse,
    FilePackage,
    StandardGsResponse,
    StandardGsErrorResponse,
    GenerationRequest,
)
from .utilities import (
    setup_logging,
    get_app_info,
    create_correlation_id,
    log_request_response,
)
from genericsuite_codegen.database.setup import (
    # get_database_connection,
    initialize_database,
    test_database_connection,
)

DEBUG = True

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if DEBUG else logging.WARNING)


EP_PREFIX = ''
PERFORM_AGENT_HEALT_CHECK = os.getenv("PERFORM_AGENT_HEALT_CHECK", "0") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.

    Args:
        app: FastAPI application instance.
    """
    # Startup
    logger.info("Starting GenericSuite CodeGen API server...")

    try:
        # Initialize database connection
        # db = get_database_connection()
        db = initialize_database()
        if not await test_database_connection(db):
            logger.error("Database connection failed during startup")
            raise RuntimeError("Database connection failed")

        # Initialize AI agent
        if PERFORM_AGENT_HEALT_CHECK:
            from genericsuite_codegen.agent.agent import initialize_agent
            agent = initialize_agent()
            health = await agent.health_check()
            if health["status"] != "healthy":
                logger.warning(f"Agent health check failed: {health}")

        logger.info("API server startup completed successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down GenericSuite CodeGen API server...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance.
    """
    # Setup logging
    setup_logging()

    # Get application info
    app_info = get_app_info()

    # Create FastAPI app
    app = FastAPI(
        title=app_info.name,
        description=app_info.description,
        version=app_info.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add middleware
    setup_middleware(app)

    # Add exception handlers
    setup_exception_handlers(app)

    # Add routes
    setup_routes(app)

    return app


def setup_middleware(app: FastAPI) -> None:
    """
    Setup middleware for the FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    # CORS middleware
    cors_origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:3001"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Trusted host middleware
    allowed_hosts = os.getenv("ALLOWED_HOSTS",
                              "localhost,127.0.0.1").split(",")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    # Request logging middleware
    @app.middleware("http")
    async def request_logging_middleware(
        request: Request,
        call_next,
    ):
        """Log requests and responses with correlation IDs."""
        correlation_id = create_correlation_id()
        request.state.correlation_id = correlation_id

        if DEBUG:
            api_requests_logger = logging.getLogger("api.requests")
            api_requests_logger.setLevel(logging.INFO)

        # Log request
        log_request_response(
            method=request.method,
            url=str(request.url),
            correlation_id=correlation_id,
            event_type="request",
        )

        logger.info(f"request_logging_middleware | Request: {request}")

        # Process request
        response = await call_next(request)

        # Log response
        log_request_response(
            method=request.method,
            url=str(request.url),
            correlation_id=correlation_id,
            event_type="response",
            status_code=response.status_code,
        )

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Setup exception handlers for the FastAPI application.

    Args:
        app: FastAPI application instance.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        """Handle request validation errors."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        error_response = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors(), "body": exc.body},
            correlation_id=correlation_id,
        )

        logger.error(f"Validation error [{correlation_id}]: {exc.errors()}")

        # return JSONResponse(status_code=422,
        #                     content=error_response.model_dump())
        return Response(status_code=422, content=str(error_response))

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ):
        """Handle HTTP exceptions."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        error_response = ErrorResponse(
            error_code="HTTP_ERROR", message=exc.detail,
            correlation_id=correlation_id
        )

        logger.error(
            f"HTTP error [{correlation_id}]: {exc.status_code}"
            f" - {exc.detail}"
        )

        # return JSONResponse(
        #     status_code=exc.status_code, content=error_response.model_dump()
        # )
        return Response(status_code=exc.status_code,
                        content=str(error_response))

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ):
        """Handle general exceptions."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        error_response = ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An internal server error occurred",
            details=(
                {"error": str(exc)} if os.getenv(
                    "SERVER_DEBUG", "0") == "1" else None
            ),
            correlation_id=correlation_id,
        )

        logger.error(
            f"Internal error [{correlation_id}]: {exc}", exc_info=True)

        # return JSONResponse(status_code=500,
        #                     content=error_response.model_dump())
        return Response(status_code=500, content=str(error_response))


def result_wrapper(
    result: Union[StandardGsResponse, StandardGsErrorResponse],
) -> Union[HTTPException, StandardGsResponse]:
    if result.error:
        raise HTTPException(
            status_code=result.status_code,
            detail=result.error_message,
        )
    return result


def setup_routes(app: FastAPI) -> None:
    """
    Setup routes for the FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    from .endpoint_methods import get_endpoint_methods

    # Get endpoint methods instance
    methods = get_endpoint_methods()

    # Health check endpoints
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Health check endpoint.

        Returns:
            HealthResponse: Application health status.
        """
        return methods.health_check_endpoint()

    @app.get("/", response_model=AppInfo, tags=["Info"])
    async def root():
        """
        Root endpoint with application information.

        Returns:
            AppInfo: Application information.
        """
        return get_app_info()

    @app.get("/status", response_model=Dict[str, Any], tags=["Health"])
    async def status():
        """
        Detailed status endpoint.

        Returns:
            Dict[str, Any]: Detailed application status.
        """
        result = result_wrapper(methods.status_endpoint())
        return result.result

    # Agent Query Endpoints

    @app.post(
        EP_PREFIX + "/query",
        response_model=QueryResponse,
        tags=["Agent"])
    async def query_agent(
        req: Request,
        request: Optional[QueryRequest] = Body(default=None),
    ):
        """
        Query the AI agent.

        Args:
            request: Query request data.
            req: FastAPI request object.

        Returns:
            QueryResponse: Agent response.
        """
        correlation_id = getattr(req.state, "correlation_id", "unknown")
        result = result_wrapper(
            await methods.query_agent(request, correlation_id))
        logger.info(f"/query | query_agent | result.result: {result}")
        logger.info(f"dict(result.result): {dict(result.result)}")
        # return result.result
        return dict(result.result)

    @app.post(EP_PREFIX + "/query/stream", tags=["Agent"])
    async def stream_query_agent(
        req: Request,
        request: Optional[QueryRequest] = Body(default=None),
    ):
        """
        Stream AI agent query response.

        Args:
            request: Query request data.
            req: FastAPI request object.

        Returns:
            StreamingResponse: Streaming agent response.
        """
        correlation_id = getattr(req.state, "correlation_id", "unknown")

        return StreamingResponse(
            methods.stream_agent_query(request, correlation_id),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Correlation-ID": correlation_id,
            },
        )

    # Conversation Management Endpoints

    @app.post(EP_PREFIX + "/conversations", response_model=Conversation,
              tags=["Conversations"])
    async def create_conversation(
        request: Optional[ConversationCreate] = Body(default=None),
        user_id: str = "default_user",  # In a real app, this would come from
        # authentication
    ):
        """
        Create a new conversation.

        Args:
            request: Conversation creation request.
            user_id: User ID from authentication.

        Returns:
            Conversation: Created conversation.
        """
        result = result_wrapper(
            await methods.create_conversation(
                request, user_id))
        return result.result

    @app.get(
        EP_PREFIX + "/conversations", response_model=ConversationList,
        tags=["Conversations"]
    )
    async def get_conversations(
        page: int = 1,
        page_size: int = 20,
        user_id: str = "default_user",  # In a real app, this would come from
        # authentication
    ):
        """
        Get user conversations with pagination.

        Args:
            page: Page number (1-based).
            page_size: Items per page.
            user_id: User ID from authentication.

        Returns:
            ConversationList: Paginated conversation list.
        """
        result = result_wrapper(
            await methods.get_conversations(user_id, page, page_size)
        )
        return result.result

    @app.get(
        EP_PREFIX + "/conversations/{conversation_id}",
        response_model=Conversation,
        tags=["Conversations"],
    )
    async def get_conversation(
        conversation_id: str,
        user_id: str = "default_user",  # In a real app, this would come from
        # authentication
    ):
        """
        Get a specific conversation.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID from authentication.

        Returns:
            Conversation: Conversation data.
        """
        result = result_wrapper(
            await methods.get_conversation(conversation_id, user_id)
        )
        return result.result

    @app.put(
        EP_PREFIX + "/conversations/{conversation_id}",
        response_model=Conversation,
        tags=["Conversations"],
    )
    async def update_conversation(
        conversation_id: str,
        request: Optional[ConversationUpdate] = Body(default=None),
        user_id: str = "default_user",  # In a real app, this would come from
        # authentication
    ):
        """
        Update a conversation.

        Args:
            conversation_id: Conversation ID.
            request: Update request.
            user_id: User ID from authentication.

        Returns:
            Conversation: Updated conversation.
        """
        result = result_wrapper(
            await methods.update_conversation(conversation_id, request,
                                              user_id)
        )
        return result.result

    @app.delete(EP_PREFIX + "/conversations/{conversation_id}",
                tags=["Conversations"])
    async def delete_conversation(
        conversation_id: str,
        user_id: str = "default_user",  # In a real app, this would come from
        # authentication
    ):
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID from authentication.

        Returns:
            Dict[str, str]: Deletion confirmation.
        """
        result = result_wrapper(
            await methods.delete_conversation(conversation_id, user_id)
        )
        return result.result

    # Knowledge Base Management Endpoints

    @app.post("/update-knowledge-base", tags=["Knowledge Base"])
    async def update_knowledge_base(
        background_tasks: BackgroundTasks,
        request: Optional[KnowledgeBaseUpdate] = Body(default=None),
    ):
        """
        Trigger knowledge base update.

        Args:
            request: Update request parameters.
            background_tasks: FastAPI background tasks.

        Returns:
            Dict[str, str]: Update initiation response.
        """
        # Use default values if no request body provided
        if request is None:
            request = KnowledgeBaseUpdate()

        logger.info(f"Received update request: {request}")
        result = result_wrapper(
            await methods.update_knowledge_base(request, background_tasks)
            # await methods.update_knowledge_base(request)
        )
        return result.result

    @app.get(
        EP_PREFIX + "/knowledge-base/status",
        response_model=KnowledgeBaseStatus,
        tags=["Knowledge Base"],
    )
    async def get_knowledge_base_status():
        """
        Get knowledge base status.

        Returns:
            KnowledgeBaseStatus: Current knowledge base status.
        """
        result = result_wrapper(await methods.get_knowledge_base_status())
        return result.result

    @app.post(
        EP_PREFIX + "/upload-document", response_model=DocumentInfo,
        tags=["Knowledge Base"]
    )
    async def upload_document(file: UploadFile,
                              description: Optional[str] = None):
        """
        Upload and process a document.

        Args:
            file: Document file to upload.
            description: Optional document description.

        Returns:
            DocumentInfo: Information about the uploaded document.
        """
        result = result_wrapper(await methods.upload_document(file,
                                                              description))
        return result.result

    @app.get(
        EP_PREFIX + "/knowledge-base/progress/{operation_id}",
        response_model=ProgressUpdate,
        tags=["Knowledge Base"],
    )
    async def get_operation_progress(operation_id: str):
        """
        Get progress of a long-running operation.

        Args:
            operation_id: Operation identifier.

        Returns:
            ProgressUpdate: Operation progress information.
        """
        # This would be implemented with a proper progress tracking system
        # For now, return a placeholder response
        return ProgressUpdate(
            operation_id=operation_id,
            status="completed",
            progress=1.0,
            message="Operation completed",
            started_at=datetime.utcnow(),
        )

    @app.get(
        EP_PREFIX + "/knowledge-base/statistics",
        response_model=Statistics,
        tags=["Knowledge Base"],
    )
    async def get_knowledge_base_statistics():
        """
        Get knowledge base statistics.

        Returns:
            Statistics: Knowledge base and system statistics.
        """
        result = result_wrapper(await methods.get_statistics())
        return result.result

    @app.post(
        EP_PREFIX + "/search",
        response_model=SearchResponse,
        tags=["Knowledge Base"]
    )
    async def search_knowledge_base(query: SearchQuery):
        """
        Search the knowledge base.

        Args:
            query: Search query parameters.

        Returns:
            SearchResponse: Search results.
        """
        result = result_wrapper(await methods.search_knowledge_base(query))
        return result.result

    @app.post(EP_PREFIX + "/knowledge-base/clean", tags=["Knowledge Base"])
    async def clean_knowledge_base():
        """
        Clean all vectors from the knowledge base.

        Returns:
            Dict[str, str]: Cleanup confirmation.
        """
        result = result_wrapper(await methods.clean_knowledge_base())
        return result.result

    # File Generation and Download Endpoints

    @app.post(
        EP_PREFIX + "/generate-file",
        response_model=GeneratedFile,
        tags=["File Generation"]
    )
    async def generate_file(
        request:  Optional[FileGenerationRequest] = Body(default=None),
    ):
        """
        Generate a file from content.

        Args:
            request: File generation request.

        Returns:
            GeneratedFile: Generated file information.
        """
        result = result_wrapper(await methods.generate_file(request))
        return result.result

    @app.post(
        EP_PREFIX + "/generate-package",
        response_model=FilePackage,
        tags=["File Generation"]
    )
    async def create_file_package(files: List[GeneratedFile]):
        """
        Create a package from multiple files.

        Args:
            files: List of files to package.

        Returns:
            FilePackage: File package information.
        """
        result = result_wrapper(await methods.create_file_package(files))
        return await result.result

    @app.get(
        EP_PREFIX + "/download/file/{filename}",
        tags=["File Generation"]
    )
    async def download_file(filename: str, content: str):
        """
        Download a generated file.

        Args:
            filename: Name of the file to download.
            content: File content (would typically be retrieved from storage).

        Returns:
            Response: File download response.
        """
        result = result_wrapper(methods.get_filename_data(filename))
        return Response(
            content=content,
            media_type=result.result["content_type"],
            headers={
                "Content-Disposition":
                f"attachment; filename={result.result['safe_filename']}"
            },
        )

    @app.get(
        EP_PREFIX + "/download/package/{package_name}",
        tags=["File Generation"]
    )
    async def download_package(package_name: str):
        """
        Download a file package as ZIP.

        Args:
            package_name: Name of the package to download.

        Returns:
            StreamingResponse: ZIP file download.
        """
        import io

        result = result_wrapper(
            methods.download_package_endpoint(package_name))
        return StreamingResponse(
            io.BytesIO(result.result["zip_buffer"].read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename="
                + f"${result.result['safe_package_name']}.zip"
            },
        )

    @app.post(
        EP_PREFIX + "/generate/json-config",
        response_model=GeneratedFilesResponse,
        tags=["Code Generation"],
    )
    async def generate_json_config(
        request: Optional[GenerationRequest] = Body(default=None),
    ):
        """
        Generate JSON configuration for GenericSuite.

        Args:
            requirements: Requirements for the configuration.
            config_type: Type of configuration (table, form, menu).

        Returns:
            GeneratedFile: Generated JSON configuration file.
        """
        logger.info(
            f"ENDPOINT >> generate_json_config | Received request: {request}")
        result = await methods.generate_json_config_endpoint(
            request.requirements,
            request.table_name,
            request.config_type,
        )
        logger.info(
            f"ENDPOINT >> generate_json_config | Result: {result}")
        result = result_wrapper(result)
        return result.result

    @app.post(
        EP_PREFIX + "/generate/python-code",
        response_model=GeneratedFilesResponse,
        tags=["Code Generation"],
    )
    async def generate_python_code(
        request: Optional[GenerationRequest] = Body(default=None),
    ):
        """
        Generate Python code for GenericSuite.

        Args:
            requirements: Requirements for the code.
            code_type: Type of code (tool, langchain, mcp).

        Returns:
            GeneratedFile: Generated Python code file.
        """
        result = result_wrapper(
            await methods.generate_python_code_endpoint(
                request.requirements,
                request.tool_name,
                request.description,
                request.type,
            )
        )
        return result.result

    @app.post(
        EP_PREFIX + "/generate/frontend-code",
        response_model=GeneratedFilesResponse,
        tags=["Code Generation"],
    )
    async def generate_frontend_code(
        request: Optional[GenerationRequest] = Body(default=None),
    ):
        """
        Generate ReactJS frontend code.

        Args:
            requirements: Requirements for the frontend code.

        Returns:
            List[GeneratedFile]: Generated frontend code files.
        """
        result = result_wrapper(
            await methods.generate_frontend_code_endpoint(request.requirements)
        )
        return result.result

    @app.post(
        EP_PREFIX + "/generate/backend-code",
        response_model=GeneratedFilesResponse,
        tags=["Code Generation"],
    )
    async def generate_backend_code(
        request: Optional[GenerationRequest] = Body(default=None),
    ):
        """
        Generate backend code for specified framework.

        Args:
            requirements: Requirements for the backend code.
            framework: Backend framework (fastapi, flask, chalice).

        Returns:
            List[GeneratedFile]: Generated backend code files.
        """
        result = result_wrapper(
            await methods.generate_backend_code_endpoint(
                request.requirements,
                request.framework
            )
        )
        return result.result


# Create the application instance
app = create_app()


def run_server():
    """Run the FastAPI server with uvicorn."""
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    debug = os.getenv("SERVER_DEBUG", "0") == "1"

    uvicorn.run(
        "genericsuite_codegen.api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info",
        access_log=True,
    )


if __name__ == "__main__":
    run_server()
