"""
FastAPI application for GenericSuite CodeGen.

This module sets up the main FastAPI application with CORS, middleware,
and all API endpoints for the GenericSuite CodeGen RAG system.
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
    FilePackage,
)
from .utilities import (
    setup_logging,
    get_app_info,
    create_correlation_id,
    log_request_response,
)
from genericsuite_codegen.agent.agent import initialize_agent, get_agent
from genericsuite_codegen.database.setup import (
    get_database_connection,
    test_database_connection,
)

# Configure logging
logger = logging.getLogger(__name__)


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
        db = get_database_connection()
        if not await test_database_connection(db):
            logger.error("Database connection failed during startup")
            raise RuntimeError("Database connection failed")

        # Initialize AI agent
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
    allowed_hosts = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    # Request logging middleware
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        """Log requests and responses with correlation IDs."""
        correlation_id = create_correlation_id()
        request.state.correlation_id = correlation_id

        # Log request
        log_request_response(
            method=request.method,
            url=str(request.url),
            correlation_id=correlation_id,
            event_type="request",
        )

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
        request: Request, exc: RequestValidationError
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

        return JSONResponse(status_code=422, content=error_response.model_dump())

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        error_response = ErrorResponse(
            error_code="HTTP_ERROR", message=exc.detail, correlation_id=correlation_id
        )

        logger.error(f"HTTP error [{correlation_id}]: {exc.status_code} - {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code, content=error_response.model_dump()
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        error_response = ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An internal server error occurred",
            details=(
                {"error": str(exc)} if os.getenv("SERVER_DEBUG", "0") == "1" else None
            ),
            correlation_id=correlation_id,
        )

        logger.error(f"Internal error [{correlation_id}]: {exc}", exc_info=True)

        return JSONResponse(status_code=500, content=error_response.model_dump())


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
        try:
            # Check database connection
            db = get_database_connection()
            db_healthy = await test_database_connection(db)

            # Check agent health
            agent = get_agent()
            agent_health = await agent.health_check()

            # Determine overall status
            status = (
                "healthy"
                if db_healthy and agent_health["status"] == "healthy"
                else "unhealthy"
            )

            return HealthResponse(
                status=status,
                timestamp=None,  # Will be set by model default
                version=get_app_info().version,
                components={
                    "database": "healthy" if db_healthy else "unhealthy",
                    "agent": agent_health["status"],
                    "model": agent_health.get("model", "unknown"),
                },
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                version=get_app_info().version,
                components={"error": str(e)},
            )

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
        try:
            # Get database stats
            db = get_database_connection()
            db_stats = {}

            try:
                # Get collection counts
                knowledge_base = db.knowledge_base
                conversations = db.ai_chatbot_conversations

                db_stats = {
                    "knowledge_base_documents": await knowledge_base.count_documents(
                        {}
                    ),
                    "conversations": await conversations.count_documents({}),
                    "connection_status": "connected",
                }
            except Exception as e:
                db_stats = {"connection_status": "error", "error": str(e)}

            # Get agent info
            agent = get_agent()
            agent_info = agent.get_model_info()

            return {
                "application": get_app_info().model_dump(),
                "database": db_stats,
                "agent": agent_info,
                "environment": {
                    "debug": os.getenv("SERVER_DEBUG", "0") == "1",
                    "cors_origins": os.getenv("CORS_ORIGINS", "").split(","),
                    "allowed_hosts": os.getenv("ALLOWED_HOSTS", "").split(","),
                },
            }

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Status check failed: {e}")

    # Agent Query Endpoints

    @app.post("/api/query", response_model=QueryResponse, tags=["Agent"])
    async def query_agent(request: QueryRequest, req: Request):
        """
        Query the AI agent.

        Args:
            request: Query request data.
            req: FastAPI request object.

        Returns:
            QueryResponse: Agent response.
        """
        correlation_id = getattr(req.state, "correlation_id", "unknown")
        return await methods.query_agent(request, correlation_id)

    @app.post("/api/query/stream", tags=["Agent"])
    async def stream_query_agent(request: QueryRequest, req: Request):
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

    @app.post("/api/conversations", response_model=Conversation, tags=["Conversations"])
    async def create_conversation(
        request: ConversationCreate,
        user_id: str = "default_user",  # In a real app, this would come from authentication
    ):
        """
        Create a new conversation.

        Args:
            request: Conversation creation request.
            user_id: User ID from authentication.

        Returns:
            Conversation: Created conversation.
        """
        return await methods.create_conversation(request, user_id)

    @app.get(
        "/api/conversations", response_model=ConversationList, tags=["Conversations"]
    )
    async def get_conversations(
        page: int = 1,
        page_size: int = 20,
        user_id: str = "default_user",  # In a real app, this would come from authentication
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
        return await methods.get_conversations(user_id, page, page_size)

    @app.get(
        "/api/conversations/{conversation_id}",
        response_model=Conversation,
        tags=["Conversations"],
    )
    async def get_conversation(
        conversation_id: str,
        user_id: str = "default_user",  # In a real app, this would come from authentication
    ):
        """
        Get a specific conversation.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID from authentication.

        Returns:
            Conversation: Conversation data.
        """
        return await methods.get_conversation(conversation_id, user_id)

    @app.put(
        "/api/conversations/{conversation_id}",
        response_model=Conversation,
        tags=["Conversations"],
    )
    async def update_conversation(
        conversation_id: str,
        request: ConversationUpdate,
        user_id: str = "default_user",  # In a real app, this would come from authentication
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
        return await methods.update_conversation(conversation_id, request, user_id)

    @app.delete("/api/conversations/{conversation_id}", tags=["Conversations"])
    async def delete_conversation(
        conversation_id: str,
        user_id: str = "default_user",  # In a real app, this would come from authentication
    ):
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID from authentication.

        Returns:
            Dict[str, str]: Deletion confirmation.
        """
        return await methods.delete_conversation(conversation_id, user_id)

    # Knowledge Base Management Endpoints

    @app.post("/api/update-knowledge-base", tags=["Knowledge Base"])
    async def update_knowledge_base(
        request: KnowledgeBaseUpdate, background_tasks: BackgroundTasks
    ):
        """
        Trigger knowledge base update.

        Args:
            request: Update request parameters.
            background_tasks: FastAPI background tasks.

        Returns:
            Dict[str, str]: Update initiation response.
        """
        return await methods.update_knowledge_base(request, background_tasks)

    @app.get(
        "/api/knowledge-base/status",
        response_model=KnowledgeBaseStatus,
        tags=["Knowledge Base"],
    )
    async def get_knowledge_base_status():
        """
        Get knowledge base status.

        Returns:
            KnowledgeBaseStatus: Current knowledge base status.
        """
        return await methods.get_knowledge_base_status()

    @app.post(
        "/api/upload-document", response_model=DocumentInfo, tags=["Knowledge Base"]
    )
    async def upload_document(file: UploadFile, description: Optional[str] = None):
        """
        Upload and process a document.

        Args:
            file: Document file to upload.
            description: Optional document description.

        Returns:
            DocumentInfo: Information about the uploaded document.
        """
        return await methods.upload_document(file, description)

    @app.get(
        "/api/knowledge-base/progress/{operation_id}",
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
        "/api/knowledge-base/statistics",
        response_model=Statistics,
        tags=["Knowledge Base"],
    )
    async def get_knowledge_base_statistics():
        """
        Get knowledge base statistics.

        Returns:
            Statistics: Knowledge base and system statistics.
        """
        return await methods.get_statistics()

    @app.post("/api/search", response_model=SearchResponse, tags=["Knowledge Base"])
    async def search_knowledge_base(query: SearchQuery):
        """
        Search the knowledge base.

        Args:
            query: Search query parameters.

        Returns:
            SearchResponse: Search results.
        """
        return await methods.search_knowledge_base(query)

    # File Generation and Download Endpoints

    @app.post(
        "/api/generate-file", response_model=GeneratedFile, tags=["File Generation"]
    )
    async def generate_file(request: FileGenerationRequest):
        """
        Generate a file from content.

        Args:
            request: File generation request.

        Returns:
            GeneratedFile: Generated file information.
        """
        return await methods.generate_file(request)

    @app.post(
        "/api/generate-package", response_model=FilePackage, tags=["File Generation"]
    )
    async def create_file_package(files: List[GeneratedFile]):
        """
        Create a package from multiple files.

        Args:
            files: List of files to package.

        Returns:
            FilePackage: File package information.
        """
        return await methods.create_file_package(files)

    @app.get("/api/download/file/{filename}", tags=["File Generation"])
    async def download_file(filename: str, content: str):
        """
        Download a generated file.

        Args:
            filename: Name of the file to download.
            content: File content (would typically be retrieved from storage).

        Returns:
            Response: File download response.
        """
        from fastapi.responses import Response
        from .utilities import sanitize_filename

        # Sanitize filename for security
        safe_filename = sanitize_filename(filename)

        # Determine content type based on file extension
        content_type = "text/plain"
        if safe_filename.endswith(".json"):
            content_type = "application/json"
        elif safe_filename.endswith(".py"):
            content_type = "text/x-python"
        elif safe_filename.endswith((".js", ".jsx")):
            content_type = "application/javascript"
        elif safe_filename.endswith((".ts", ".tsx")):
            content_type = "application/typescript"

        return Response(
            content=content,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={safe_filename}"},
        )

    @app.get("/api/download/package/{package_name}", tags=["File Generation"])
    async def download_package(package_name: str):
        """
        Download a file package as ZIP.

        Args:
            package_name: Name of the package to download.

        Returns:
            StreamingResponse: ZIP file download.
        """
        import io
        import zipfile
        from fastapi.responses import StreamingResponse
        from .utilities import sanitize_filename

        # This is a placeholder implementation
        # In a real system, you would retrieve the package from storage

        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add placeholder files (in real implementation, get from storage)
            zip_file.writestr(
                "README.md",
                "# Generated Code Package\n\nThis package contains generated code files.",
            )
            zip_file.writestr(
                "example.json", '{"message": "This is a generated JSON file"}'
            )

        zip_buffer.seek(0)

        safe_package_name = sanitize_filename(package_name)

        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={safe_package_name}.zip"
            },
        )

    @app.post(
        "/api/generate/json-config",
        response_model=GeneratedFile,
        tags=["Code Generation"],
    )
    async def generate_json_config(requirements: str, config_type: str = "table"):
        """
        Generate JSON configuration for GenericSuite.

        Args:
            requirements: Requirements for the configuration.
            config_type: Type of configuration (table, form, menu).

        Returns:
            GeneratedFile: Generated JSON configuration file.
        """
        try:
            # Use agent to generate JSON config
            agent = get_agent()
            response = await agent.generate_json_config(requirements, config_type)

            # Extract JSON from response content
            from .utilities import extract_code_blocks

            code_blocks = extract_code_blocks(response.content, "json")

            if not code_blocks:
                raise HTTPException(
                    status_code=400,
                    detail="No JSON configuration found in agent response",
                )

            json_content = code_blocks[0]["code"]
            filename = f"{config_type}_config_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

            return GeneratedFile(
                filename=filename,
                content=json_content,
                file_type="json",
                size=len(json_content.encode("utf-8")),
                description=f"Generated {config_type} configuration",
            )

        except Exception as e:
            logger.error(f"JSON config generation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate JSON config: {str(e)}"
            )

    @app.post(
        "/api/generate/python-code",
        response_model=GeneratedFile,
        tags=["Code Generation"],
    )
    async def generate_python_code(requirements: str, code_type: str = "tool"):
        """
        Generate Python code for GenericSuite.

        Args:
            requirements: Requirements for the code.
            code_type: Type of code (tool, langchain, mcp).

        Returns:
            GeneratedFile: Generated Python code file.
        """
        try:
            # Use agent to generate Python code
            agent = get_agent()
            response = await agent.generate_python_code(requirements, code_type)

            # Extract Python code from response content
            from .utilities import extract_code_blocks

            code_blocks = extract_code_blocks(response.content, "python")

            if not code_blocks:
                raise HTTPException(
                    status_code=400, detail="No Python code found in agent response"
                )

            python_content = code_blocks[0]["code"]
            filename = f"{code_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.py"

            return GeneratedFile(
                filename=filename,
                content=python_content,
                file_type="python",
                size=len(python_content.encode("utf-8")),
                description=f"Generated {code_type} Python code",
            )

        except Exception as e:
            logger.error(f"Python code generation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate Python code: {str(e)}"
            )

    @app.post(
        "/api/generate/frontend-code",
        response_model=List[GeneratedFile],
        tags=["Code Generation"],
    )
    async def generate_frontend_code(requirements: str):
        """
        Generate ReactJS frontend code.

        Args:
            requirements: Requirements for the frontend code.

        Returns:
            List[GeneratedFile]: Generated frontend code files.
        """
        try:
            # Use agent to generate frontend code
            agent = get_agent()
            response = await agent.generate_frontend_code(requirements)

            # Extract code blocks from response
            from .utilities import extract_code_blocks

            generated_files = []

            # Extract different types of code blocks
            for lang in ["jsx", "tsx", "javascript", "typescript", "css"]:
                code_blocks = extract_code_blocks(response.content, lang)

                for i, block in enumerate(code_blocks):
                    ext = "jsx" if lang in ["jsx", "tsx"] else lang[:2]
                    filename = f"component_{i+1}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{ext}"

                    generated_files.append(
                        GeneratedFile(
                            filename=filename,
                            content=block["code"],
                            file_type=lang,
                            size=len(block["code"].encode("utf-8")),
                            description=f"Generated {lang} frontend code",
                        )
                    )

            if not generated_files:
                # Fallback: create a single file with the full response
                filename = (
                    f"frontend_code_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsx"
                )
                generated_files.append(
                    GeneratedFile(
                        filename=filename,
                        content=response.content,
                        file_type="jsx",
                        size=len(response.content.encode("utf-8")),
                        description="Generated frontend code",
                    )
                )

            return generated_files

        except Exception as e:
            logger.error(f"Frontend code generation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate frontend code: {str(e)}"
            )

    @app.post(
        "/api/generate/backend-code",
        response_model=List[GeneratedFile],
        tags=["Code Generation"],
    )
    async def generate_backend_code(requirements: str, framework: str = "fastapi"):
        """
        Generate backend code for specified framework.

        Args:
            requirements: Requirements for the backend code.
            framework: Backend framework (fastapi, flask, chalice).

        Returns:
            List[GeneratedFile]: Generated backend code files.
        """
        try:
            # Use agent to generate backend code
            agent = get_agent()
            response = await agent.generate_backend_code(requirements, framework)

            # Extract Python code blocks from response
            from .utilities import extract_code_blocks

            code_blocks = extract_code_blocks(response.content, "python")

            generated_files = []

            for i, block in enumerate(code_blocks):
                filename = f"{framework}_backend_{i+1}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.py"

                generated_files.append(
                    GeneratedFile(
                        filename=filename,
                        content=block["code"],
                        file_type="python",
                        size=len(block["code"].encode("utf-8")),
                        description=f"Generated {framework} backend code",
                    )
                )

            if not generated_files:
                # Fallback: create a single file with the full response
                filename = f"{framework}_backend_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.py"
                generated_files.append(
                    GeneratedFile(
                        filename=filename,
                        content=response.content,
                        file_type="python",
                        size=len(response.content.encode("utf-8")),
                        description=f"Generated {framework} backend code",
                    )
                )

            return generated_files

        except Exception as e:
            logger.error(f"Backend code generation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate backend code: {str(e)}"
            )


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
