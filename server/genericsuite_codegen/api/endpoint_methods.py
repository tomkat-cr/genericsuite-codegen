"""
Core endpoint implementations for the FastAPI application.

This module contains the implementation logic for all API endpoints,
separated from the route definitions for better organization and testing.
"""

from typing import Dict, Any, List, Optional

from genericsuite_codegen.api.types import (
    QueryRequest,
    QueryResponse,
    KnowledgeBaseUpdate,
    KnowledgeBaseStatus,
    DocumentInfo,
    FileGenerationRequest,
    GeneratedFile,
    GeneratedFilesResponse,
    FilePackage,
    SearchQuery,
    # SearchResponse,
    Statistics,
    HealthResponse,
    UpdateSettingsRequest,
)
from genericsuite_codegen.agent.types import (
    AgentModel,
)
from genericsuite_codegen.api.types import (
    KnowledgeBaseStatistics,
    ConversationStatistics,
    SystemStatistics,
)

from genericsuite_codegen.utilities import (
    std_error_response,
    std_response,
    sanitize_filename,
    get_content_type,
    extract_code_blocks,
    get_app_info,
    local_path_to_url,
    get_utcnow_fmt,
    get_utcnow,
)
from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_error,
)
from genericsuite_codegen.utilities.env_vars import get_envvar
from genericsuite_codegen.agent.agent import (
    get_agent,
    QueryRequest as AgentQueryRequest
)
from genericsuite_codegen.database.setup import (
    get_database_connection,
    test_database_connection,
)

from genericsuite_codegen.document_processing.ingestion import (
    IngestionResult,
    IngestionStatistics,
    IngestionProgress,
    IngestionStatus,
    save_progress_to_file,
    load_progress_from_file,
)
from genericsuite_codegen.conversations.service import \
    ConversationsService
from genericsuite_codegen.settings.manager import \
    SettingsManager


DEBUG = False


class EndpointMethods():
    """
    Core endpoint implementation methods.

    This class contains all the business logic for API endpoints,
    separated from FastAPI route definitions for better testability.
    """

    def __init__(self):
        """Initialize endpoint methods."""
        self.db = get_database_connection()
        self.agent = get_agent()

    # Agent Query Methods

    async def query_agent(
        self,
        request: QueryRequest,
        correlation_id: str,
        user_id: str,
        translate_path: bool = False,
    ) -> Dict[str, Any]:
        """
        Process an agent query request.

        Args:
            request: Query request data.
            correlation_id: Request correlation ID.
            translate_path: Whether to translate the path to the local path.
        Returns:
            QueryResponse: Agent response.

        Raises:
            HTTPException: If query processing fails.
        """
        try:
            _ = DEBUG and log_debug(
                "Processing agent query "
                f"[{correlation_id}]: {request.query[:100]}...")

            # Validate request
            if not request.query or not request.query.strip():
                return std_error_response(
                    status_code=400,
                    detail="Query cannot be empty"
                )

            # If no conversation_id provided, create a new conversation
            conversation = ConversationsService(request.conversation_id)
            await conversation.init(query=request.query, user_id=user_id)

            # Convert API request to agent request
            agent_request = AgentQueryRequest(
                query=request.query,
                task_type=request.task_type.value,
                framework=(
                    request.framework.value if request.framework
                    else None),
                context_limit=request.context_limit,
                include_sources=request.include_sources,
            )

            # Process query with agent
            agent_response = await self.agent.query(
                request=agent_request,
                context=conversation.context)

            _ = DEBUG and log_debug(f">>> Agent response: {agent_response}")

            sources = ([local_path_to_url(source)
                       for source in agent_response.sources]
                       if translate_path else agent_response.sources)
            content = (local_path_to_url(agent_response.content, False)
                       if translate_path else agent_response.content)

            # Convert agent response to API response
            response = QueryResponse(
                content=content,
                sources=sources,
                task_type=request.task_type,
                model_used=agent_response.model_used,
                token_usage=agent_response.token_usage,
                conversation_id=conversation.conversation_id,
            )

            # Save message to database
            await conversation.save_message(
                query=request.query,
                content=content,
                sources=sources,
                task_type=request.task_type,
                model_used=agent_response.model_used,
                token_usage=agent_response.token_usage,
            )

            _ = DEBUG and log_debug(
                f"Query processed successfully [{correlation_id}]")
            return std_response(result=response)

        except Exception as e:
            log_error(f"Query processing failed [{correlation_id}]: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Query processing failed: {str(e)}"
            )

    async def stream_agent_query(
        self,
        request: QueryRequest,
        correlation_id: str,
        user_id: str,
    ):
        """
        Stream agent query response for long-running queries.

        Args:
            request: Query request data.
            correlation_id: Request correlation ID.

        Yields:
            str: Streaming response chunks.
        """
        try:
            _ = DEBUG and log_debug(
                f"Starting streaming query [{correlation_id}]:"
                f" {request.query[:100]}...")

            # For now, we'll implement basic streaming by yielding the full
            # response.
            # TODO: In a full implementation, this would integrate with
            # the agent's streaming capabilities.
            result = await self.query_agent(
                request=request,
                correlation_id=correlation_id,
                user_id=user_id
            )
            if result.error:
                yield f"data: ERROR: {result.details}\n\n"
                return

            # Yield response in chunks
            content = result.result.content
            chunk_size = 100

            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                yield f"data: {chunk}\n\n"

            # Send completion signal
            yield "data: [DONE]\n\n"

        except Exception as e:
            log_error(f"Streaming query failed [{correlation_id}]: {e}")
            yield f"data: ERROR: {str(e)}\n\n"

    # Knowledge Base Management Methods

    async def schedule_update_knowledge_base(
        self,
        request: KnowledgeBaseUpdate,
    ) -> Dict[str, str]:
        """
        Schedule knowledge base update.

        Args:
            request: Knowledge base update request.

        Returns:
            Dict[str, str]: Update initiation response.
        """
        progress = IngestionProgress(
            status=IngestionStatus.SCHEDULED,
            current_step="Knowledge base update scheduled",
            total_steps=6,  # clone, process, chunk, embed, store, complete
            completed_steps=0,
            repository_url=request.repository_url,
            force_refresh=request.force_refresh
        )
        save_progress_to_file(progress)
        return std_response(
            result=IngestionResult(
                success=True,
                status="Knowledge base update scheduled",
                statistics=IngestionStatistics(
                    total_documents=0,
                    total_chunks=0,
                    total_embeddings=0,
                    duration_seconds=0,
                ),
                progress=IngestionProgress(
                    status=IngestionStatus.CLONING,
                    current_step="Knowledge base update scheduled",
                    total_steps=1,
                    completed_steps=0,
                )
            ).to_dict(),
        )

    async def update_knowledge_base(self) -> Dict[str, str]:
        """
        Verify if update is scheduled.

        Args:
            request: Update request.
        """
        progress = load_progress_from_file()
        _ = DEBUG and log_debug(f"Knowledge base update progress: {progress}")
        _ = DEBUG and log_debug(f"Progress status: {progress.status}")
        _ = DEBUG and log_debug(
            f"IngestionStatus.SCHEDULED: {IngestionStatus.SCHEDULED.value}")
        if not progress or progress.status != IngestionStatus.SCHEDULED.value:
            return std_response(result=None)
        return await self.update_knowledge_base_run(progress)

    async def update_knowledge_base_run(
        self,
        progress: IngestionProgress,
    ) -> Dict[str, str]:
        """
        Trigger knowledge base update.

        Args:
            progress: Ingestion progress.

        Returns:
            Dict[str, str]: Update initiation response.
        """
        try:
            # Start background update task
            operation_id = f"kb_update_{get_utcnow_fmt()}"
            _ = DEBUG and log_debug(
                f"Knowledge base update started [{operation_id}]"
                " in BACKGROUND...")
            return std_response(
                result=await self._update_knowledge_base_background(
                    operation_id, progress
                ))

        except Exception as e:
            log_error(f"Failed to start knowledge base update: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to start update: {str(e)}"
            )

    async def get_operation_progress(
        self,
    ) -> Dict[str, Any]:
        """
        Get document processing operation progress.
        """
        from genericsuite_codegen.document_processing.ingestion import \
            get_ingestion_progress
        result = get_ingestion_progress()
        return std_response(result=result)

    async def get_knowledge_base_status(self) -> Dict[str, str]:
        """
        Get knowledge base status.

        Returns:
            Dict[str, str]: Current status as result=KnowledgeBaseStatus().
        """
        try:
            knowledge_base = self.db.database.knowledge_base

            # Get document and chunk counts
            document_count = knowledge_base.count_documents({})

            # Get unique file count (chunks from same file)
            pipeline = [
                {"$group": {"_id": "$path"}},
                {"$count": "unique_files"}
            ]

            unique_files_result = knowledge_base.aggregate(pipeline) \
                .to_list(1)
            unique_files = unique_files_result[0]["unique_files"] \
                if unique_files_result else 0

            # Get repository info from environment
            repository_url = get_envvar("REMOTE_REPO_URL", "")
            repository_branch = get_envvar("REMOTE_REPO_BRANCH", "")

            return std_response(
                result=KnowledgeBaseStatus(
                    status="idle",  # TODO: This would be tracked in a real
                                    # implementation
                    document_count=unique_files,
                    chunk_count=document_count,
                    repository_url=repository_url,
                    repository_branch=repository_branch,
                    last_update=None  # TODO: This would be tracked in database
                )
            )

        except Exception as e:
            log_error(f"Failed to get knowledge base status: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to get status: {str(e)}"
            )

    async def upload_document(
        self,
        file: any,
        description: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Upload and process a document.

        Args:
            file: Uploaded file.
            description: Optional description.

        Returns:
            Dict[str, str]: Document information as result=DocumentInfo().
        """
        try:
            # Read file content
            content = await file.read()

            # TODO: Process document (this would integrate with document
            # processing pipeline)
            # For now, we'll create a placeholder implementation

            document_info = DocumentInfo(
                id=f"doc_{get_utcnow_fmt()}",
                filename=file.filename or "unknown",
                file_type=file.content_type or "unknown",
                size=len(content),
                upload_date=get_utcnow(),
                description=description,
                chunk_count=1  # Placeholder
            )

            _ = DEBUG and log_debug(f"Document uploaded: {file.filename}"
                                    f" ({len(content)} bytes)")
            return std_response(result=document_info)

        except Exception as e:
            log_error(f"Failed to upload document: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to upload document: {str(e)}"
            )

    # File Generation Methods

    async def generate_file(
        self,
        request: FileGenerationRequest
    ) -> Dict[str, str]:
        """
        Generate a file from content.

        Args:
            request: File generation request.

        Returns:
            Dict[str, str]: Generated file information as
                result=GeneratedFile().
        """
        try:
            # Validate content based on file type
            if request.file_type == "json":
                from genericsuite_codegen.utilities import \
                    validate_json_content
                is_valid, error = validate_json_content(request.content)
                if not is_valid:
                    return std_error_response(
                        status_code=400,
                        detail=f"Invalid JSON content: {error}"
                    )

            generated_file = GeneratedFile(
                filename=request.filename,
                content=request.content,
                file_type=request.file_type,
                size=len(request.content.encode('utf-8')),
                description=request.description
            )
            return std_response(result=generated_file)

        except Exception as e:
            # if isinstance(e, HTTPException):
            #     raise
            log_error(f"Failed to generate file: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to generate file: {str(e)}"
            )

    async def create_file_package(
        self,
        files: List[GeneratedFile]
    ) -> Dict[str, str]:
        """
        Create a package from multiple files.

        Args:
            files: List of files to package.

        Returns:
            Dict[str, str]: File package information as result=FilePackage().
        """
        try:
            total_size = sum(file.size for file in files)
            package_name = "genericsuite_codegen_" + \
                           f"{get_utcnow_fmt()}"
            package = FilePackage(
                package_name=package_name,
                files=files,
                format="zip",
                total_size=total_size
            )
            return std_response(result=package)

        except Exception as e:
            log_error(f"Failed to create file package: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to create package: {str(e)}"
            )

    # Search Methods

    async def search_knowledge_base(
        self,
        query: SearchQuery,
        translate_path: bool = False,
    ) -> Dict[str, str]:
        """
        Search the knowledge base.

        Args:
            query: Search query.

        Returns:
            Dict[str, str]: Search results as
                result=KnowledgeBaseSearchResults().
        """
        try:
            # import time
            # start_time = time.time()

            # Use the knowledge base tool for search
            from genericsuite_codegen.agent.tools import KnowledgeBaseTool
            kb_tool = KnowledgeBaseTool()

            # Perform search
            results = kb_tool.search_similar_documents(
                query.query,
                limit=query.limit,
                file_type_filter=query.file_type_filter,
                similarity_threshold=query.similarity_threshold,
                translate_path=translate_path
            )

            return std_response(result=results)

            # execution_time = time.time() - start_time
            # return std_response(
            #     result=SearchResponse(
            #         results=results,
            #         total_results=results.total_results,
            #         query=query.query,
            #         execution_time=execution_time
            #     ))

        except Exception as e:
            log_error(f"Knowledge base search failed: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )

    async def clean_knowledge_base(self) -> Dict[str, str]:
        """
        Clean all vectors from the knowledge base.

        Returns:
            Dict[str, str]: Cleanup confirmation.
        """
        try:
            _ = DEBUG and log_debug("Cleaning knowledge base...")

            # Delete all vectors
            success = self.db.delete_all_vectors()

            if success:
                _ = DEBUG and log_debug("Knowledge base cleaned successfully")
                return std_response(
                    result={"message": "Knowledge base cleaned successfully"}
                )
            else:
                log_error("Failed to clean knowledge base")
                return std_error_response(
                    status_code=500,
                    detail="Failed to clean knowledge base"
                )

        except Exception as e:
            log_error(f"Knowledge base cleanup failed: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Cleanup failed: {str(e)}"
            )

    # Statistics Methods

    async def get_statistics(self) -> Statistics:
        """
        Get system statistics.

        Returns:
            Dict[str, str]: System statistics as result=Statistics().
        """
        try:
            # Get knowledge base stats
            knowledge_base = self.db.database.knowledge_base
            kb_count = knowledge_base.count_documents({})

            # Get conversations stats
            conversation = ConversationsService()
            conv_stats = await conversation.statistics()
            conv_count = conv_stats.total_conversations

            # Get agent info
            agent_info = self.agent.get_model_info()

            # stats = Statistics(
            #     knowledge_base={
            #         "total_chunks": kb_count,
            #         "last_updated": None  # TODO: Would be tracked in real
            #                               # implementation
            #     },
            #     conversations={
            #         "total_conversations": conv_count
            #     },
            #     agent=agent_info,
            #     system={
            #         "uptime": "unknown",  # TODO: Would be tracked in real
            #                               # implementation
            #         "memory_usage": "unknown"
            #     }
            # )
            stats = Statistics(
                knowledge_base=KnowledgeBaseStatistics(
                    total_chunks=kb_count,
                    last_updated=None  # TODO: Would be tracked in real
                    # implementation
                ),
                conversations=ConversationStatistics(
                    total_conversations=conv_count
                ),
                agent=AgentModel(agent_info),
                system=SystemStatistics(
                    uptime="unknown",  # TODO: Would be tracked in real
                    # implementation
                    memory_usage="unknown"
                )
            )

            return std_response(result=stats)

        except Exception as e:
            log_error(f"Failed to get statistics: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to get statistics: {str(e)}"
            )

    # Helper Methods

    def _get_working_data(self, repository_url: str = None) -> str:
        """Get working data."""
        repo_url = repository_url or get_envvar("REMOTE_REPO_URL")
        repo_branch = get_envvar("REMOTE_REPO_BRANCH")
        local_dir = get_envvar("LOCAL_REPO_DIR")
        if repo_url and local_dir:
            return std_response(result={
                "repository_url": repo_url,
                "repository_branch": repo_branch,
                "local_dir": local_dir
            })
        return std_error_response(
            status_code=400,
            detail="No repository URL or local directory provided"
        )

    async def _update_knowledge_base_background(
        self,
        operation_id: str,
        progress: IngestionProgress
    ) -> None:
        """Background task for knowledge base update."""
        try:
            _ = DEBUG and log_debug(
                f"Starting knowledge base update [{operation_id}]")

            from genericsuite_codegen.document_processing.ingestion import \
                run_ingestion

            # Process repository
            working_data = self._get_working_data(progress.repository_url)
            if working_data.error:
                return working_data
            repo_url = working_data.result["repository_url"]
            repo_branch = working_data.result["repository_branch"]
            local_dir = working_data.result["local_dir"]
            result = run_ingestion(
                repo_url=repo_url,
                repo_branch=repo_branch,
                local_dir=local_dir,
                force_refresh=progress.force_refresh or True,
                database_manager=self.db,
            )

            _ = DEBUG and log_debug(
                f"Knowledge base update completed [{operation_id}]")
            return std_response(result=result)

        except Exception as e:
            log_error(f"Knowledge base update failed [{operation_id}]: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to update knowledge base: {str(e)}"
            )

    async def health_check_endpoint(self) -> dict:
        """
        Health check endpoint.

        Returns:
            HealthResponse: Application health status.
        """
        try:
            # Check database connection
            db = get_database_connection()
            # db = initialize_database()
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
            log_error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                version=get_app_info().version,
                components={"error": str(e)},
            )

    async def status_endpoint(self) -> dict:
        """
        Detailed status endpoint.

        Returns:
            Dict[str, Any]: Detailed application status.
        """
        try:
            # Get database stats
            db = get_database_connection()
            # db = initialize_database()
            db_stats = {}

            try:
                # Get collection counts
                knowledge_base = db.knowledge_base
                conversations = db.ai_chatbot_conversations

                db_stats = {
                    "knowledge_base_documents":
                        knowledge_base.count_documents({}),
                    "conversations": conversations.count_documents({}),
                    "connection_status": "connected",
                }
            except Exception as e:
                db_stats = {"connection_status": "error", "error": str(e)}

            # Get agent info
            agent = get_agent()
            agent_info = agent.get_model_info()

            return std_response(result={
                "application": get_app_info().model_dump(),
                "database": db_stats,
                "agent": agent_info,
                "environment": {
                    "debug": get_envvar("SERVER_DEBUG", "0") == "1",
                    "cors_origins": get_envvar("CORS_ORIGIN", "").split(","),
                    "allowed_hosts":
                        get_envvar("ALLOWED_HOSTS", "").split(","),
                },
            })

        except Exception as e:
            log_error(f"Status check failed: {e}")
            return std_error_response(
                status_code=500, detail=f"Status check failed: {e}")

    async def get_settings(self) -> Dict[str, Any]:
        """
        Get application settings based on .env.example.

        Returns:
            Dict[str, Any]: Standard response with SettingsResponse.
        """
        settings = SettingsManager()
        return await settings.get_all()

    async def update_settings(
        self,
        request: UpdateSettingsRequest
    ) -> Dict[str, Any]:
        """
        Update application settings.

        Args:
            request: Update settings request.

        Returns:
            Dict[str, Any]: Standard response.
        """
        settings = SettingsManager()
        return await settings.update(request)

    async def get_filename_data(
        self,
        filename: str
    ) -> dict:
        """
        Download a generated file.

        Args:
            filename: Name of the file to download.
            content: File content (would typically be retrieved from storage).

        Returns:
            Response: File download response.
        """

        # Sanitize filename for security
        safe_filename = sanitize_filename(filename)
        # Get content type
        content_type = get_content_type(safe_filename)

        return std_response(
            resultset={
                'safe_filename': safe_filename,
                'content_type': content_type,
            }
        )

    async def download_package_endpoint(
        self,
        package_name: str
    ) -> dict:
        """
        Download a file package as ZIP.

        Args:
            package_name: Name of the package to download.

        Returns:
            StreamingResponse: ZIP file download.
        """
        import io
        import zipfile

        # TODO: This is a placeholder implementation
        # In a real system, you would retrieve the package from storage

        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer, "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:
            # Add placeholder files
            # (TODO: in real implementation, get from storage)
            zip_file.writestr(
                "README.md",
                "# Generated Code Package\n\n"
                "This package contains generated code files.",
            )
            zip_file.writestr(
                "example.json", '{"message": "This is a generated JSON file"}'
            )

        zip_buffer.seek(0)

        safe_package_name = sanitize_filename(package_name)

        return std_response(
            result={
                'safe_package_name': safe_package_name,
                'zip_buffer': zip_buffer,
            }
        )

    async def generate_json_config_endpoint(
        self,
        requirements: str,
        table_name: str,
        user_id: str,
        config_type: str = "table",
    ) -> dict:
        """
        Generate JSON configuration for GenericSuite.

        Args:
            requirements: Requirements for the configuration.
            table_name: Name of the table.
            config_type: Type of configuration (table, form, menu).

        Returns:
            GeneratedFile: Generated JSON configuration file.
        """
        from genericsuite_codegen.utilities import extract_code_blocks

        conversation = ConversationsService()
        query = f"{config_type.capitalize()} {table_name}," \
            + f" with the requirements: {requirements}"
        await conversation.init(query=query, user_id=user_id)

        try:
            # Use agent to generate JSON config
            agent = get_agent()
            response = await agent.generate_json_config(
                requirements=requirements,
                config_type=config_type,
                table_name=table_name,
            )

            # Extract JSON from response content
            # code_blocks = extract_code_blocks(response.content, "json")
            code_blocks = extract_code_blocks(response.content)

            if not code_blocks:
                return std_error_response(
                    status_code=400,
                    detail="No JSON configuration found in agent response",
                )

            files = []
            for code_block in code_blocks:
                json_content = code_block["code"]
                filename = f"{config_type}_config_" + \
                    f"{get_utcnow_fmt()}.json"
                files.append(
                    GeneratedFile(
                        filename=filename,
                        content=json_content,
                        file_type="json",
                        size=len(json_content.encode("utf-8")),
                        description=f"Generated {config_type}"
                        " configuration for"
                        f" table '{table_name}'",
                    )
                )

            await conversation.save_message(
                query=query,
                content=response.content,
                sources=response.sources,
                task_type=response.task_type,
                model_used=response.model_used,
                token_usage=response.token_usage,
            )

            result = std_response(
                result=GeneratedFilesResponse(files=files)
            )
            return result

        except Exception as e:
            log_error(f"JSON config generation failed: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to generate JSON config: {str(e)}"
            )

    async def generate_python_code_endpoint(
        self,
        requirements: str,
        tool_name: str,
        description: str,
        user_id: str,
        code_type: str = "tool",
    ) -> dict:
        """
        Generate Python code for GenericSuite.

        Args:
            requirements: Requirements for the code.
            tool_name: Name of the tool/langchain/mcp.
            description: Description of the tool/langchain/mcp.
            code_type: Type of code (tool, langchain, mcp).

        Returns:
            GeneratedFile: Generated Python code file.
        """
        conversation = ConversationsService()
        query = f"Name: '{tool_name}', Type: '{code_type}', " \
            + f"Description: '{description}', Requirements: '{requirements}'"
        await conversation.init(query=query, user_id=user_id)

        try:
            # Use agent to generate Python code
            agent = get_agent()
            response = await agent.generate_python_code(
                requirements=requirements,
                tool_name=tool_name,
                description=description,
                code_type=code_type,
            )

            # Extract Python code from response content
            code_blocks = extract_code_blocks(response.content, "python")
            if not code_blocks:
                return std_error_response(
                    status_code=400,
                    detail="No Python code found in agent response"
                )

            files = []
            for code_block in code_blocks:
                python_content = code_block["code"]
                filename = \
                    f"{code_type}_{get_utcnow_fmt()}.py"
                files.append(
                    GeneratedFile(
                        filename=filename,
                        content=python_content,
                        file_type="python",
                        size=len(python_content.encode("utf-8")),
                        description=f"Generated {code_type} Python code",
                    )
                )

            await conversation.save_message(
                query=query,
                content=response.content,
                sources=response.sources,
                task_type=response.task_type,
                model_used=response.model_used,
                token_usage=response.token_usage,
            )

            return std_response(
                result=GeneratedFilesResponse(files=files)
            )

        except Exception as e:
            log_error(f"Python code generation failed: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to generate Python code: {str(e)}"
            )

    async def generate_frontend_code_endpoint(
        self,
        requirements: str,
        user_id: str,
    ) -> dict:
        """
        Generate ReactJS frontend code.

        Args:
            requirements: Requirements for the frontend code.

        Returns:
            List[GeneratedFile]: Generated frontend code files.
        """

        conversation = ConversationsService()
        query = "Name: 'Code Generation', Type: 'Frontend', " \
            + f"Requirements: '{requirements}'"
        await conversation.init(query=query, user_id=user_id)

        try:
            # Use agent to generate frontend code
            agent = get_agent()
            response = await agent.generate_frontend_code(requirements)

            # Extract code blocks from response
            generated_files = []

            # Extract different types of code blocks
            for lang in ["jsx", "tsx", "javascript", "typescript", "css"]:
                code_blocks = extract_code_blocks(response.content, lang)

                for i, block in enumerate(code_blocks):
                    ext = "jsx" if lang in ["jsx", "tsx"] else lang[:2]
                    filename = f"component_{i+1}_" + \
                        f"{get_utcnow_fmt()}.{ext}"

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
                    "frontend_code_" +
                    f"{get_utcnow_fmt()}.jsx"
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

            await conversation.save_message(
                query=query,
                content=response.content,
                sources=response.sources,
                task_type=response.task_type,
                model_used=response.model_used,
                token_usage=response.token_usage,
            )

            return std_response(result=GeneratedFilesResponse(
                files=generated_files))

        except Exception as e:
            log_error(f"Frontend code generation failed: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to generate frontend code: {str(e)}"
            )

    async def generate_backend_code_endpoint(
        self,
        requirements: str,
        user_id: str,
        framework: str = None,
    ) -> dict:
        """
        Generate backend code for specified framework.

        Args:
            requirements: Requirements for the backend code.
            framework: Backend framework (fastapi, flask, chalice).

        Returns:
            List[GeneratedFile]: Generated backend code files.
        """

        # Validate framework
        if framework is None:
            framework = "fastapi"
        valid_frameworks = ["fastapi", "flask", "chalice"]
        if framework.lower() not in valid_frameworks:
            return std_error_response(
                status_code=400,
                detail="Invalid framework. Must be one of:"
                f" {valid_frameworks}"
            )

        conversation = ConversationsService()
        query = "Name: 'Code Generation', Type: 'Backend', " \
            + f"Requirements: '{requirements}'"
        await conversation.init(query=query, user_id=user_id)

        try:
            # Use agent to generate backend code
            agent = get_agent()
            response = await agent.generate_backend_code(
                requirements, framework, user_id)

            # Extract Python code blocks from response
            from genericsuite_codegen.utilities import extract_code_blocks

            code_blocks = extract_code_blocks(response.content, "python")

            generated_files = []

            for i, block in enumerate(code_blocks):
                filename = f"{framework}_backend_{i+1}_" + \
                    f"{get_utcnow_fmt()}.py"

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
                filename = f"{framework}_backend_{get_utcnow_fmt()}.py"
                generated_files.append(
                    GeneratedFile(
                        filename=filename,
                        content=response.content,
                        file_type="python",
                        size=len(response.content.encode("utf-8")),
                        description=f"Generated {framework} backend code",
                    )
                )

            await conversation.save_message(
                query=query,
                content=response.content,
                sources=response.sources,
                task_type=response.task_type,
                model_used=response.model_used,
                token_usage=response.token_usage,
            )

            return std_response(result=GeneratedFilesResponse(
                files=generated_files))

        except Exception as e:
            log_error(f"Backend code generation failed: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to generate backend code: {str(e)}"
            )


# --------------------


# Global endpoint methods instance
_endpoint_methods: Optional[EndpointMethods] = None


def get_endpoint_methods() -> EndpointMethods:
    """
    Get or create the global endpoint methods instance.

    Returns:
        EndpointMethods: Global endpoint methods instance.
    """
    global _endpoint_methods

    if _endpoint_methods is None:
        _endpoint_methods = EndpointMethods()

    return _endpoint_methods
