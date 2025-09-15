"""
Core endpoint implementations for the FastAPI application.

This module contains the implementation logic for all API endpoints,
separated from the route definitions for better organization and testing.
"""

import os
import logging
# import re
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

# from fastapi import HTTPException, UploadFile, BackgroundTasks
# from fastapi.responses import StreamingResponse

from .types import (
    QueryRequest,
    QueryResponse,
    ConversationCreate,
    ConversationUpdate,
    Conversation,
    ConversationList,
    KnowledgeBaseUpdate,
    KnowledgeBaseStatus,
    DocumentInfo,
    FileGenerationRequest,
    GeneratedFile,
    GeneratedFilesResponse,
    FilePackage,
    SearchQuery,
    SearchResponse,
    Statistics,
    HealthResponse,
)
from genericsuite_codegen.agent.types import (
    AgentModel,
    AgentContext,
)
from genericsuite_codegen.api.types import (
    KnowledgeBaseStatistics,
    ConversationStatistics,
    SystemStatistics,
)
from genericsuite_codegen.document_processing.types import (
    IngestionStatistics,
    IngestionResult,
)

from .utilities import (
    std_error_response,
    std_response,
    sanitize_filename,
    get_content_type,
    extract_code_blocks,
    get_app_info,
    get_utcnow_fmt,
)
from genericsuite_codegen.agent.agent import (
    get_agent,
    QueryRequest as AgentQueryRequest
)
from genericsuite_codegen.database.setup import (
    get_database_connection,
    # initialize_database,
    test_database_connection,
)

DEBUG = False

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if DEBUG else logging.WARNING)


class EndpointMethods:
    """
    Core endpoint implementation methods.

    This class contains all the business logic for API endpoints,
    separated from FastAPI route definitions for better testability.
    """

    def __init__(self):
        """Initialize endpoint methods."""
        self.db = get_database_connection()
        # self.db = initialize_database()
        self.agent = get_agent()

    # Agent Query Methods

    async def query_agent(
        self,
        request: QueryRequest,
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Process an agent query request.

        Args:
            request: Query request data.
            correlation_id: Request correlation ID.

        Returns:
            QueryResponse: Agent response.

        Raises:
            HTTPException: If query processing fails.
        """
        try:
            logger.info(
                "Processing agent query "
                f"[{correlation_id}]: {request.query[:100]}...")

            # Validate request
            if not request.query or not request.query.strip():
                return std_error_response(
                    status_code=400,
                    detail="Query cannot be empty"
                )

            # If no conversation_id provided, create a new conversation
            conversation_id = request.conversation_id
            if not conversation_id:
                # Create new conversation with the query as initial message
                from .types import ConversationCreate
                create_request = ConversationCreate(
                    initial_message=request.query.strip()
                )

                # TODO: Use default user for now (in real app, this would come
                # from auth)
                user_id = "default_user"

                create_result = await self.create_conversation(
                    create_request, user_id)
                if create_result.error:
                    logger.error(
                        "Failed to create conversation:"
                        f" {create_result.error_message}")
                    return create_result

                conversation_id = create_result.result.id
                logger.info(
                    f"Created new conversation {conversation_id} for query")

            # Convert API request to agent request
            agent_request = AgentQueryRequest(
                query=request.query,
                task_type=request.task_type.value,
                framework=(
                    request.framework.value if request.framework
                    else None),
                context_limit=request.context_limit,
                include_sources=request.include_sources
            )

            # Get conversation context if conversation exists
            agent_context = None
            if conversation_id:
                agent_context = await self._get_conversation_context(
                    conversation_id)

            # Process query with agent
            agent_response = await self.agent.query(agent_request,
                                                    context=agent_context)

            # Convert agent response to API response
            response = QueryResponse(
                content=agent_response.content,
                sources=agent_response.sources,
                task_type=request.task_type,
                model_used=agent_response.model_used,
                token_usage=agent_response.token_usage,
                conversation_id=conversation_id
            )

            # Save messages to conversation (only if conversation already
            # existed)
            if request.conversation_id:
                await self._add_message_to_conversation(
                    conversation_id,
                    request.query,
                    agent_response.content,
                    agent_response.sources,
                    agent_response.token_usage
                )
            else:
                # For new conversations, just add the assistant response
                # (user message was already added during conversation creation)
                await self._add_assistant_message_to_conversation(
                    conversation_id,
                    agent_response.content,
                    agent_response.sources,
                    agent_response.token_usage
                )

            logger.info(f"Query processed successfully [{correlation_id}]")
            return std_response(result=response)

        except Exception as e:
            logger.error(f"Query processing failed [{correlation_id}]: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Query processing failed: {str(e)}"
            )

    async def stream_agent_query(
        self,
        request: QueryRequest,
        correlation_id: str
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
            logger.info(f"Starting streaming query [{correlation_id}]:"
                        f" {request.query[:100]}...")

            # For now, we'll implement basic streaming by yielding the full
            # response.
            # TODO: In a full implementation, this would integrate with
            # the agent's streaming capabilities.
            result = await self.query_agent(request, correlation_id)
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
            logger.error(f"Streaming query failed [{correlation_id}]: {e}")
            yield f"data: ERROR: {str(e)}\n\n"

    # Conversation Management Methods

    async def create_conversation(
        self,
        request: ConversationCreate,
        user_id: str
    ) -> Conversation:
        """
        Create a new conversation.

        Args:
            request: Conversation creation request.
            user_id: User ID.

        Returns:
            Conversation: Created conversation.
        """
        try:
            # Validate user_id
            if not user_id or not user_id.strip():
                return std_error_response(
                    status_code=400,
                    detail="User ID is required"
                )

            conversations = self.db.database.ai_chatbot_conversations

            # Generate unique title based on initial message or timestamp
            if request.initial_message and request.initial_message.strip():
                # Use first 50 characters of initial message for title
                title = request.initial_message.strip()[:50]
                if len(request.initial_message.strip()) > 50:
                    title += "..."
            elif request.title and request.title.strip():
                title = request.title.strip()
            else:
                title = "New Conversation " + \
                        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"

            # Ensure title uniqueness for this user
            title = await self._ensure_unique_title(title, user_id)

            conversation_data = {
                "user_id": user_id,
                "title": title,
                "messages": [],
                "creation_date": datetime.utcnow(),
                "update_date": datetime.utcnow()
            }

            # Add initial message if provided
            if request.initial_message and request.initial_message.strip():
                message_id = str(uuid.uuid4())
                conversation_data["messages"].append({
                    "id": message_id,
                    "role": "user",
                    "content": request.initial_message.strip(),
                    "timestamp": datetime.utcnow(),
                    "sources": None,
                    "token_usage": None
                })

            result = conversations.insert_one(conversation_data)
            conversation_data["_id"] = result.inserted_id

            # Return the complete conversation object
            created_conversation = self._convert_conversation_document(
                conversation_data)

            logger.info(
                f"Created conversation {result.inserted_id} for user"
                f" {user_id}")

            return std_response(result=created_conversation)

        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to create conversation: {str(e)}"
            )

    async def get_conversations(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> ConversationList:
        """
        Get user conversations with pagination.

        Args:
            user_id: User ID.
            page: Page number (1-based).
            page_size: Items per page.

        Returns:
            ConversationList: Paginated conversation list.
        """
        try:
            # Validate inputs
            if not user_id or not user_id.strip():
                return std_error_response(
                    status_code=400,
                    detail="User ID is required"
                )

            if page < 1:
                return std_error_response(
                    status_code=400,
                    detail="Page number must be greater than 0"
                )

            if page_size < 1 or page_size > 100:
                return std_error_response(
                    status_code=400,
                    detail="Page size must be between 1 and 100"
                )

            conversations = self.db.database.ai_chatbot_conversations

            # Calculate skip value
            skip = (page - 1) * page_size

            # Get total count
            total = conversations.count_documents({"user_id": user_id})

            # Get conversations
            cursor = conversations.find(
                {"user_id": user_id}
            ).sort("update_date", -1).skip(skip).limit(page_size)

            conversation_docs = cursor.to_list(length=page_size)
            conversation_list = [
                self._convert_conversation_document(doc)
                for doc in conversation_docs
            ]

            return std_response(
                result=ConversationList(
                    conversations=conversation_list,
                    total=total,
                    page=page,
                    page_size=page_size
                ))

        except Exception as e:
            logger.error(f"Failed to get conversations: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to get conversations: {str(e)}"
            )

    async def get_conversation(
        self,
        conversation_id: str,
        user_id: str
    ) -> Conversation:
        """
        Get a specific conversation.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID.

        Returns:
            Conversation: Conversation data.
        """
        try:
            from bson import ObjectId

            # Validate inputs
            if not conversation_id or not conversation_id.strip():
                return std_error_response(
                    status_code=400,
                    detail="Conversation ID is required"
                )

            if not user_id or not user_id.strip():
                return std_error_response(
                    status_code=400,
                    detail="User ID is required"
                )

            conversations = self.db.database.ai_chatbot_conversations

            conversation_doc = conversations.find_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id
            })

            if not conversation_doc:
                return std_error_response(
                    status_code=404,
                    detail="Conversation not found or access denied"
                )

            return std_response(
                result=self._convert_conversation_document(conversation_doc))

        except Exception as e:
            logger.error(f"Failed to get conversation: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to get conversation: {str(e)}"
            )

    async def update_conversation(
        self,
        conversation_id: str,
        request: ConversationUpdate,
        user_id: str
    ) -> Conversation:
        """
        Update a conversation.

        Args:
            conversation_id: Conversation ID.
            request: Update request.
            user_id: User ID.

        Returns:
            Conversation: Updated conversation.
        """
        try:
            from bson import ObjectId

            # Validate inputs
            if not conversation_id or not conversation_id.strip():
                return std_error_response(
                    status_code=400,
                    detail="Conversation ID is required"
                )

            if not user_id or not user_id.strip():
                return std_error_response(
                    status_code=400,
                    detail="User ID is required"
                )

            conversations = self.db.database.ai_chatbot_conversations

            update_data = {"update_date": datetime.utcnow()}

            if request.title is not None:
                # Validate title
                title = request.title.strip()
                if not title:
                    return std_error_response(
                        status_code=400,
                        detail="Title cannot be empty"
                    )

                # Ensure title uniqueness (excluding current conversation)
                existing = conversations.find_one({
                    "user_id": user_id,
                    "title": title,
                    "_id": {"$ne": ObjectId(conversation_id)}
                })

                if existing:
                    # Generate unique title
                    title = await self._ensure_unique_title(title, user_id)

                update_data["title"] = title

            # Verify conversation exists and belongs to user
            existing_conversation = conversations.find_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id
            })

            if not existing_conversation:
                return std_error_response(
                    status_code=404,
                    detail="Conversation not found or access denied"
                )

            result = conversations.update_one(
                {"_id": ObjectId(conversation_id), "user_id": user_id},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                return std_error_response(
                    status_code=500,
                    detail="Failed to update conversation"
                )

            # Get updated conversation
            updated_result = await self.get_conversation(conversation_id,
                                                         user_id)
            if updated_result.error:
                return updated_result

            logger.info(
                f"Updated conversation {conversation_id} for user {user_id}")
            return std_response(result=updated_result.result)

        except Exception as e:
            logger.error(f"Failed to update conversation: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to update conversation: {str(e)}"
            )

    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: str
    ) -> Dict[str, str]:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID.

        Returns:
            Dict[str, str]: Deletion confirmation.
        """
        try:
            from bson import ObjectId

            # Validate inputs
            if not conversation_id or not conversation_id.strip():
                return std_error_response(
                    status_code=400,
                    detail="Conversation ID is required"
                )

            if not user_id or not user_id.strip():
                return std_error_response(
                    status_code=400,
                    detail="User ID is required"
                )

            conversations = self.db.database.ai_chatbot_conversations

            # Verify conversation exists and belongs to user before deletion
            existing_conversation = conversations.find_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id
            })

            if not existing_conversation:
                return std_error_response(
                    status_code=404,
                    detail="Conversation not found or access denied"
                )

            result = conversations.delete_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id
            })

            if result.deleted_count == 0:
                return std_error_response(
                    status_code=500,
                    detail="Failed to delete conversation"
                )

            logger.info(
                f"Deleted conversation {conversation_id} for user {user_id}")
            return std_response(result={
                "message": "Conversation deleted successfully"})

        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to delete conversation: {str(e)}"
            )

    # Knowledge Base Management Methods

    async def update_knowledge_base(
        self,
        request: KnowledgeBaseUpdate,
        background_tasks: Optional[any] = None
    ) -> Dict[str, str]:
        """
        Trigger knowledge base update.

        Args:
            request: Update request.
            background_tasks: FastAPI background tasks.

        Returns:
            Dict[str, str]: Update initiation response.
        """
        try:
            # Start background update task
            logger.error(">> Starting knowledge base update")
            operation_id = f"kb_update_{get_utcnow_fmt()}"

            if background_tasks:
                background_tasks.add_task(
                    self._update_knowledge_base_background,
                    operation_id,
                    request
                )
                return std_response(
                    result=IngestionResult(
                        success=True,
                        status="Knowledge base update started",
                        statistics=IngestionStatistics(
                            total_documents=0,
                            total_chunks=0,
                            total_embeddings=0,
                            duration_seconds=0,
                        )
                    )
                )
            return std_response(
                result=await self._update_knowledge_base_background(
                    operation_id,
                    request
                ))

        except Exception as e:
            logger.error(f"Failed to start knowledge base update: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to start update: {str(e)}"
            )

    async def get_operation_progress(
        self,
    ) -> Dict[str, Any]:
        """Get operation progress."""
        from genericsuite_codegen.document_processing.ingestion import \
            get_ingestion_progress
        # working_data = self._get_working_data(None)
        # if working_data.error:
        #     return working_data
        # repo_url = working_data.result["repository_url"]
        # local_dir = working_data.result["local_dir"]
        # result = get_ingestion_progress(
        #     repo_url=repo_url,
        #     local_dir=local_dir,
        #     database_manager=self.db,
        # )
        result = get_ingestion_progress()
        return std_response(result=result)

    async def get_knowledge_base_status(self) -> KnowledgeBaseStatus:
        """
        Get knowledge base status.

        Returns:
            KnowledgeBaseStatus: Current status.
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
            import os
            repository_url = os.getenv("REMOTE_REPO_URL", "")

            return std_response(
                result=KnowledgeBaseStatus(
                    status="idle",  # TODO: This would be tracked in a real
                                    # implementation
                    document_count=unique_files,
                    chunk_count=document_count,
                    repository_url=repository_url,
                    last_update=None  # TODO: This would be tracked in database
                )
            )

        except Exception as e:
            logger.error(f"Failed to get knowledge base status: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to get status: {str(e)}"
            )

    async def upload_document(
        self,
        file: any,
        description: Optional[str] = None
    ) -> DocumentInfo:
        """
        Upload and process a document.

        Args:
            file: Uploaded file.
            description: Optional description.

        Returns:
            DocumentInfo: Document information.
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
                upload_date=datetime.utcnow(),
                description=description,
                chunk_count=1  # Placeholder
            )

            logger.info(f"Document uploaded: {file.filename}"
                        f" ({len(content)} bytes)")
            return std_response(result=document_info)

        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to upload document: {str(e)}"
            )

    # File Generation Methods

    async def generate_file(
        self,
        request: FileGenerationRequest
    ) -> GeneratedFile:
        """
        Generate a file from content.

        Args:
            request: File generation request.

        Returns:
            GeneratedFile: Generated file information.
        """
        try:
            # Validate content based on file type
            if request.file_type == "json":
                from .utilities import validate_json_content
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
            logger.error(f"Failed to generate file: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to generate file: {str(e)}"
            )

    async def create_file_package(
        self,
        files: List[GeneratedFile]
    ) -> FilePackage:
        """
        Create a package from multiple files.

        Args:
            files: List of files to package.

        Returns:
            FilePackage: File package information.
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
            logger.error(f"Failed to create file package: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to create package: {str(e)}"
            )

    # Search Methods

    async def search_knowledge_base(
        self,
        query: SearchQuery
    ) -> SearchResponse:
        """
        Search the knowledge base.

        Args:
            query: Search query.

        Returns:
            SearchResponse: Search results.
        """
        try:
            import time
            start_time = time.time()

            # Use the knowledge base tool for search
            from genericsuite_codegen.agent.tools import KnowledgeBaseTool
            kb_tool = KnowledgeBaseTool()

            # Perform search
            results = await kb_tool.search_similar_documents(
                query.query,
                limit=query.limit,
                file_type_filter=query.file_type_filter,
                similarity_threshold=query.similarity_threshold
            )

            execution_time = time.time() - start_time

            return std_response(
                result=SearchResponse(
                    results=results,
                    total_results=len(results),
                    query=query.query,
                    execution_time=execution_time
                ))

        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
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
            logger.info("Cleaning knowledge base...")

            # Delete all vectors
            success = self.db.delete_all_vectors()

            if success:
                logger.info("Knowledge base cleaned successfully")
                return std_response(
                    result={"message": "Knowledge base cleaned successfully"}
                )
            else:
                logger.error("Failed to clean knowledge base")
                return std_error_response(
                    status_code=500,
                    detail="Failed to clean knowledge base"
                )

        except Exception as e:
            logger.error(f"Knowledge base cleanup failed: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Cleanup failed: {str(e)}"
            )

    # Statistics Methods

    async def get_statistics(self) -> Statistics:
        """
        Get system statistics.

        Returns:
            Statistics: System statistics.
        """
        try:
            # Get knowledge base stats
            knowledge_base = self.db.database.knowledge_base
            conversations = self.db.database.ai_chatbot_conversations

            kb_count = knowledge_base.count_documents({})
            conv_count = conversations.count_documents({})

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
            logger.error(f"Failed to get statistics: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to get statistics: {str(e)}"
            )

    # Helper Methods

    def _convert_conversation_document(
        self,
        doc: Dict[str, Any]
    ) -> Conversation:
        """Convert MongoDB document to Conversation model."""
        from .types import Message

        messages = []
        for msg_data in doc.get("messages", []):
            # Ensure message has an ID (for backward compatibility)
            message_id = msg_data.get("id", str(uuid.uuid4()))

            messages.append(Message(
                id=message_id,
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=msg_data["timestamp"],
                sources=msg_data.get("sources"),
                token_usage=msg_data.get("token_usage")
            ))

        return Conversation(
            id=str(doc["_id"]),
            title=doc["title"],
            messages=messages,
            created_at=doc["creation_date"],
            updated_at=doc["update_date"],
            message_count=len(messages)
        )

    async def _ensure_unique_title(
            self,
            base_title: str,
            user_id: str) -> str:
        """Ensure conversation title is unique for the user."""
        try:
            conversations = self.db.database.ai_chatbot_conversations

            # Check if base title already exists
            existing = conversations.find_one({
                "user_id": user_id,
                "title": base_title
            })

            if not existing:
                return base_title

            # Generate unique title by appending number
            counter = 1
            while True:
                new_title = f"{base_title} ({counter})"
                existing = conversations.find_one({
                    "user_id": user_id,
                    "title": new_title
                })

                if not existing:
                    return new_title

                counter += 1

                # Safety check to prevent infinite loop
                if counter > 1000:
                    import time
                    return f"{base_title} ({int(time.time())})"

        except Exception as e:
            logger.error(f"Error ensuring unique title: {e}")
            # Fallback to timestamp-based title
            import time
            return f"{base_title} ({int(time.time())})"

    async def _add_assistant_message_to_conversation(
        self,
        conversation_id: str,
        assistant_message: str,
        sources: Optional[List[str]],
        token_usage: Optional[Dict[str, int]]
    ) -> None:
        """Add only an assistant message to a conversation."""
        try:
            from bson import ObjectId

            # Validate conversation_id
            if not conversation_id or not conversation_id.strip():
                logger.error("Invalid conversation_id provided")
                return

            conversations = self.db.database.ai_chatbot_conversations

            # Verify conversation exists
            conversation_exists = conversations.find_one(
                {"_id": ObjectId(conversation_id)})
            if not conversation_exists:
                logger.error(f"Conversation {conversation_id} not found")
                return

            # Generate unique ID for assistant message
            assistant_message_id = str(uuid.uuid4())

            message_to_add = {
                "id": assistant_message_id,
                "role": "assistant",
                "content": assistant_message,
                "timestamp": datetime.utcnow(),
                "sources": sources or [],
                "token_usage": token_usage
            }

            # Update conversation with new message
            result = conversations.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$push": {"messages": message_to_add},
                    "$set": {"update_date": datetime.utcnow()}
                }
            )

            if result.modified_count == 0:
                logger.error(
                    "Failed to add assistant message to conversation"
                    f" {conversation_id}")
            else:
                logger.info(
                    "Added assistant message to conversation"
                    f" {conversation_id}")

        except Exception as e:
            logger.error(
                f"Failed to add assistant message to conversation: {e}")

    async def _get_conversation_context(
        self, conversation_id: str
    ) -> Optional['AgentContext']:
        """
        Get conversation context for the agent including message history.

        Args:
            conversation_id: ID of the conversation to get context for

        Returns:
            AgentContext: Context object with conversation history, or None
            if not found
        """
        try:
            from bson import ObjectId
            from genericsuite_codegen.agent.types import AgentContext

            # Validate conversation_id
            if not conversation_id or not conversation_id.strip():
                logger.warning("Invalid conversation_id provided for context")
                return None

            conversations = self.db.database.ai_chatbot_conversations

            # Get conversation with messages
            conversation_doc = conversations.find_one(
                {"_id": ObjectId(conversation_id)}
            )

            if not conversation_doc:
                logger.warning(
                    f"Conversation {conversation_id} not found for context")
                return None

            # Convert messages to agent context format
            conversation_history = []
            messages = conversation_doc.get("messages", [])

            # Include recent messages (last 10 to maintain context whil
            # avoiding token limits)
            recent_messages = messages[-10:] if len(
                messages) > 10 else messages

            for msg in recent_messages:
                if msg.get("role") in ["user", "assistant"]:
                    conversation_history.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg.get("timestamp"),
                        "sources": msg.get("sources", [])
                        if msg["role"] == "assistant" else None
                    })

            # Create agent context
            agent_context = AgentContext(
                user_id=conversation_doc.get("user_id"),
                session_id=conversation_id,
                conversation_history=conversation_history,
                preferences={}
            )

            logger.info(
                f"Retrieved context for conversation {conversation_id} "
                f"with {len(conversation_history)} messages")
            return agent_context

        except Exception as e:
            logger.error(
                "Failed to get conversation context for"
                f" {conversation_id}: {e}")
            return None

    async def _add_message_to_conversation(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        sources: Optional[List[str]],
        token_usage: Optional[Dict[str, int]]
    ) -> None:
        """
        Add messages to a conversation with proper validation and ID
        generation.
        """
        try:
            from bson import ObjectId
            import uuid

            # Validate conversation_id
            if not conversation_id or not conversation_id.strip():
                logger.error("Invalid conversation_id provided")
                return

            conversations = self.db.database.ai_chatbot_conversations

            # Verify conversation exists
            conversation_exists = conversations.find_one(
                {"_id": ObjectId(conversation_id)})
            if not conversation_exists:
                logger.error(f"Conversation {conversation_id} not found")
                return

            # Generate unique IDs for messages
            user_message_id = str(uuid.uuid4())
            assistant_message_id = str(uuid.uuid4())

            messages_to_add = [
                {
                    "id": user_message_id,
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.utcnow(),
                    "sources": None,
                    "token_usage": None
                },
                {
                    "id": assistant_message_id,
                    "role": "assistant",
                    "content": assistant_message,
                    "timestamp": datetime.utcnow(),
                    "sources": sources or [],
                    "token_usage": token_usage
                }
            ]

            # Update conversation with new messages
            result = conversations.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$push": {"messages": {"$each": messages_to_add}},
                    "$set": {"update_date": datetime.utcnow()}
                }
            )

            if result.modified_count == 0:
                logger.error(
                    "Failed to add messages to conversation"
                    f" {conversation_id}")
            else:
                logger.info(
                    f"Added 2 messages to conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to add messages to conversation: {e}")
            # Don't raise exception here as it's not critical to the
            # main operation

    def _get_working_data(self, repository_url: str = None) -> str:
        """Get working data."""
        repo_url = repository_url or os.getenv("REMOTE_REPO_URL")
        local_dir = os.getenv("LOCAL_REPO_DIR")
        if repo_url and local_dir:
            return std_response(result={
                "repository_url": repo_url,
                "local_dir": local_dir
            })
        return std_error_response(
            status_code=400,
            detail="No repository URL or local directory provided"
        )

    async def _update_knowledge_base_background(
        self,
        operation_id: str,
        request: KnowledgeBaseUpdate
    ) -> None:
        """Background task for knowledge base update."""
        try:
            logger.info(f"Starting knowledge base update [{operation_id}]")

            from genericsuite_codegen.document_processing.ingestion import \
                run_ingestion

            # Process repository
            working_data = self._get_working_data(request.repository_url)
            if working_data.error:
                return working_data
            repo_url = working_data.result["repository_url"]
            local_dir = working_data.result["local_dir"]
            result = run_ingestion(
                repo_url=repo_url,
                local_dir=local_dir,
                force_refresh=request.force_refresh or True,
                database_manager=self.db,
            )

            logger.info(f"Knowledge base update completed [{operation_id}]")
            return std_response(result=result)

        except Exception as e:
            logger.error(f"Knowledge base update failed [{operation_id}]: {e}")
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
            logger.error(f"Health check failed: {e}")
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
                    "debug": os.getenv("SERVER_DEBUG", "0") == "1",
                    "cors_origins": os.getenv("CORS_ORIGINS", "").split(","),
                    "allowed_hosts": os.getenv("ALLOWED_HOSTS", "").split(","),
                },
            })

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return std_error_response(
                status_code=500, detail=f"Status check failed: {e}")

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
        config_type: str = "table",
    ) -> dict:
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
            response = await agent.generate_json_config(
                requirements=requirements,
                config_type=config_type,
                table_name=table_name,
            )

            # Extract JSON from response content
            from .utilities import extract_code_blocks

            code_blocks = extract_code_blocks(response.content, "json")

            if not code_blocks:
                return std_error_response(
                    status_code=400,
                    detail="No JSON configuration found in agent response",
                )

            json_content = code_blocks[0]["code"]
            filename = f"{config_type}_config_" + \
                f"{get_utcnow_fmt()}.json"

            result = std_response(
                result=GeneratedFilesResponse(
                    files=[
                        GeneratedFile(
                            filename=filename,
                            content=json_content,
                            file_type="json",
                            size=len(json_content.encode("utf-8")),
                            description=f"Generated {config_type}"
                            " configuration for"
                            f" table '{table_name}'",
                        )
                    ]
                )
            )
            return result

        except Exception as e:
            logger.error(f"JSON config generation failed: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to generate JSON config: {str(e)}"
            )

    async def generate_python_code_endpoint(
        self,
        requirements: str,
        tool_name: str,
        description: str,
        code_type: str = "tool",
    ) -> dict:
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

            python_content = code_blocks[0]["code"]
            filename = \
                f"{code_type}_{get_utcnow_fmt()}.py"

            return std_response(
                result=GeneratedFilesResponse(
                    files=[
                        GeneratedFile(
                            filename=filename,
                            content=python_content,
                            file_type="python",
                            size=len(python_content.encode("utf-8")),
                            description=f"Generated {code_type} Python code",
                        )
                    ]
                )
            )

        except Exception as e:
            logger.error(f"Python code generation failed: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to generate Python code: {str(e)}"
            )

    async def generate_frontend_code_endpoint(
        self,
        requirements: str
    ) -> dict:
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

            return std_response(result=GeneratedFilesResponse(
                files=generated_files))

        except Exception as e:
            logger.error(f"Frontend code generation failed: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to generate frontend code: {str(e)}"
            )

    async def generate_backend_code_endpoint(
        self,
        requirements: str,
        framework: str = "fastapi"
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
        valid_frameworks = ["fastapi", "flask", "chalice"]
        if framework.lower() not in valid_frameworks:
            return std_error_response(
                status_code=400,
                detail="Invalid framework. Must be one of:"
                f" {valid_frameworks}"
            )

        try:
            # Use agent to generate backend code
            agent = get_agent()
            response = await agent.generate_backend_code(
                requirements, framework)

            # Extract Python code blocks from response
            from .utilities import extract_code_blocks

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

            return std_response(result=GeneratedFilesResponse(
                files=generated_files))

        except Exception as e:
            logger.error(f"Backend code generation failed: {e}")
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
