"""
Core endpoint implementations for the FastAPI application.

This module contains the implementation logic for all API endpoints,
separated from the route definitions for better organization and testing.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse

from .types import (
    QueryRequest,
    QueryResponse,
    ConversationCreate,
    ConversationUpdate,
    Conversation,
    ConversationList,
    KnowledgeBaseUpdate,
    KnowledgeBaseStatus,
    ProgressUpdate,
    DocumentUpload,
    DocumentInfo,
    FileGenerationRequest,
    GeneratedFile,
    FilePackage,
    SearchQuery,
    SearchResponse,
    Statistics,
    create_error_response
)
from genericsuite_codegen.agent.agent import get_agent, AgentConfig, QueryRequest as AgentQueryRequest
from genericsuite_codegen.database.setup import get_database_connection

# Configure logging
logger = logging.getLogger(__name__)


class EndpointMethods:
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
    
    async def query_agent(self, request: QueryRequest, correlation_id: str) -> QueryResponse:
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
            logger.info(f"Processing agent query [{correlation_id}]: {request.query[:100]}...")
            
            # Convert API request to agent request
            agent_request = AgentQueryRequest(
                query=request.query,
                task_type=request.task_type.value,
                framework=request.framework.value if request.framework else None,
                context_limit=request.context_limit,
                include_sources=request.include_sources
            )
            
            # Process query with agent
            agent_response = await self.agent.query(agent_request)
            
            # Convert agent response to API response
            response = QueryResponse(
                content=agent_response.content,
                sources=agent_response.sources,
                task_type=request.task_type,
                model_used=agent_response.model_used,
                token_usage=agent_response.token_usage,
                conversation_id=request.conversation_id
            )
            
            # Save to conversation if conversation_id provided
            if request.conversation_id:
                await self._add_message_to_conversation(
                    request.conversation_id,
                    request.query,
                    agent_response.content,
                    agent_response.sources,
                    agent_response.token_usage
                )
            
            logger.info(f"Query processed successfully [{correlation_id}]")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed [{correlation_id}]: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Query processing failed: {str(e)}"
            )
    
    async def stream_agent_query(self, request: QueryRequest, correlation_id: str):
        """
        Stream agent query response for long-running queries.
        
        Args:
            request: Query request data.
            correlation_id: Request correlation ID.
            
        Yields:
            str: Streaming response chunks.
        """
        try:
            logger.info(f"Starting streaming query [{correlation_id}]: {request.query[:100]}...")
            
            # For now, we'll implement basic streaming by yielding the full response
            # In a full implementation, this would integrate with the agent's streaming capabilities
            response = await self.query_agent(request, correlation_id)
            
            # Yield response in chunks
            content = response.content
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
    
    async def create_conversation(self, request: ConversationCreate, user_id: str) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            request: Conversation creation request.
            user_id: User ID.
            
        Returns:
            Conversation: Created conversation.
        """
        try:
            conversations = self.db.ai_chatbot_conversations
            
            # Generate title if not provided
            title = request.title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            
            conversation_data = {
                "user_id": user_id,
                "title": title,
                "messages": [],
                "creation_date": datetime.utcnow(),
                "update_date": datetime.utcnow()
            }
            
            # Add initial message if provided
            if request.initial_message:
                conversation_data["messages"].append({
                    "role": "user",
                    "content": request.initial_message,
                    "timestamp": datetime.utcnow(),
                    "sources": None,
                    "token_usage": None
                })
            
            result = await conversations.insert_one(conversation_data)
            conversation_data["_id"] = result.inserted_id
            
            return self._convert_conversation_document(conversation_data)
            
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise HTTPException(
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
            conversations = self.db.ai_chatbot_conversations
            
            # Calculate skip value
            skip = (page - 1) * page_size
            
            # Get total count
            total = await conversations.count_documents({"user_id": user_id})
            
            # Get conversations
            cursor = conversations.find(
                {"user_id": user_id}
            ).sort("update_date", -1).skip(skip).limit(page_size)
            
            conversation_docs = await cursor.to_list(length=page_size)
            conversation_list = [
                self._convert_conversation_document(doc)
                for doc in conversation_docs
            ]
            
            return ConversationList(
                conversations=conversation_list,
                total=total,
                page=page,
                page_size=page_size
            )
            
        except Exception as e:
            logger.error(f"Failed to get conversations: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get conversations: {str(e)}"
            )
    
    async def get_conversation(self, conversation_id: str, user_id: str) -> Conversation:
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
            
            conversations = self.db.ai_chatbot_conversations
            
            conversation_doc = await conversations.find_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id
            })
            
            if not conversation_doc:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found"
                )
            
            return self._convert_conversation_document(conversation_doc)
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error(f"Failed to get conversation: {e}")
            raise HTTPException(
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
            
            conversations = self.db.ai_chatbot_conversations
            
            update_data = {"update_date": datetime.utcnow()}
            
            if request.title is not None:
                update_data["title"] = request.title
            
            result = await conversations.update_one(
                {"_id": ObjectId(conversation_id), "user_id": user_id},
                {"$set": update_data}
            )
            
            if result.matched_count == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found"
                )
            
            return await self.get_conversation(conversation_id, user_id)
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error(f"Failed to update conversation: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update conversation: {str(e)}"
            )
    
    async def delete_conversation(self, conversation_id: str, user_id: str) -> Dict[str, str]:
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
            
            conversations = self.db.ai_chatbot_conversations
            
            result = await conversations.delete_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id
            })
            
            if result.deleted_count == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found"
                )
            
            return {"message": "Conversation deleted successfully"}
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error(f"Failed to delete conversation: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete conversation: {str(e)}"
            )
    
    # Knowledge Base Management Methods
    
    async def update_knowledge_base(
        self,
        request: KnowledgeBaseUpdate,
        background_tasks: BackgroundTasks
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
            operation_id = f"kb_update_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            background_tasks.add_task(
                self._update_knowledge_base_background,
                operation_id,
                request
            )
            
            return {
                "message": "Knowledge base update started",
                "operation_id": operation_id
            }
            
        except Exception as e:
            logger.error(f"Failed to start knowledge base update: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start update: {str(e)}"
            )
    
    async def get_knowledge_base_status(self) -> KnowledgeBaseStatus:
        """
        Get knowledge base status.
        
        Returns:
            KnowledgeBaseStatus: Current status.
        """
        try:
            knowledge_base = self.db.knowledge_base
            
            # Get document and chunk counts
            document_count = await knowledge_base.count_documents({})
            
            # Get unique file count (chunks from same file)
            pipeline = [
                {"$group": {"_id": "$path"}},
                {"$count": "unique_files"}
            ]
            
            unique_files_result = await knowledge_base.aggregate(pipeline).to_list(1)
            unique_files = unique_files_result[0]["unique_files"] if unique_files_result else 0
            
            # Get repository info from environment
            import os
            repository_url = os.getenv("REMOTE_REPO_URL", "")
            
            return KnowledgeBaseStatus(
                status="idle",  # This would be tracked in a real implementation
                document_count=unique_files,
                chunk_count=document_count,
                repository_url=repository_url,
                last_update=None  # This would be tracked in database
            )
            
        except Exception as e:
            logger.error(f"Failed to get knowledge base status: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get status: {str(e)}"
            )
    
    async def upload_document(
        self,
        file: UploadFile,
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
            
            # Process document (this would integrate with document processing pipeline)
            # For now, we'll create a placeholder implementation
            
            document_info = DocumentInfo(
                id=f"doc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                filename=file.filename or "unknown",
                file_type=file.content_type or "unknown",
                size=len(content),
                upload_date=datetime.utcnow(),
                description=description,
                chunk_count=1  # Placeholder
            )
            
            logger.info(f"Document uploaded: {file.filename} ({len(content)} bytes)")
            return document_info
            
        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload document: {str(e)}"
            )
    
    # File Generation Methods
    
    async def generate_file(self, request: FileGenerationRequest) -> GeneratedFile:
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
                    raise HTTPException(
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
            
            return generated_file
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error(f"Failed to generate file: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate file: {str(e)}"
            )
    
    async def create_file_package(self, files: List[GeneratedFile]) -> FilePackage:
        """
        Create a package from multiple files.
        
        Args:
            files: List of files to package.
            
        Returns:
            FilePackage: File package information.
        """
        try:
            total_size = sum(file.size for file in files)
            package_name = f"genericsuite_codegen_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            package = FilePackage(
                package_name=package_name,
                files=files,
                format="zip",
                total_size=total_size
            )
            
            return package
            
        except Exception as e:
            logger.error(f"Failed to create file package: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create package: {str(e)}"
            )
    
    # Search Methods
    
    async def search_knowledge_base(self, query: SearchQuery) -> SearchResponse:
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
            
            return SearchResponse(
                results=results,
                total_results=len(results),
                query=query.query,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
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
            knowledge_base = self.db.knowledge_base
            conversations = self.db.ai_chatbot_conversations
            
            kb_count = await knowledge_base.count_documents({})
            conv_count = await conversations.count_documents({})
            
            # Get agent info
            agent_info = self.agent.get_model_info()
            
            stats = Statistics(
                knowledge_base={
                    "total_chunks": kb_count,
                    "last_updated": None  # Would be tracked in real implementation
                },
                conversations={
                    "total_conversations": conv_count
                },
                agent=agent_info,
                system={
                    "uptime": "unknown",  # Would be tracked in real implementation
                    "memory_usage": "unknown"
                }
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get statistics: {str(e)}"
            )
    
    # Helper Methods
    
    def _convert_conversation_document(self, doc: Dict[str, Any]) -> Conversation:
        """Convert MongoDB document to Conversation model."""
        from .types import Message
        
        messages = []
        for msg_data in doc.get("messages", []):
            messages.append(Message(
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
    
    async def _add_message_to_conversation(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        sources: List[str],
        token_usage: Optional[Dict[str, int]]
    ) -> None:
        """Add messages to a conversation."""
        try:
            from bson import ObjectId
            
            conversations = self.db.ai_chatbot_conversations
            
            messages_to_add = [
                {
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.utcnow(),
                    "sources": None,
                    "token_usage": None
                },
                {
                    "role": "assistant",
                    "content": assistant_message,
                    "timestamp": datetime.utcnow(),
                    "sources": sources,
                    "token_usage": token_usage
                }
            ]
            
            await conversations.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$push": {"messages": {"$each": messages_to_add}},
                    "$set": {"update_date": datetime.utcnow()}
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to add messages to conversation: {e}")
            # Don't raise exception here as it's not critical to the main operation
    
    async def _update_knowledge_base_background(
        self,
        operation_id: str,
        request: KnowledgeBaseUpdate
    ) -> None:
        """Background task for knowledge base update."""
        try:
            logger.info(f"Starting knowledge base update [{operation_id}]")
            
            # This would integrate with the document processing pipeline
            # For now, we'll just log the operation
            
            from genericsuite_codegen.document_processing.ingestion import DocumentProcessor
            processor = DocumentProcessor()
            
            # Process repository
            repo_url = request.repository_url or os.getenv("REMOTE_REPO_URL")
            if repo_url:
                await processor.process_repository(repo_url, force_refresh=request.force_refresh)
            
            logger.info(f"Knowledge base update completed [{operation_id}]")
            
        except Exception as e:
            logger.error(f"Knowledge base update failed [{operation_id}]: {e}")


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