import uuid
from typing import Dict, Any, List, Optional

from bson import ObjectId

from genericsuite_codegen.agent.types import AgentContext
from genericsuite_codegen.conversations.types import (
    ConversationInit,
    ConversationCreate,
    ConversationUpdate,
    Conversation,
    ConversationList,
    Message,
)
from genericsuite_codegen.api.types import (
    ConversationStatistics,
)
from genericsuite_codegen.database.setup import get_database_connection
from genericsuite_codegen.utilities import (
    std_error_response,
    std_response,
    get_utcnow_fmt,
    get_utcnow,
)
from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_warning,
    log_error,
)


DEBUG = False


class ConversationsService:

    def __init__(self, conversation_id: Optional[str] = None):
        """Initialize endpoint methods."""
        self.db = get_database_connection()
        self.conversation_id = conversation_id
        self.context = None
        self.new_conversation = False

    # Methods for Query operations

    async def init(
        self,
        query: str,
        user_id: str
    ) -> ConversationInit:
        """
        Initialize a new conversation.

        Args:
            request_conversation_id: Conversation ID.
            query: Query to initialize the conversation.
            user_id: User ID.

        Returns:
            QueryResponse: Agent response.

        Raises:
            HTTPException: If query processing fails.
        """

        self.context = None
        self.new_conversation = False
        if self.conversation_id:
            # Get conversation context if conversation exists
            self.context = await self._get_conversation_context(
                self.conversation_id)
        else:
            self.new_conversation = True

            # Create new conversation with the query as initial message
            create_request = ConversationCreate(
                initial_message=query.strip()
            )

            create_result = await self.create(
                create_request, user_id)

            if create_result.error:
                log_error(
                    "Failed to create conversation:"
                    f" {create_result.error_message}")
                return ConversationInit(
                    error=create_result.error_message)

            self.conversation_id = create_result.result.id
            _ = DEBUG and log_debug(
                f"Created new conversation {self.conversation_id} for query")

        _ = DEBUG and log_debug(
            f"Conversation {self.conversation_id} context: {self.context}")

        return ConversationInit(
            conversation_id=self.conversation_id,
            new_conversation=self.new_conversation,
            context=self.context)

    async def save_message(
        self,
        query: str,
        content: str,
        sources: List[str],
        task_type: Optional[str] = None,
        model_used: Optional[str] = None,
        token_usage: Optional[int] = None,
    ) -> None:
        # Save messages to conversation (only if conversation already
        # existed)
        if self.new_conversation:
            # For new conversations, just add the assistant response
            # (user message was already added during conversation creation)
            await self._add_assistant_message_to_conversation(
                self.conversation_id,
                content,
                sources,
                token_usage
            )
        else:
            await self._add_message_to_conversation(
                self.conversation_id,
                query,
                content,
                sources,
                task_type,
                model_used,
                token_usage
            )

    # Methods for Conversation Management

    async def create(
        self,
        request: ConversationCreate,
        user_id: str
    ) -> Dict[str, str]:
        """
        Create a new conversation.

        Args:
            request: Conversation creation request.
            user_id: User ID.

        Returns:
            Dict[str, str]: standardized response with create
                conversation as result=StandardGsResponse().
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
                        f"{get_utcnow_fmt()}"

            # Ensure title uniqueness for this user
            title = await self._ensure_unique_title(title, user_id)

            conversation_data = {
                "user_id": user_id,
                "title": title,
                "messages": [],
                "creation_date": get_utcnow(),
                "update_date": get_utcnow()
            }

            # Add initial message if provided
            if request.initial_message and request.initial_message.strip():
                message_id = str(uuid.uuid4())
                conversation_data["messages"].append({
                    "id": message_id,
                    "role": "user",
                    "content": request.initial_message.strip(),
                    "timestamp": get_utcnow(),
                    "sources": None,
                    "token_usage": None
                })

            result = conversations.insert_one(conversation_data)
            conversation_data["_id"] = result.inserted_id

            # Return the complete conversation object
            created_conversation = self._convert_conversation_document(
                conversation_data)

            _ = DEBUG and log_debug(
                f"Created conversation {result.inserted_id} for user"
                f" {user_id}")

            return std_response(result=created_conversation)

        except Exception as e:
            log_error(f"Failed to create conversation: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to create conversation: {str(e)}"
            )

    async def list(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, str]:
        """
        Get user conversations with pagination.

        Args:
            user_id: User ID.
            page: Page number (1-based).
            page_size: Items per page.

        Returns:
            Dict[str, str]: Paginated conversation list as
                result=ConversationList().
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
            log_error(f"Failed to get conversations: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to get conversations: {str(e)}"
            )

    async def get(
        self,
        user_id: str
    ) -> Dict[str, str]:
        """
        Get a specific conversation.

        Args:
            user_id: User ID.

        Returns:
            Dict[str, str]: Conversation data as result=Conversation().
        """
        try:

            # Validate inputs
            if not self.conversation_id or not self.conversation_id.strip():
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
                "_id": ObjectId(self.conversation_id),
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
            log_error(f"Failed to get conversation: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to get conversation: {str(e)}"
            )

    async def update(
        self,
        request: ConversationUpdate,
        user_id: str
    ) -> Dict[str, str]:
        """
        Update a conversation.

        Args:
            request: Update request.
            user_id: User ID.

        Returns:
            Dict[str, str]: Updated conversation as result=Conversation().
        """
        try:
            # Validate inputs
            if not self.conversation_id or \
               not self.conversation_id.strip():
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

            update_data = {"update_date": get_utcnow()}

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
                    "_id": {"$ne": ObjectId(self.conversation_id)}
                })

                if existing:
                    # Generate unique title
                    title = await self._ensure_unique_title(title, user_id)

                update_data["title"] = title

            # Verify conversation exists and belongs to user
            existing_conversation = conversations.find_one({
                "_id": ObjectId(self.conversation_id),
                "user_id": user_id
            })

            if not existing_conversation:
                return std_error_response(
                    status_code=404,
                    detail="Conversation not found or access denied"
                )

            result = conversations.update_one(
                {"_id": ObjectId(self.conversation_id), "user_id": user_id},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                return std_error_response(
                    status_code=500,
                    detail="Failed to update conversation"
                )

            # Get updated conversation
            updated_result = await self.get(user_id)
            if updated_result.error:
                return updated_result

            _ = DEBUG and log_debug(
                f"Updated conversation {self.conversation_id}"
                f" for user {user_id}")
            return std_response(result=updated_result.result)

        except Exception as e:
            log_error(f"Failed to update conversation: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to update conversation: {str(e)}"
            )

    async def delete(
        self,
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
            # Validate inputs
            if not self.conversation_id or not self.conversation_id.strip():
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
                "_id": ObjectId(self.conversation_id),
                "user_id": user_id
            })

            if not existing_conversation:
                return std_error_response(
                    status_code=404,
                    detail="Conversation not found or access denied"
                )

            result = conversations.delete_one({
                "_id": ObjectId(self.conversation_id),
                "user_id": user_id
            })

            if result.deleted_count == 0:
                return std_error_response(
                    status_code=500,
                    detail="Failed to delete conversation"
                )

            _ = DEBUG and log_debug(
                f"Deleted conversation {self.conversation_id}"
                f" for user {user_id}")
            return std_response(result={
                "message": "Conversation deleted successfully"})

        except Exception as e:
            log_error(f"Failed to delete conversation: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to delete conversation: {str(e)}"
            )

    # Helper Methods

    def _convert_conversation_document(
        self,
        doc: Dict[str, Any]
    ) -> Dict[str, str]:
        """Convert MongoDB document to Conversation model."""

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
                task_type=msg_data.get("task_type"),
                model_used=msg_data.get("model_used"),
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
            log_error(f"Error ensuring unique title: {e}")
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
            # Validate conversation_id
            if not conversation_id or not conversation_id.strip():
                log_error("Invalid conversation_id provided")
                return

            conversations = self.db.database.ai_chatbot_conversations

            # Verify conversation exists
            conversation_exists = conversations.find_one(
                {"_id": ObjectId(conversation_id)})
            if not conversation_exists:
                log_error(f"Conversation {conversation_id} not found")
                return

            # Generate unique ID for assistant message
            assistant_message_id = str(uuid.uuid4())

            message_to_add = {
                "id": assistant_message_id,
                "role": "assistant",
                "content": assistant_message,
                "timestamp": get_utcnow(),
                "sources": sources or [],
                "token_usage": token_usage
            }

            # Update conversation with new message
            result = conversations.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$push": {"messages": message_to_add},
                    "$set": {"update_date": get_utcnow()}
                }
            )

            if result.modified_count == 0:
                log_error(
                    "Failed to add assistant message to conversation"
                    f" {conversation_id}")
            else:
                _ = DEBUG and log_debug(
                    "Added assistant message to conversation"
                    f" {conversation_id}")

        except Exception as e:
            log_error(
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
            # Validate conversation_id
            if not conversation_id or not conversation_id.strip():
                log_warning("Invalid conversation_id provided for context")
                return None

            conversations = self.db.database.ai_chatbot_conversations

            # Get conversation with messages
            conversation_doc = conversations.find_one(
                {"_id": ObjectId(conversation_id)}
            )

            if not conversation_doc:
                log_warning(
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

            _ = DEBUG and log_debug(
                f"Retrieved context for conversation {conversation_id} "
                f"with {len(conversation_history)} messages")
            return agent_context

        except Exception as e:
            log_error(
                "Failed to get conversation context for"
                f" {conversation_id}: {e}")
            return None

    async def _add_message_to_conversation(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        sources: Optional[List[str]],
        task_type: Optional[str],
        model_used: Optional[str],
        token_usage: Optional[Dict[str, int]]
    ) -> None:
        """
        Add messages to a conversation with proper validation and ID
        generation.
        """
        try:
            # Validate conversation_id
            if not conversation_id or not conversation_id.strip():
                log_error("Invalid conversation_id provided")
                return

            conversations = self.db.database.ai_chatbot_conversations

            # Verify conversation exists
            conversation_exists = conversations.find_one(
                {"_id": ObjectId(conversation_id)})
            if not conversation_exists:
                log_error(f"Conversation {conversation_id} not found")
                return

            # Generate unique IDs for messages
            user_message_id = str(uuid.uuid4())
            assistant_message_id = str(uuid.uuid4())

            messages_to_add = [
                {
                    "id": user_message_id,
                    "role": "user",
                    "content": user_message,
                    "timestamp": get_utcnow(),
                    "sources": None,
                    "task_type": "chat",
                    "model_used": "",
                    "token_usage": None
                },
                {
                    "id": assistant_message_id,
                    "role": "assistant",
                    "content": assistant_message,
                    "timestamp": get_utcnow(),
                    "sources": sources or [],
                    "task_type": task_type,
                    "model_used": model_used,
                    "token_usage": token_usage
                }
            ]

            # Update conversation with new messages
            result = conversations.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$push": {"messages": {"$each": messages_to_add}},
                    "$set": {"update_date": get_utcnow()}
                }
            )

            if result.modified_count == 0:
                log_error(
                    "Failed to add messages to conversation"
                    f" {conversation_id}")
            else:
                _ = DEBUG and log_debug(
                    f"Added 2 messages to conversation {conversation_id}")

        except Exception as e:
            # Don't raise exception here as it's not critical to the
            # main operation
            log_error(f"Failed to add messages to conversation: {e}")

    async def statistics(self) -> ConversationStatistics:
        """
        Get conversations statistics.

        Returns:
            Dict[str, str]: System statistics as result=Statistics().
        """
        try:
            # Get knowledge base stats
            conversations = self.db.database.ai_chatbot_conversations
            conv_count = conversations.count_documents({})
            return ConversationStatistics(total_conversations=conv_count)
        except Exception as e:
            log_error(f"Failed to get conversations statistics: {e}")
            raise e
            # return ConversationStatistics(
            #   total_conversations=0, error=str(e))
