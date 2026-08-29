
from typing import Dict, List, Optional
from datetime import datetime
import datetime as dt

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from genericsuite_codegen.agent.types import (
    AgentContext,
)
from genericsuite_codegen.api.types import (
    BaseResponse,
)


# Conversation Models

class Message(BaseModel):
    """Individual message in a conversation."""
    id: str = Field(description="Unique message identifier")
    role: str = Field(description="Message role (user/assistant)")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(
        default=dt.datetime.now(dt.UTC),
        description="Message timestamp")
    sources: Optional[List[str]] = Field(
        default=None, description="Source documents for assistant messages")
    task_type: Optional[str] = Field(
        default=None, description="Type of task performed")
    model_used: Optional[str] = Field(
        default=None, description="Model used for generation")
    token_usage: Optional[Dict[str, int]] = Field(
        default=None, description="Token usage for this message")

    @field_validator('role')
    def validate_role(cls, v):
        """Validate message role."""
        if v not in ['user', 'assistant']:
            raise ValueError('Role must be either "user" or "assistant"')
        return v


class ConversationInit(BaseModel):
    """Request model for creating a new conversation."""
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID")
    context: Optional[AgentContext] = Field(
        default=None, description="Conversation context")
    new_conversation: Optional[bool] = Field(
        default=False,
        description="Indicates if a new conversation was created")
    error: Optional[str] = Field(
        default=None, description="Error message if any")


class ConversationCreate(BaseModel):
    """Request model for creating a new conversation."""
    title: Optional[str] = Field(
        default=None, max_length=200, description="Conversation title")
    initial_message: Optional[str] = Field(
        default=None, description="Initial message to start the conversation")


class ConversationUpdate(BaseModel):
    """Request model for updating a conversation."""
    title: Optional[str] = Field(
        default=None, max_length=200, description="New conversation title")


class Conversation(BaseModel):
    """Conversation model."""
    id: str = Field(description="Conversation ID")
    title: str = Field(description="Conversation title")
    messages: List[Message] = Field(
        default_factory=list, description="Conversation messages")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    message_count: int = Field(
        description="Number of messages in conversation")


class ConversationList(BaseResponse):
    """Response model for conversation list."""
    conversations: List[Conversation] = Field(
        description="List of conversations")
