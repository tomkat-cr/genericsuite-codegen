# Design Document

## Overview

This design addresses the conversation management issues in the GenericSuite CodeGen chat system. The problems stem from improper state management in the frontend, incorrect API data handling in the backend, and missing user interface functionality. The solution involves fixing the conversation flow, improving data persistence, and adding essential UI features for conversation management.

## Architecture

### Frontend Architecture Changes

The ChatPage component needs restructuring to properly manage conversation state and API interactions:

1. **State Management Improvements**
   - Separate conversation creation from message sending
   - Implement proper conversation ID tracking
   - Fix conversation title generation and persistence
   - Add edit mode state for title editing

2. **API Integration Fixes**
   - Correct conversation creation API calls
   - Proper message association with conversations
   - Implement conversation update and delete operations
   - Add error handling and user feedback

3. **UI Enhancements**
   - Add inline title editing functionality
   - Implement delete confirmation dialogs
   - Improve conversation list management
   - Add loading states and error displays

### Backend Architecture Changes

The endpoint_methods.py needs fixes for proper conversation handling:

1. **Conversation Creation Logic**
   - Fix conversation creation to return proper conversation objects
   - Ensure unique title generation
   - Implement proper user association

2. **Message Storage Improvements**
   - Fix message association with correct conversations
   - Implement proper conversation updates
   - Add validation for conversation operations

3. **API Response Standardization**
   - Ensure consistent response formats
   - Proper error handling and status codes
   - Add conversation validation

## Components and Interfaces

### Frontend Components

#### ChatPage Component Updates
```typescript
interface ChatPageState {
  conversations: Conversation[]
  currentConversation: Conversation | null
  message: string
  isLoading: boolean
  isStreaming: boolean
  streamingMessage: string
  error: string | null
  editingTitleId: string | null  // New: for title editing
  editingTitle: string          // New: for title editing
}

interface ConversationOperations {
  createConversation(): Promise<Conversation>
  saveConversation(conversation: Conversation): Promise<Conversation>
  updateConversationTitle(id: string, title: string): Promise<void>
  deleteConversation(id: string): Promise<void>
  loadConversations(): Promise<Conversation[]>
}
```

#### New Components
- `ConversationTitleEditor`: Inline editing component for conversation titles
- `DeleteConfirmationDialog`: Confirmation dialog for conversation deletion
- `ConversationListItem`: Enhanced conversation item with edit/delete actions

### Backend Interface Updates

#### API Endpoints
```python
# Enhanced conversation endpoints
POST /api/conversations          # Create new conversation
GET /api/conversations           # List user conversations
GET /api/conversations/{id}      # Get specific conversation
PUT /api/conversations/{id}      # Update conversation (title, etc.)
DELETE /api/conversations/{id}   # Delete conversation
POST /api/query                  # Send message (with proper conversation handling)
```

#### Data Models
```python
class ConversationCreate(BaseModel):
    title: Optional[str] = None
    initial_message: Optional[str] = None

class ConversationUpdate(BaseModel):
    title: Optional[str] = None

class Message(BaseModel):
    id: str
    role: Literal['user', 'assistant']
    content: str
    timestamp: datetime
    sources: Optional[List[SourceInfo]] = None
```

## Data Models

### Conversation Storage Schema
```python
{
    "_id": ObjectId,
    "user_id": str,
    "title": str,
    "messages": [
        {
            "id": str,
            "role": "user" | "assistant",
            "content": str,
            "timestamp": datetime,
            "sources": Optional[List],
            "token_usage": Optional[Dict]
        }
    ],
    "creation_date": datetime,
    "update_date": datetime
}
```

### Frontend State Management
```typescript
interface ConversationState {
  // Conversation list management
  conversations: Map<string, Conversation>
  currentConversationId: string | null
  
  // UI state
  isLoading: boolean
  error: string | null
  editingTitleId: string | null
  
  // Message handling
  pendingMessage: string
  isStreaming: boolean
  streamingContent: string
}
```

## Error Handling

### Frontend Error Handling
1. **API Communication Errors**
   - Network failures during conversation operations
   - Invalid response formats
   - Authentication/authorization errors

2. **User Input Validation**
   - Empty conversation titles
   - Invalid message content
   - Conversation not found errors

3. **State Management Errors**
   - Conversation synchronization issues
   - Message ordering problems
   - UI state inconsistencies

### Backend Error Handling
1. **Database Operations**
   - Connection failures
   - Document not found errors
   - Validation failures

2. **Conversation Management**
   - Invalid conversation IDs
   - User permission errors
   - Message association failures

3. **API Response Handling**
   - Proper HTTP status codes
   - Consistent error message formats
   - Detailed error information for debugging

## Testing Strategy

### Frontend Testing
1. **Unit Tests**
   - Conversation state management functions
   - Message handling logic
   - Title editing functionality
   - Delete confirmation flows

2. **Integration Tests**
   - API communication with backend
   - Conversation CRUD operations
   - Message streaming functionality
   - Error handling scenarios

3. **User Interface Tests**
   - Conversation list rendering
   - Message display and formatting
   - Title editing interactions
   - Delete confirmation dialogs

### Backend Testing
1. **Unit Tests**
   - Conversation creation logic
   - Message storage functions
   - Title update operations
   - Delete operations

2. **API Tests**
   - Endpoint response validation
   - Error handling verification
   - Authentication/authorization
   - Data serialization/deserialization

3. **Database Tests**
   - Conversation document operations
   - Message array management
   - User isolation verification
   - Data consistency checks

### End-to-End Testing
1. **Conversation Lifecycle**
   - Create conversation → Send messages → Save → Reload → Verify persistence
   - Edit title → Save → Verify update
   - Delete conversation → Verify removal

2. **Multi-Conversation Scenarios**
   - Create multiple conversations with different titles
   - Switch between conversations
   - Verify message isolation
   - Test concurrent operations

3. **Error Recovery**
   - Network interruption during operations
   - Invalid data handling
   - User permission scenarios
   - Database connectivity issues