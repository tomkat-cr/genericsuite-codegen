import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  Send,
  MessageSquare,
  Bot,
  User,
  Trash2,
  Plus,
  ExternalLink,
  Copy,
  Check,
  Edit2,
  X,
  Save
} from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { apiService } from '@/lib/api'

import type { Conversation, Message } from '@/lib/api'

export function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null)
  const [message, setMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingMessage, setStreamingMessage] = useState('')
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isCreatingConversation, setIsCreatingConversation] = useState(false)
  const [conversationError, setConversationError] = useState<string | null>(null)
  const [isLoadingConversation, setIsLoadingConversation] = useState(false)

  // Title editing state
  const [editingTitleId, setEditingTitleId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState('')
  const [titleError, setTitleError] = useState<string | null>(null)
  const [isUpdatingTitle, setIsUpdatingTitle] = useState(false)

  // Delete confirmation state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [conversationToDelete, setConversationToDelete] = useState<Conversation | null>(null)
  const [isDeletingConversation, setIsDeletingConversation] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [currentConversation?.messages, streamingMessage])

  // Clear errors when switching conversations
  useEffect(() => {
    setError(null)
    setConversationError(null)
    // Cancel any ongoing title editing when switching conversations
    if (editingTitleId && currentConversation?.id !== editingTitleId) {
      cancelEditingTitle()
    }
  }, [currentConversation?.id, editingTitleId])

  // Load conversations on mount
  useEffect(() => {
    loadConversations()
  }, [])

  // Handle clicking outside to cancel title editing
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (editingTitleId && !(event.target as Element)?.closest('.conversation-item')) {
        cancelEditingTitle()
      }
    }

    if (editingTitleId) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [editingTitleId])

  const loadConversations = async () => {
    const result = await apiService.getConversations()
    if (result.success && result.data) {
      setConversations(result.data.conversations || [])
      setConversationError(null)
    } else {
      setConversationError(result.error || 'Failed to load conversations')
    }
  }

  const createNewConversation = async (initialMessage?: string) => {
    if (isCreatingConversation) return null

    setIsCreatingConversation(true)
    setConversationError(null)

    try {
      const result = await apiService.createConversation({
        initial_message: initialMessage,
        title: initialMessage ? undefined : 'New Conversation'
      })

      if (result.success && result.data) {
        setConversations(prev => [result.data!, ...prev])
        setCurrentConversation(result.data)
        return result.data
      } else {
        setConversationError(result.error || 'Failed to create conversation')
        return null
      }
    } catch (error) {
      console.error('Failed to create conversation:', error)
      setConversationError('Failed to create conversation')
      return null
    } finally {
      setIsCreatingConversation(false)
    }
  }



  const openDeleteDialog = (conversation: Conversation) => {
    setConversationToDelete(conversation)
    setDeleteDialogOpen(true)
  }

  const closeDeleteDialog = () => {
    setDeleteDialogOpen(false)
    setConversationToDelete(null)
  }

  const confirmDeleteConversation = async () => {
    if (!conversationToDelete) return

    setIsDeletingConversation(true)
    setConversationError(null)

    try {
      const result = await apiService.deleteConversation(conversationToDelete.id)
      if (result.success) {
        // Remove from conversations list
        setConversations(prev => prev.filter(conv => conv.id !== conversationToDelete.id))

        // Clear current conversation if it's the one being deleted
        if (currentConversation?.id === conversationToDelete.id) {
          setCurrentConversation(null)
        }

        // Close dialog
        closeDeleteDialog()
        setConversationError(null)
      } else {
        setConversationError(result.error || 'Failed to delete conversation')
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error)
      setConversationError('Failed to delete conversation')
    } finally {
      setIsDeletingConversation(false)
    }
  }

  const startEditingTitle = (conversation: Conversation) => {
    setEditingTitleId(conversation.id)
    setEditingTitle(conversation.title)
    setTitleError(null)
  }

  const cancelEditingTitle = () => {
    setEditingTitleId(null)
    setEditingTitle('')
    setTitleError(null)
  }

  const saveConversationTitle = async (conversationId: string) => {
    const trimmedTitle = editingTitle.trim()

    // Validate title
    if (!trimmedTitle) {
      setTitleError('Title cannot be empty')
      return
    }

    if (trimmedTitle.length > 100) {
      setTitleError('Title must be 100 characters or less')
      return
    }

    setIsUpdatingTitle(true)
    setTitleError(null)

    try {
      const result = await apiService.updateConversation(conversationId, {
        title: trimmedTitle
      })

      if (result.success) {
        // Update conversations list
        setConversations(prev =>
          prev.map(conv =>
            conv.id === conversationId
              ? { ...conv, title: trimmedTitle }
              : conv
          )
        )

        // Update current conversation if it's the one being edited
        if (currentConversation?.id === conversationId) {
          setCurrentConversation(prev =>
            prev ? { ...prev, title: trimmedTitle } : null
          )
        }

        setEditingTitleId(null)
        setEditingTitle('')
        setConversationError(null)
      } else {
        setTitleError(result.error || 'Failed to update title')
      }
    } catch (error) {
      console.error('Failed to update conversation title:', error)
      setTitleError('Failed to update title')
    } finally {
      setIsUpdatingTitle(false)
    }
  }

  const handleTitleKeyPress = (e: React.KeyboardEvent, conversationId: string) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      saveConversationTitle(conversationId)
    } else if (e.key === 'Escape') {
      e.preventDefault()
      cancelEditingTitle()
    }
  }

  const loadConversation = async (conversationId: string) => {
    if (isLoadingConversation) return

    setIsLoadingConversation(true)
    const result = await apiService.getConversation(conversationId)
    if (result.success && result.data) {
      setCurrentConversation(result.data)
      setConversationError(null)
    } else {
      setConversationError(result.error || 'Failed to load conversation')
    }
    setIsLoadingConversation(false)
  }

  const sendMessage = async () => {
    if (!message.trim() || isLoading) return

    const messageContent = message.trim()
    setMessage('')
    setError(null)
    setConversationError(null)

    // Create conversation if none exists
    let conversation = currentConversation
    if (!conversation) {
      conversation = await createNewConversation(messageContent)
      if (!conversation) {
        setMessage(messageContent) // Restore message on failure
        return
      }
    }

    // Optimistically update UI with user message
    const tempUserMessage: Message = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: messageContent,
      timestamp: new Date().toISOString()
    }

    const optimisticConversation = {
      ...conversation,
      messages: [...conversation.messages, tempUserMessage],
      updated_at: new Date().toISOString(),
      message_count: conversation.message_count + 1
    }
    setCurrentConversation(optimisticConversation)

    setIsLoading(true)
    setIsStreaming(true)
    setStreamingMessage('')

    try {
      const result = await apiService.queryAgent({
        query: messageContent,
        conversation_id: conversation.id
      })

      if (result.success) {
        // Update conversation with real messages from backend
        await loadConversation(conversation.id)
      } else {
        throw new Error(result.error || 'Failed to get response from AI')
      }

    } catch (error) {
      setError('Failed to get response from AI. Please try again.')
      console.error('Chat error:', error)

      // Revert optimistic update on error
      setCurrentConversation(conversation)
      setMessage(messageContent) // Restore message for retry
    } finally {
      setIsLoading(false)
      setIsStreaming(false)
      setStreamingMessage('')
      textareaRef.current?.focus()
    }
  }

  const copyMessage = async (content: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedMessageId(messageId)
      setTimeout(() => setCopiedMessageId(null), 2000)
    } catch (error) {
      console.error('Failed to copy message:', error)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const formatTimestamp = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      month: 'short',
      day: 'numeric'
    }).format(date)
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-6">
      {/* Conversations Sidebar */}
      <div className="w-96 flex flex-col">
        <Card className="flex-1">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">Conversations</CardTitle>
              <Button
                size="sm"
                onClick={() => createNewConversation()}
                disabled={isCreatingConversation}
              >
                <Plus className="h-4 w-4 mr-1" />
                {isCreatingConversation ? 'Creating...' : 'New'}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {conversationError && (
              <div className="p-4 pt-0">
                <Alert className="border-red-200 bg-red-50">
                  <AlertDescription className="text-red-800 text-xs">
                    {conversationError}
                  </AlertDescription>
                </Alert>
              </div>
            )}
            <ScrollArea className="h-[calc(100vh-12rem)]">
              <div className="space-y-2 p-4 pt-0">
                {conversations.map((conv) => (
                  <div
                    key={conv.id}
                    className={`conversation-item group p-3 rounded-lg transition-colors ${currentConversation?.id === conv.id
                        ? 'bg-primary/10 border border-primary/20'
                        : 'hover:bg-muted'
                      } ${editingTitleId === conv.id ? 'cursor-default' : 'cursor-pointer'}`}
                    onClick={() => editingTitleId !== conv.id && loadConversation(conv.id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        {editingTitleId === conv.id ? (
                          <div className="space-y-2">
                            <Input
                              value={editingTitle}
                              onChange={(e) => setEditingTitle(e.target.value)}
                              onKeyDown={(e) => handleTitleKeyPress(e, conv.id)}
                              className="h-7 text-sm font-medium"
                              placeholder="Enter conversation title"
                              autoFocus
                              disabled={isUpdatingTitle}
                            />
                            {titleError && (
                              <p className="text-xs text-red-600">{titleError}</p>
                            )}
                            <div className="flex gap-1">
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  saveConversationTitle(conv.id)
                                }}
                                disabled={isUpdatingTitle}
                                className="h-6 w-6 p-0"
                              >
                                <Save className="h-3 w-3" />
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  cancelEditingTitle()
                                }}
                                disabled={isUpdatingTitle}
                                className="h-6 w-6 p-0"
                              >
                                <X className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <>
                            <div className="flex items-center gap-2">
                              <p className="font-medium text-sm 1truncate1 text-wrap flex-1">{conv.title}</p>
                              <div className="flex gap-1 opacity-70 group-hover:opacity-100">
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    startEditingTitle(conv)
                                  }}
                                  className="h-6 w-6 p-0 hover:bg-blue-100 hover:text-blue-600"
                                  title="Edit conversation title"
                                >
                                  <Edit2 className="h-3 w-3" />
                                </Button>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    openDeleteDialog(conv)
                                  }}
                                  className="h-6 w-6 p-0 hover:bg-red-100 hover:text-red-600"
                                  title="Delete conversation"
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </div>
                            </div>
                            <p className="text-xs text-muted-foreground">
                              {formatTimestamp(new Date(conv.updated_at))}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {conv.message_count} messages
                            </p>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {conversations.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No conversations yet</p>
                    <p className="text-xs">Start a new chat to begin</p>
                  </div>
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Chat Interface */}
      <div className="flex-1 flex flex-col">
        <Card className="flex-1 flex flex-col">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">
                {isLoadingConversation
                  ? 'Loading conversation...'
                  : currentConversation
                    ? currentConversation.title
                    : 'Select or start a conversation'
                }
              </CardTitle>

            </div>
          </CardHeader>

          {error && (
            <div className="px-6">
              <Alert className="border-red-200 bg-red-50">
                <AlertDescription className="text-red-800">
                  {error}
                </AlertDescription>
              </Alert>
            </div>
          )}

          {conversationError && (
            <div className="px-6">
              <Alert className="border-red-200 bg-red-50">
                <AlertDescription className="text-red-800">
                  {conversationError}
                </AlertDescription>
              </Alert>
            </div>
          )}

          <CardContent className="flex-1 flex flex-col p-0">
            {/* Messages Area */}
            <ScrollArea className="flex-1 px-6">
              <div className="space-y-4 py-4">
                {currentConversation?.messages.map((msg) => (
                  <div key={msg.id} className="space-y-2">
                    <div className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`flex gap-3 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                        <div className="flex-shrink-0">
                          {msg.role === 'user' ? (
                            <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                              <User className="h-4 w-4 text-primary-foreground" />
                            </div>
                          ) : (
                            <div className="w-8 h-8 bg-secondary rounded-full flex items-center justify-center">
                              <Bot className="h-4 w-4 text-secondary-foreground" />
                            </div>
                          )}
                        </div>
                        <div className="space-y-2">
                          <div className={`p-3 rounded-lg ${msg.role === 'user'
                              ? 'bg-primary text-primary-foreground'
                              : 'bg-muted'
                            }`}>
                            <div className="flex items-start justify-between gap-2">
                              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => copyMessage(msg.content, msg.id)}
                                className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100"
                              >
                                {copiedMessageId === msg.id ? (
                                  <Check className="h-3 w-3" />
                                ) : (
                                  <Copy className="h-3 w-3" />
                                )}
                              </Button>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <span>{formatTimestamp(new Date(msg.timestamp))}</span>
                          </div>
                          {/* Source Attribution */}
                          {msg.sources && msg.sources.length > 0 && (
                            <div className="space-y-2">
                              <p className="text-xs font-medium text-muted-foreground">Sources:</p>
                              <div className="space-y-1">
                                {msg.sources.map((source, index) => (
                                  <div key={index} className="flex items-center gap-2 text-xs">
                                    <Badge variant="outline" className="text-xs">
                                      {(source.similarity * 100).toFixed(0)}% match
                                    </Badge>
                                    <span className="text-muted-foreground truncate">
                                      {source.title || source.path}
                                    </span>
                                    <Button size="sm" variant="ghost" className="h-4 w-4 p-0">
                                      <ExternalLink className="h-3 w-3" />
                                    </Button>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}

                {/* Streaming Message */}
                {isStreaming && streamingMessage && (
                  <div className="flex gap-3 justify-start">
                    <div className="flex gap-3 max-w-[80%]">
                      <div className="flex-shrink-0">
                        <div className="w-8 h-8 bg-secondary rounded-full flex items-center justify-center">
                          <Bot className="h-4 w-4 text-secondary-foreground animate-pulse" />
                        </div>
                      </div>
                      <div className="p-3 rounded-lg bg-muted">
                        <p className="text-sm whitespace-pre-wrap">{streamingMessage}</p>
                        <div className="mt-2">
                          <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>

            <Separator />

            {/* Message Input */}
            <div className="p-4 space-y-3">
              <div className="flex gap-2">
                <Textarea
                  ref={textareaRef}
                  placeholder="Ask about GenericSuite documentation, request code generation, or get help with configurations..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={handleKeyPress}
                  className="min-h-[60px] resize-none"
                  disabled={isLoading}
                />
                <Button
                  onClick={sendMessage}
                  disabled={!message.trim() || isLoading}
                  size="lg"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Press Enter to send, Shift+Enter for new line</span>
                {isLoading && <span>AI is thinking...</span>}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Conversation</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{conversationToDelete?.title}"?
              This action cannot be undone and will permanently remove all messages in this conversation.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={closeDeleteDialog}
              disabled={isDeletingConversation}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={confirmDeleteConversation}
              disabled={isDeletingConversation}
            >
              {isDeletingConversation ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}