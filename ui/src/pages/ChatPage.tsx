import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Send, 
  MessageSquare, 
  Bot, 
  User, 
  Save, 
  Trash2, 
  Plus,
  ExternalLink,
  Copy,
  Check
} from 'lucide-react'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  sources?: Array<{
    title: string
    path: string
    similarity: number
  }>
}

interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  updatedAt: Date
}

export function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null)
  const [message, setMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingMessage, setStreamingMessage] = useState('')
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [currentConversation?.messages, streamingMessage])

  // Load conversations on mount
  useEffect(() => {
    loadConversations()
  }, [])

  const loadConversations = async () => {
    try {
      const response = await fetch('/api/conversations')
      if (response.ok) {
        const data = await response.json()
        setConversations(data.conversations || [])
      }
    } catch (error) {
      console.error('Failed to load conversations:', error)
    }
  }

  const createNewConversation = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: 'New Conversation',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    }
    setCurrentConversation(newConversation)
    setConversations(prev => [newConversation, ...prev])
  }

  const saveConversation = async (conversation: Conversation) => {
    try {
      const response = await fetch('/api/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(conversation)
      })
      
      if (response.ok) {
        const saved = await response.json()
        setConversations(prev => 
          prev.map(conv => conv.id === conversation.id ? saved : conv)
        )
        if (currentConversation?.id === conversation.id) {
          setCurrentConversation(saved)
        }
      }
    } catch (error) {
      console.error('Failed to save conversation:', error)
    }
  }

  const deleteConversation = async (conversationId: string) => {
    try {
      const response = await fetch(`/api/conversations/${conversationId}`, {
        method: 'DELETE'
      })
      
      if (response.ok) {
        setConversations(prev => prev.filter(conv => conv.id !== conversationId))
        if (currentConversation?.id === conversationId) {
          setCurrentConversation(null)
        }
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error)
    }
  }

  const sendMessage = async () => {
    if (!message.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: message.trim(),
      timestamp: new Date()
    }

    // Create conversation if none exists
    let conversation = currentConversation
    if (!conversation) {
      conversation = {
        id: Date.now().toString(),
        title: message.trim().slice(0, 50) + (message.length > 50 ? '...' : ''),
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date()
      }
      setCurrentConversation(conversation)
      setConversations(prev => [conversation!, ...prev])
    }

    // Add user message
    const updatedConversation = {
      ...conversation,
      messages: [...conversation.messages, userMessage],
      updatedAt: new Date()
    }
    setCurrentConversation(updatedConversation)
    setMessage('')
    setIsLoading(true)
    setIsStreaming(true)
    setStreamingMessage('')
    setError(null)

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMessage.content,
          conversationId: conversation.id
        })
      })

      if (!response.ok) {
        throw new Error('Failed to get response from AI')
      }

      // Handle streaming response
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let assistantContent = ''
      let sources: Message['sources'] = []

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value)
          const lines = chunk.split('\n')

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                if (data.type === 'content') {
                  assistantContent += data.content
                  setStreamingMessage(assistantContent)
                } else if (data.type === 'sources') {
                  sources = data.sources
                } else if (data.type === 'done') {
                  // Streaming complete
                  break
                }
              } catch (e) {
                // Ignore malformed JSON
              }
            }
          }
        }
      }

      // Create assistant message
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: assistantContent,
        timestamp: new Date(),
        sources
      }

      // Update conversation with assistant message
      const finalConversation = {
        ...updatedConversation,
        messages: [...updatedConversation.messages, assistantMessage],
        updatedAt: new Date()
      }
      
      setCurrentConversation(finalConversation)
      await saveConversation(finalConversation)

    } catch (error) {
      setError('Failed to get response from AI. Please try again.')
      console.error('Chat error:', error)
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
      <div className="w-80 flex flex-col">
        <Card className="flex-1">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">Conversations</CardTitle>
              <Button size="sm" onClick={createNewConversation}>
                <Plus className="h-4 w-4 mr-1" />
                New
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[calc(100vh-12rem)]">
              <div className="space-y-2 p-4 pt-0">
                {conversations.map((conv) => (
                  <div
                    key={conv.id}
                    className={`p-3 rounded-lg cursor-pointer transition-colors ${
                      currentConversation?.id === conv.id
                        ? 'bg-primary/10 border border-primary/20'
                        : 'hover:bg-muted'
                    }`}
                    onClick={() => setCurrentConversation(conv)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm truncate">{conv.title}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatTimestamp(conv.updatedAt)}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {conv.messages.length} messages
                        </p>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation()
                          deleteConversation(conv.id)
                        }}
                        className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
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
                {currentConversation ? currentConversation.title : 'Select or start a conversation'}
              </CardTitle>
              {currentConversation && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => saveConversation(currentConversation)}
                >
                  <Save className="h-4 w-4 mr-1" />
                  Save
                </Button>
              )}
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
                          <div className={`p-3 rounded-lg ${
                            msg.role === 'user' 
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
                            <span>{formatTimestamp(msg.timestamp)}</span>
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
    </div>
  )
}