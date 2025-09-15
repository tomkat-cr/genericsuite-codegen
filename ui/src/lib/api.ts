/**
 * API service for conversation management
 */

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: Array<{
    title: string
    path: string
    similarity: number
  }>
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  created_at: string
  updated_at: string
  message_count: number
}

export interface ConversationList {
  conversations: Conversation[]
  total: number
  page: number
  page_size: number
}

export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
}

export interface ConversationCreateRequest {
  initial_message?: string
  title?: string
}

export interface ConversationUpdateRequest {
  title?: string
}

export interface QueryRequest {
  query: string
  conversation_id?: string
  task_type?: string
  framework?: string
  context_limit?: number
  include_sources?: boolean
}

export const baseUrl = process.env.UI_API_BASE_URL

class ApiService {

  private async handleResponse<T>(response: Response): Promise<ApiResponse<T>> {
    try {
      if (response.ok) {
        const data = await response.json()
        return { success: true, data }
      } else {
        const errorData = await response.json().catch(() => ({}))
        return { 
          success: false, 
          error: errorData.detail || `HTTP ${response.status}: ${response.statusText}` 
        }
      }
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error occurred' 
      }
    }
  }

  async getConversations(page = 1, pageSize = 20): Promise<ApiResponse<ConversationList>> {
    console.log('getConversations | baseUrl: ', baseUrl)
    const response = await fetch(`${baseUrl}/conversations?page=${page}&page_size=${pageSize}`)
    return this.handleResponse<ConversationList>(response)
  }

  async getConversation(conversationId: string): Promise<ApiResponse<Conversation>> {
    const response = await fetch(`${baseUrl}/conversations/${conversationId}`)
    return this.handleResponse<Conversation>(response)
  }

  async createConversation(request: ConversationCreateRequest): Promise<ApiResponse<Conversation>> {
    const response = await fetch(`${baseUrl}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    })
    return this.handleResponse<Conversation>(response)
  }

  async updateConversation(conversationId: string, request: ConversationUpdateRequest) {
    const response = await fetch(`${baseUrl}/conversations/${conversationId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    })
    return this.handleResponse(response)
  }

  async deleteConversation(conversationId: string) {
    const response = await fetch(`${baseUrl}/conversations/${conversationId}`, {
      method: 'DELETE'
    })
    return this.handleResponse(response)
  }

  async queryAgent(request: QueryRequest) {
    const response = await fetch(`${baseUrl}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    })
    return this.handleResponse(response)
  }
}

export const apiService = new ApiService()