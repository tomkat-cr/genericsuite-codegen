import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Link } from 'react-router-dom'
import { Database, MessageSquare, Code, ArrowRight } from 'lucide-react'

const gsDocumentationUrl: string = "https://genericsuite.carlosjramirez.com"
const pydanticAiDocumentationUrl: string = "https://ai.pydantic.dev/"

export function HomePage() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Welcome to GenericSuite CodeGen
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          A comprehensive RAG AI system for querying GenericSuite documentation 
          and generating configuration files, tools, and application code.
        </p>
      </div>

      {/* Feature Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="flex items-center space-x-2">
              <Database className="h-5 w-5 text-primary" />
              <CardTitle>Knowledge Base</CardTitle>
            </div>
            <CardDescription>
              Manage and update the GenericSuite documentation repository
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Keep your knowledge base up-to-date with the latest GenericSuite 
              documentation and examples.
            </p>
            <Button asChild className="w-full">
              <Link to="/knowledge-base">
                Manage Knowledge Base
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="flex items-center space-x-2">
              <MessageSquare className="h-5 w-5 text-primary" />
              <CardTitle>AI Chat</CardTitle>
            </div>
            <CardDescription>
              Interactive chat with the GenericSuite AI assistant
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Ask questions about GenericSuite and get intelligent responses 
              based on the documentation.
            </p>
            <Button asChild className="w-full">
              <Link to="/chat">
                Start Chatting
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="flex items-center space-x-2">
              <Code className="h-5 w-5 text-primary" />
              <CardTitle>Code Generation</CardTitle>
            </div>
            <CardDescription>
              Generate JSON configs, Python tools, and application code
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Create GenericSuite configurations, Langchain tools, MCP tools, 
              and frontend/backend code.
            </p>
            <Button asChild className="w-full">
              <Link to="/code-generation">
                Generate Code
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <div className="text-center p-6 bg-muted rounded-lg">
          <div className="text-2xl font-bold text-primary">RAG</div>
          <div className="text-sm text-muted-foreground">
            Retrieval-Augmented Generation
          </div>
        </div>
        <div className="text-center p-6 bg-muted rounded-lg">
          <div className="text-2xl font-bold text-primary">
            <a
              href={pydanticAiDocumentationUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              AI
            </a>
          </div>
          <div className="text-sm text-muted-foreground">
            <a
              href={pydanticAiDocumentationUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              Powered by Pydantic AI
            </a>
          </div>
        </div>
        <div className="text-center p-6 bg-muted rounded-lg">
          <div className="text-2xl font-bold text-primary">
            <a
                href={gsDocumentationUrl}
                target="_blank"
                rel="noopener noreferrer"
            >
              GenericSuite
            </a>
          </div>
          <div className="text-sm text-muted-foreground">
            <a
              href={gsDocumentationUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              Documentation & Examples
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}