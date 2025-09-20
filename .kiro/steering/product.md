# GenericSuite CodeGen Product Overview

GenericSuite CodeGen is an AI-powered RAG (Retrieval-Augmented Generation) system that generates JSON configuration files, Python tools, and application code following GenericSuite patterns.

## Core Features

- **AI-Powered Code Generation**: Generate JSON configurations, Python tools, and application code
- **Intelligent Knowledge Base**: Vector search through GenericSuite documentation using MongoDB
- **Multiple Code Types**: Support for JSON configs, LangChain tools, MCP tools, frontend, and backend code
- **Web Interface**: React-based UI for interactive code generation
- **MCP Integration**: Model Context Protocol server for AI development environments
- **Real-time Streaming**: Streaming responses for better user experience

## Target Use Cases

- Generating GenericSuite-compatible configurations
- Creating Python tools (LangChain and MCP)
- Frontend and backend code scaffolding
- Documentation-driven development assistance
- AI development workflow integration

## Architecture

The system combines:
- FastAPI backend with Pydantic AI agent
- React frontend with TypeScript
- MongoDB with vector search capabilities
- MCP server for external AI tool integration
- Docker-based deployment infrastructure