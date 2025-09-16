# Technology Stack & Build System

## Backend Stack
- **Python**: 3.12+ (managed with Poetry)
- **FastAPI**: Modern Python web framework
- **Pydantic AI**: AI agent framework with tool integration
- **MongoDB**: Document database with vector search capabilities
- **Sentence Transformers**: Text embeddings for semantic search
- **OpenAI**: LLM provider support
- **FastMCP**: MCP server implementation

## Frontend Stack
- **React 18**: Modern React with TypeScript
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **Shadcn/ui**: Modern UI component library
- **Lucide React**: Icon library

## Infrastructure
- **Docker**: Containerization and deployment
- **Poetry**: Python dependency management
- **npm**: Node.js package management (workspaces)
- **Nginx**: Reverse proxy and static file serving

## Code Quality Tools
- **Python**: Black (formatting), isort (imports), flake8 (linting), mypy (type checking)
- **TypeScript**: ESLint (linting), Prettier (formatting)

## Common Commands

### Development
```bash
# Start full development environment
make dev

# Start individual services
make run-db-only          # MongoDB only
cd server && make run     # Backend only
cd ui && make run         # Frontend only
cd mcp-server && make run # MCP server only
```

### Production
```bash
make run                  # Start all production services
make status               # Check service status
make logs-f               # Follow logs
make down                 # Stop services
make clean-docker         # Clean up containers and volumes
```

### Setup & Installation
```bash
make init-app-environment # Initialize .env and config files
make install              # Install all dependencies
make build                # Build all components
```

### Code Quality
```bash
# Backend
cd server && make format && make lint && make test

# Frontend  
cd ui && make format && make lint

# MCP Server
cd mcp-server && make format && make lint && make test
```

## Environment Requirements
- **Node.js**: 18.0+
- **Python**: 3.12+
- **Docker**: For containerized deployment
- **OpenAI API Key**: Required for AI functionality