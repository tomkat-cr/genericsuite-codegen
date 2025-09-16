# Project Structure & Organization

## Root Structure
```
genericsuite-codegen/
├── server/                     # FastAPI backend application
├── ui/                        # React frontend application  
├── mcp-server/               # MCP server standalone
├── deploy/                   # Docker deployment configuration
├── scripts/                  # Utility scripts
├── local_repo_files/        # Local knowledge base files
├── local_mongodb_data/      # MongoDB data persistence
├── .env.example             # Environment template
└── package.json             # Root workspace configuration
```

## Backend Structure (`server/`)
```
server/
├── genericsuite_codegen/
│   ├── agent/              # AI agent implementation
│   ├── api/                # FastAPI endpoints and routing
│   ├── database/           # MongoDB setup and utilities
│   ├── document_processing/ # Document ingestion and processing
│   └── mcp_server/         # MCP server implementation
├── pyproject.toml          # Poetry dependencies and config
├── Makefile               # Server-specific commands
└── run-server.sh          # Server startup script
```

## Frontend Structure (`ui/`)
```
ui/
├── src/
│   ├── components/         # Reusable UI components (Shadcn/ui)
│   ├── pages/             # Application pages/routes
│   ├── lib/               # Utilities and API client
│   └── main.tsx           # Application entry point
├── package.json           # Frontend dependencies
├── tailwind.config.js     # Tailwind CSS configuration
├── vite.config.ts         # Vite build configuration
└── run-client.sh          # Frontend startup script
```

## MCP Server Structure (`mcp-server/`)
```
mcp-server/
├── genericsuite_codegen/  # Shared codebase with server
├── pyproject.toml         # MCP-specific dependencies
├── start_mcp_server.py    # MCP server entry point
└── *_config.json          # MCP client configurations
```

## Deployment Structure (`deploy/`)
```
deploy/
├── docker-compose.yml     # Service orchestration
├── docker_images/         # Custom Dockerfiles
├── nginx.conf            # Nginx reverse proxy config
└── server-entrypoint.sh  # Container startup script
```

## Key Conventions

### File Naming
- **Python**: snake_case for modules, PascalCase for classes
- **TypeScript**: camelCase for variables/functions, PascalCase for components
- **Config files**: kebab-case or standard conventions (.env, docker-compose.yml)

### Directory Organization
- Each major component (server, ui, mcp-server) has its own Makefile
- Shared code between server and mcp-server via symlinks/imports
- Environment configs use .example templates
- Docker-related files centralized in `deploy/`

### Import Patterns
- **Backend**: Relative imports within genericsuite_codegen package
- **Frontend**: Absolute imports from src/ using Vite aliases
- **Shared**: Common utilities in respective lib/ directories

### Configuration Management
- Root `.env` for shared environment variables
- Component-specific configs in respective directories
- Docker environment variables in docker-compose.yml
- MCP client configs as separate JSON files