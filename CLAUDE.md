# CLAUDE.md

This file provides guidance to AI Coding Assistants (Claude Code, Gemini CLI, Cursor, Antigravity, etc.) when working with code in this repository.

## What This Project Does

GenericSuite CodeGen is an AI-powered RAG system that generates JSON configuration files, Python tools, and application code following GenericSuite patterns. It has three main runtime components:

- **FastAPI backend** (`server/`) — REST API on port 8002, Pydantic AI agent, MongoDB vector search
- **React/TypeScript frontend** (`ui/`) — Vite dev server on port 3002
- **MCP server** (`mcp-server/`) — Model Context Protocol integration on port 8072 for Kiro, Claude Desktop, VS Code

## Setup

```bash
make init-app-environment   # copies .env.example → .env and other config files
make install                # installs all workspaces (Poetry + npm)
```

Requires: Node.js 18+, Python 3.12+, Docker, and an `OPENAI_API_KEY` in `.env`.

## Development Commands

```bash
make dev                    # start all services concurrently (server, ui, mcp-server)
make run-db-only            # start MongoDB only
cd server && make run       # backend only
cd ui && make run           # frontend only
cd mcp-server && make run   # MCP server only
```

## Testing

```bash
# Backend tests (requires local MongoDB — make test spins it up via Docker)
cd server && make test

# Run a single test file
cd server && poetry run pytest tests/test_agent_core.py -v

# Run by marker
cd server && poetry run pytest -m unit -v
cd server && poetry run pytest -m integration -v

# MCP server tests
cd mcp-server && make test

# Dependency-sync tests
cd deploy/dependency-sync && poetry run pytest -v

# Run SAST testing
make sast-test
```

Markers defined in `server/pytest.ini`: `unit`, `integration`, `slow`, `error_handling`, `fixtures`.

Test files with `.disabled` extension are intentionally excluded from test runs.

## Linting & Formatting

```bash
# Backend
cd server && make format    # black + isort
cd server && make lint      # flake8 + mypy

# Frontend
cd ui && npm run lint       # eslint
cd ui && npm run format     # prettier
cd ui && npm run type-check # tsc --noEmit

# MCP server
cd mcp-server && make format
cd mcp-server && make lint
```

Python line length: 88 chars (Black default). Config in `server/pyproject.toml`.

## Production

```bash
make run          # build + docker compose up
make status       # docker compose ps
make logs-f       # follow server+client logs
make down         # stop services
make clean-docker # down -v + system prune
```

## Architecture

### Backend (`server/genericsuite_codegen/`)

- `agent/` — Pydantic AI agent core. `agent.py`/`agent_super.py` drive queries; individual tools are in `tool_json_config_generator.py`, `tool_python_code_generator.py`, `tool_knowledge_base.py`, etc. Enhanced dual-search engine in `enhanced_search.py`.
- `api/` — FastAPI routes. All endpoints are prefixed `/v1`. `main.py` mounts routes; `endpoint_methods.py` has the handler logic; `batch.py` handles long-running generation jobs.
- `database/` — MongoDB setup and vector index creation.
- `document_processing/` — Document ingestion, chunking, and embedding pipeline.
- `conversations/` — `ConversationsService` adds history to non-agent tools (JSON config, code gen).
- `config/` — Enhanced search config JSONs (`.development.json`, `.docker.json`, `.production.json`) and `search_templates.json` per code type.
- `utilities/` — Centralized `app_logger.py`; all logging goes through here.
- `assets/llm_models_data.json` — LLM model catalog with context window sizes and token pricing.

### Key API Endpoints

| Route | Purpose |
|---|---|
| `POST /v1/query` / `POST /v1/query/stream` | AI agent queries |
| `POST /v1/generate/json-config` | Generate GenericSuite JSON config |
| `POST /v1/generate/python-code` | Generate Python tool code |
| `POST /v1/generate/frontend-code` | Generate frontend code |
| `POST /v1/generate/backend-code` | Generate backend code |
| `POST /v1/update-knowledge-base` | Trigger knowledge base ingestion |
| `POST /v1/upload-document` | Upload document to knowledge base |
| `POST /v1/search` | Semantic search |
| `GET/POST /v1/settings` | App settings |

### Frontend (`ui/src/`)

Pages: `ChatPage`, `CodeGenerationPage`, `HomePage`, `KnowledgeBasePage`, `ConfigValidatorPage`, `SettingsPage`. Components use Shadcn/ui. API calls go through `src/lib/` utilities.

### MCP Server

Wraps the server package. Tools expose knowledge base search, code generation, and document retrieval to MCP-compatible clients. Configured via `.kiro/settings/mcp.json`. Requires `MCP_API_KEY` for bearer token auth.

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required — primary LLM provider |
| `LLM_PROVIDER` | `openai`, `huggingface`, `groq`, `ollama`, etc. |
| `OPENAI_MODEL_NAME` | Default: `gpt-4o-mini` |
| `APP_DB_URI` | MongoDB connection URI |
| `APP_DB_NAME` | MongoDB database name |
| `USE_LOCAL_MONGODB` | `1` to use local Docker MongoDB |
| `SERVER_PORT` | Default: `8002` |
| `CORS_ORIGIN` | Default: `http://localhost:3002` |
| `MCP_API_KEY` | MCP server auth key |
| `MCP_SERVER_PORT` | Default: `8072` |
| `REMOTE_REPO_URL` | Git repo URL for knowledge base source |
| `LOCAL_REPO_DIR` | Absolute path to cloned knowledge base repo |
| `EMBEDDINGS_PROVIDER` | `openai` or `huggingface` |
| `ENHANCED_SEARCH_ENABLED` | Enable dual-search engine |
| `LOGFIRE_ENABLED` | Enable Pydantic Logfire telemetry |

Full list in `.env.example`.

## Important Notes

- The files `AGENTS.md`, `GEMINI.md`, etc. (if present) have only a referece to `@CLAUDE.md` — edit only `CLAUDE.md`.
- Skills, commands, rules, and sub-agents are located in the `.claude/` directory.
- **Authentication**: `DEFAULT_USER_ID` is used throughout as a placeholder — real auth is not yet implemented. TODOs exist in `endpoint_methods.py`.
- **Knowledge base ingestion** filters files to the `BASE_LOCAL_PATH` subdirectory of the cloned repo and ignores `requirements.txt` files.
- **Dependency sync**: `deploy/dependency-sync/` contains a tool to keep `pyproject.toml` dependencies in sync with Dockerfiles. Run via `make sync-deps` from `deploy/`.
- **Shared code**: The `mcp-server/` package imports from `server/genericsuite_codegen/` — changes to the server package affect the MCP server.
