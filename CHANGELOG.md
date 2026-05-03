# CHANGELOG

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/) and [Keep a Changelog](http://keepachangelog.com/).


## [Unreleased] - Date

### Added

### Changed

### Fixed

### Removed

### Security


## [Unreleased] - Date

### Added
- AGENTS.md, GEMINI.md, and CLAUDE.md files to provide context and instructions to AI Coding Assistants [GS-303].
- Add SAST testing [GS-315].
- Implement comprehensive integration test suite for MCP server and core system components [GS-172].

### Changed
- `Kiro-Usage.md`, `Kiro-SDLC-Screenshots.md`, and referenced images moved to `.kiro/docs/`
- Updated author information on `package.json`.


## [1.4.0] - 2025-12-25

### Added
- CRON process to update the knowledge base.
- Settings page to configure the AI providers and other settings.
- Settings stored in the JSON file, if the envvar is not set there, defaults to the .env file variables.
- Huggingface, Groq AI/ML API, Together.ai, OpenRouter, Nvidia, XAI, Ollama and Rhymes providers.
- TOGETHER_STOP environment variable to stop tokens.
- Rename LLM_MODEL_NAME to AI provider specific environment variables: OPENAI_API_KEY, OPENAI_MODEL_NAME, HF_TOKEN, HF_MODEL_NAME, GROQ_API_KEY, GROQ_MODEL_NAME, TOGETHER_API_KEY, TOGETHER_MODEL_NAME, OPENROUTER_API_KEY, OPENROUTER_MODEL_NAME, NVIDIA_API_KEY, NVIDIA_MODEL_NAME, XAI_API_KEY, XAI_MODEL_NAME, RHYMES_API_KEY, RHYMES_MODEL_NAME, and OLLAMA_MODEL_NAME.
- Rename LLM_BASE_URL to AI provider specific environment variables: OPENAI_BASE_URL, HF_BASE_URL, GROQ_BASE_URL, TOGETHER_BASE_URL, OPENROUTER_BASE_URL, NVIDIA_BASE_URL, XAI_BASE_URL, RHYMES_BASE_URL, and OLLAMA_BASE_URL.
- Logfire integration.
- Script to start and stop the Logfire Telemetry container.
- UI: Knowledge Base Search.
- UI: CRUD editor JSON config files validation tool.
- TEMP_BASE_WEB_URL envvar and "make dev-local-basecamp" to support path translation for sources using the local GenericSuite (Basecamp) web documentation.
- DOCUMENT_RETRIEVAL_MAX_FILE_SIZE_MB envvar to limit the file size for retrieval.
- APP_LOGGER_OPTIONS envvar to silent the logger startup debug messages on the batch server.
- Add the "exc_info" parameter to log_error() function to log the exception traceback.
- MAX_PROMPT_LENGTH envvar to limit the prompt length.
- MCP_SERVER_DEBUG envvar to enable debug mode.
- Bearer token support in MCP server and security check on all tools and resources.
- ConversationsService class to add conversation history to other tools different than Agent queries, e.g. JSON config and Python code generation.
- "server/genericsuite_codegen/assets/llm_models_data.json" file to store the AI models data, including the context window size and token pricing.
- File name exclusions to the Ingestion process to ignore "requirements.txt" files.

### Changed
- Rename "/knowledge-base/status" endpoint to "/update-knowledge-base/status".
- Rename "/knowledge-base/progress" endpoint to "/update-knowledge-base/progress".
- Rename MONGODB_URI and MONGODB_DB_NAME envvars to APP_DB_URI and APP_DB_NAME.
- Rename LLM_MODEL envvar to LLM_MODEL_NAME.
- All logging is now configurable centralized in app_logger.py.
- "utilities.py" moved from "server/genericsuite_codegen/ai" to "server/genericsuite_codegen/utilities".
- "model_api" added to AgentConfig model.
- Reduce noise from external (pymongo) and internal libraries (debug messages on uvicorn).
- UI: Timestamps are now displayed in local time.
- UI: task_type, model_used and token_usage are now displayed in the chat interface when debugging is enabled.
- UI: "KnowledgeBasePage.tsx" was splited to separate components.
- Ingestion: filter files from the cloned repo that are not under the "docs" directory (BASE_LOCAL_PATH). Also copy the files "CrudEditorConfigInterface.ts" and "crud_editor_config_classes.py" once the repo is updated.
- Rename DEFAULT_MAX_CONTEXT_LENGTH envvar to CONTEXT_DEFAULT_MAX_LENGTH.
- Rename the CORS_ORIGINS envvar to CORS_ORIGIN.
- Rename MCP server tools removing the "mcp_" prefix to the function names.
- Change ports 8000, 8070, 3000 and 3001 to 8002, 8072, 3002 and 3003 respectively.
- Change enhanced search max_context_length and context_limit from 8000 to 2000000.
- Separate AI tools in individual files.

### Fixed
- Logger debug set correctly when DEBUG envvar is set to 1.
- UI: API calls issue because of the additional "data" property in the response.
- The JSON config generation to include both frontend and backend files following the GenericSuite rules.
- Issue with extract_code_blocks() separating the blocks between backticks ` that are prefixed/suffixed with other texts.

### Removed
- LLM_API_KEY envvar, replaced by AI provider specific variables.
- MONGODB_HOST_NAME, MONGODB_HOST_PORT, MONGODB_USER, MONGODB_PASSWORD envvars from ".env.example" file.


## [1.3.0] - 2025-10-01

### Added
- Enhanced Search Types: Comprehensive type definitions for dual search operations and context-aware generation
  - `CodeGenerationContext` for code generation context information
  - `DualSearchResult` for dual vector search results
  - `DocumentContent` and `DocumentMetadata` for document handling
  - `SearchTemplate` and `EnhancedSearchConfig` for configurable search templates
- Complete API models for enhanced search operations
- Exception hierarchy for enhanced search error handling
- API /v1 to all endpoints.
- Remote repository branch with the REMOTE_REPO_BRANCH environment variable.

### Changed
- MCP server main file "start_mcp_server.py" moved from "mcp-server/" to "server/".
- Server test runs with local MongoDB and "run-server.sh test".

### Fixed
- UI API call error handling verifying the response status.

### Security
- Update dependencies according to Github Dependabot suggestions:
  - Change: Bump Vite to version 5.4.20 and Black to version 24.10.0 in package-lock.json, poetry.lock, and pyproject.toml files.
  - Change: Adjust Python version requirement for Black to >=3.9.
  - Change: Update content hashes in lock files for consistency.


## [1.2.0] - 2025-09-30

### Added
- Add profiles to docker-compose.yml.example to allow/disable using local MongoDB and MongoDB Express, and USE_LOCAL_MONGODB environment variable to control it.
- Add MONGODB_DB_NAME to .env.example, docker-compose.yml.example and "server/genericsuite_codegen/database/setup.py" to support the database name for the MongoDB database.
- Added ALT_BASE_LOCAL_PATH to .env.example for alternative local repository paths to handle both "/var/local_repo_files" and "./local_repo_files".
- API: New endpoint to retrieve local repository information.

### Changed
- Deploy: Updated "run-deploy.sh" to conditionally use local MongoDB based on port availability and USE_LOCAL_MONGODB environment variable.
- Refactored "local_path_to_url" function for improved URL handling and support for ALT_BASE_LOCAL_PATH.
- Updated MCP Server to return detailed context and sources in KB search results. 
- Updated KnowledgeBaseTool to use DEFAULT_MAX_CONTEXT_LENGTH for context generation and raise its value from 4000 to 10000. 
- API: Update RepositoryCloner to return detailed response on cloning status.
- API: Enhance error handling and logging in ingestion process.
- UI: Update KnowledgeBasePage to improve error messaging, debugging output and KB update status follow up.

### Fixed
- Agent Tools: Fix path translation in search_similar_documents method to properly pass replace_extension parameter for markdown-to-HTML URL conversion.
- Database: Fix the issue with the vector search index creation.


## [1.1.0] - 2025-09-21

### Added
- UI: Add path translation for sources in the chat interface, so local paths are displayed as GenericSuite web documentation URLs.
- Add BASE_LOCAL_PATH and BASE_WEB_URL to .env.example and docker-compose.yml.example to support path translation for sources.
- API: New endpoint to retrieve local repository information.

### Changed
- MCP Server: Enhance server startup (python mcp_server.py) with new configuration methods and output formatting. Those method were moved from mac_server_startup.sh to mcp_server.py.
- MCP Server: bash script does "ln -s ../server/genericsuite_codegen ." to link the server to the root of the project, instead of copying it.
- MCP Server: Update Makefile to include requirements export command.
- MCP Server: Update package.json for improved keyword formatting.
- API: Refactored API endpoint methods (/search and /query) to support path translation for sources.
- API: Update RepositoryCloner to return detailed response on cloning status.
- API: Enhance error handling and logging in ingestion process.
- UI: Enhanced knowledge base search with improved result formatting.
- UI: Updated Makefile to include a new command for rebuilding the UI (make rebuild-ui).
- UI: Update KnowledgeBasePage to improve error messaging, debugging output and KB update status follow up.

### Fixed
- MCP Server: MCP Inspector run fixed, now loads the server appropriately.
- MCP Server: Fixed the issue with the Knowledge Base search tool, now it returns the results correctly.
- UI: Fixed the issue where sources were not displayed correctly in the chat interface.


## [1.0.0] - 2025-09-15

### Added
- Project ideation, initial prompt development and code generation with Kiro for the [Code with Kiro Hackathon](https://kiro.devpost.com/) [GS-172].
