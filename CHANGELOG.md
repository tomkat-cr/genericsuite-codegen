# CHANGELOG

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/) and [Keep a Changelog](http://keepachangelog.com/).


## [Unreleased] - Date

### Added

### Changed

### Fixed

### Removed

### Security


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
