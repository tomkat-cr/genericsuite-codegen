# CHANGELOG

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/) and [Keep a Changelog](http://keepachangelog.com/).


## [Unreleased] - Date

### Added

### Changed

### Fixed

### Removed

### Security


## [1.1.0] - 2025-09-21

### Added
- UI: Add path translation for sources in the chat interface, so local paths are displayed as GenericSuite web documentation URLs.
- Add BASE_LOCAL_PATH and BASE_WEB_URL to .env.example and docker-compose.yml.example to support path translation for sources.

### Changed
- MCP Server: Enhance server startup (python mcp_server.py) with new configuration methods and output formatting. Those method were moved from mac_server_startup.sh to mcp_server.py.
- MCP Server: bash script does "ln -s ../server/genericsuite_codegen ." to link the server to the root of the project, instead of copying it.
- MCP Server: Update Makefile to include requirements export command.
- MCP Server: Update package.json for improved keyword formatting.
- API: Refactored API endpoint methods to support path translation for sources.
- UI: Enhanced knowledge base search with improved result formatting.
- UI: Updated Makefile to include a new command for rebuilding the UI (make rebuild-ui).

### Fixed
- MCP Server: MCP Inspector run fixed, now loads the server appropriately.
- MCP Server: Fixed the issue with the Knowledge Base search tool, now it returns the results correctly.
- UI: Fixed the issue where sources were not displayed correctly in the chat interface.


## [1.0.0] - 2025-09-15

### Added
- Project ideation, initial prompt development and code generation with Kiro for the [Code with Kiro Hackathon](https://kiro.devpost.com/) [GS-172].
