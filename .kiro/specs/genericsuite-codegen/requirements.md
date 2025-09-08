# Requirements Document

## Introduction

The Generic Suite CodeGen is a comprehensive Retrieval-Augmented Generation (RAG) AI system that allows developers to query the GenericSuite documentation and knowledgebase, and generates starting application frontend and backend code, JSON configuration files, Python Langchain Tools, and Python MCP Tools for GenericSuite library-based projects. The system combines document ingestion, vector search, AI agent capabilities, and a user-friendly interface to help developers create configuration files and tools based on GenericSuite documentation and examples.

## Requirements

### Requirement 1: Document Ingestion and Processing

**User Story:** As a developer, I want the system to automatically ingest and process GenericSuite documentation so that the AI agent has access to up-to-date knowledge for code generation.

#### Acceptance Criteria

1. WHEN the system starts THEN the system SHALL clone the GenericSuite basecamp repository from `https://github.com/tomkat-cr/genericsuite-basecamp.git`
2. WHEN processing repository files THEN the system SHALL read all files while respecting .gitignore exclusions
3. WHEN processing documents THEN the system SHALL support multiple file formats including .md, .py, .js, .jsx, .ts, .tsx, .json, .toml, .env.example (among many others specified in the `BuildPrompt.md` file), and PDF files
4. WHEN chunking documents THEN the system SHALL create appropriately sized chunks for embedding generation
5. WHEN generating embeddings THEN the system SHALL support both OpenAI and HuggingFace embedding models based on configuration

### Requirement 2: Vector Database and Search

**User Story:** As a developer, I want the system to store and efficiently search through documentation embeddings so that relevant context can be retrieved for my queries.

#### Acceptance Criteria

1. WHEN storing embeddings THEN the system SHALL use MongoDB Vector Search as the primary database
2. WHEN performing searches THEN the system SHALL support semantic search for efficient retrieval
3. WHEN updating the knowledge base THEN the system SHALL delete existing vectors before storing new ones to prevent duplicates
4. WHEN querying the database THEN the system SHALL return relevant document chunks with similarity scores

### Requirement 3: AI Agent with Code Generation

**User Story:** As a developer, I want an AI agent that can generate JSON configuration files and Python code based on GenericSuite patterns so that I can quickly scaffold new projects.

#### Acceptance Criteria

1. WHEN querying the agent THEN the agent SHALL use Pydantic AI framework for response generation
2. WHEN generating responses THEN the system SHALL integrate retrieved context from the knowledge base
3. WHEN creating JSON files THEN the agent SHALL follow GenericSuite table configuration patterns
4. WHEN generating Python tools THEN the agent SHALL create both Langchain Tools and MCP Tools
5. WHEN generating Python backend code THEN the agent SHALL create backend code following ExampleApp structure and allow user selection between FastAPI (default), Flask, or Chalice frameworks
6. WHEN generating ReactJS frontend code THEN the agent SHALL create frontend code following ExampleApp structure and patterns
7. WHEN using LLM providers THEN the system SHALL support OpenAI API and LiteLLM with configurable models

### Requirement 4: User Interface for Interaction

**User Story:** As a developer, I want a clean web interface where I can update the knowledge base, upload documents, and interact with the AI agent so that I can easily generate the code I need.

#### Acceptance Criteria

1. WHEN accessing the interface THEN the interface SHALL be built with ReactJS and ShadCn components
2. WHEN updating knowledge base THEN the interface SHALL provide controls to refresh repository data and regenerate embeddings
3. WHEN uploading documents THEN the system SHALL allow users to add additional context files for queries
4. WHEN querying the agent THEN the interface SHALL display responses with source attribution
5. WHEN receiving generated files THEN the system SHALL allow users to download created JSON, Python, and ReactJS files
6. WHEN having conversations THEN the system SHALL allow users to ask follow-up questions in the same session

### Requirement 5: API and MCP Server Integration

**User Story:** As a developer, I want robust API endpoints and MCP server integration so that the system can be used programmatically and integrated with other tools.

#### Acceptance Criteria

1. WHEN accessing the API THEN the API SHALL be built with FastAPI framework
2. WHEN using MCP integration THEN the system SHALL provide a FastMCP server with tools and resources
3. WHEN making API calls THEN the system SHALL support proper authentication and error handling for all endpoints
4. WHEN integrating with external tools THEN the MCP server SHALL expose agent capabilities as standardized MCP tools

### Requirement 6: Deployment and Configuration

**User Story:** As a developer, I want easy deployment options with Docker so that I can run the system in different environments with minimal setup.

#### Acceptance Criteria

1. WHEN deploying THEN the system SHALL provide Docker containers for all components
2. WHEN configuring THEN the system SHALL use environment variables for all settings
3. WHEN running locally THEN the system SHALL use docker-compose to orchestrate all services including MongoDB
4. WHEN scaling THEN the system SHALL support configurable LLM providers, embedding models, and database connections
5. WHEN developing THEN the system SHALL provide separate development and production configurations

### Requirement 7: Code Generation Quality and Patterns

**User Story:** As a developer, I want the generated code to follow GenericSuite best practices and patterns so that it integrates seamlessly with existing projects.

#### Acceptance Criteria

1. WHEN generating JSON configurations THEN the system SHALL follow GenericSuite table definition patterns from the documentation
2. WHEN creating Langchain Tools THEN the system SHALL match ExampleApp AI Agent implementation patterns
3. WHEN generating MCP Tools THEN the system SHALL be compatible with FastMCP framework standards
4. WHEN creating frontend code THEN the system SHALL follow GenericSuite ExampleApp ReactJS patterns and structure
5. WHEN generating backend code THEN the system SHALL follow GenericSuite ExampleApp API patterns for the selected framework (FastAPI, Flask, or Chalice)

### Requirement 8: Performance and Reliability

**User Story:** As a developer, I want the system to perform efficiently and handle errors gracefully so that I can rely on it for production use.

#### Acceptance Criteria

1. WHEN processing large repositories THEN the system SHALL handle memory efficiently during ingestion
2. WHEN performing vector searches THEN response times SHALL be optimized for user experience
3. WHEN encountering errors THEN the system SHALL provide meaningful error messages and recovery options
4. WHEN handling concurrent requests THEN the API SHALL maintain performance and stability
5. WHEN updating embeddings THEN the process SHALL not interfere with ongoing queries