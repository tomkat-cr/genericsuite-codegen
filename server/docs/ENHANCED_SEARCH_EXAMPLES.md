# Enhanced Search Examples

This document provides comprehensive examples of the enhanced vector search capabilities, including dual search results and document retrieval functionality.

## Table of Contents

1. [Dual Search Examples](#dual-search-examples)
2. [Document Retrieval Examples](#document-retrieval-examples)
3. [Context-Aware Generation Examples](#context-aware-generation-examples)
4. [Error Handling Examples](#error-handling-examples)
5. [Integration Examples](#integration-examples)

## Dual Search Examples

### Example 1: JSON Configuration Generation

**User Request:**
```json
{
  "query": "Create a user management table with name, email, and role fields",
  "task_type": "json_config"
}
```

**Dual Search Process:**

1. **User Query Search:**
   - Query: "Create a user management table with name, email, and role fields"
   - Results: General table creation examples

2. **Contextual Rules Search:**
   - Query: "examples of how to create a JSON table configuration files in Genericsuite"
   - Results: Specific GenericSuite table configuration patterns

**Search Results:**

```json
{
  "dual_search_results": {
    "user_results": [
      {
        "content": "User management typically requires fields for identification, contact, and authorization...",
        "source": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/general-patterns.md",
        "score": 0.85,
        "metadata": {
          "file_type": "md",
          "section": "user-management"
        }
      }
    ],
    "context_results": [
      {
        "content": "{\n  \"table_name\": \"users\",\n  \"fields\": [\n    {\n      \"name\": \"name\",\n      \"type\": \"string\",\n      \"required\": true,\n      \"validation\": \"min_length:2\"\n    },\n    {\n      \"name\": \"email\",\n      \"type\": \"email\",\n      \"required\": true,\n      \"unique\": true\n    },\n    {\n      \"name\": \"role\",\n      \"type\": \"select\",\n      \"options\": [\"admin\", \"user\", \"viewer\"]\n    }\n  ]\n}",
        "source": "local_repo_files/genericsuite-basecamp/mkdocs_root/code/genericsuite-configs/frontend/users.json",
        "score": 0.95,
        "metadata": {
          "file_type": "json",
          "section": "table-configuration"
        }
      }
    ],
    "merged_results": [
      {
        "content": "GenericSuite table configuration with user management fields...",
        "source": "local_repo_files/genericsuite-basecamp/mkdocs_root/code/genericsuite-configs/frontend/users.json",
        "score": 0.95,
        "priority": "contextual",
        "metadata": {
          "file_type": "json",
          "section": "table-configuration",
          "merge_reason": "contextual_priority"
        }
      },
      {
        "content": "User management typically requires fields for identification...",
        "source": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/general-patterns.md",
        "score": 0.85,
        "priority": "user",
        "metadata": {
          "file_type": "md",
          "section": "user-management",
          "merge_reason": "user_requirements"
        }
      }
    ],
    "context_used": {
      "code_type": "json",
      "framework": null,
      "confidence": 0.9,
      "contextual_search": "examples of how to create a JSON table configuration files in Genericsuite"
    }
  }
}
```

### Example 2: LangChain Tool Generation

**User Request:**
```json
{
  "query": "Create a tool for user authentication with JWT tokens",
  "task_type": "python_code",
  "code_type": "langchain"
}
```

**Dual Search Results:**

```json
{
  "dual_search_results": {
    "user_results": [
      {
        "content": "JWT authentication requires token validation, user lookup, and session management...",
        "source": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Backend-Development/authentication.md",
        "score": 0.88
      }
    ],
    "context_results": [
      {
        "content": "from langchain.tools import BaseTool\nfrom typing import Optional, Type\nfrom pydantic import BaseModel, Field\n\nclass UserAuthTool(BaseTool):\n    name = \"user_auth\"\n    description = \"Authenticate users using JWT tokens\"\n    \n    def _run(self, token: str) -> dict:\n        # GenericSuite authentication pattern\n        return self.validate_jwt_token(token)",
        "source": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Sample-Code/langchain-tools/auth_tool.py",
        "score": 0.92
      }
    ],
    "merged_results": [
      {
        "content": "GenericSuite LangChain authentication tool implementation...",
        "source": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Sample-Code/langchain-tools/auth_tool.py",
        "score": 0.92,
        "priority": "contextual"
      }
    ]
  }
}
```

### Example 3: Frontend Code Generation with AI

**User Request:**
```json
{
  "query": "Create a chat interface with AI integration",
  "task_type": "frontend_code",
  "include_ai": true
}
```

**Contextual Search Triggered:**
- Query: "examples of how to create frontend with AI code in Genericsuite"
- File Filter: "jsx"

**Results:**
```json
{
  "dual_search_results": {
    "context_results": [
      {
        "content": "import React, { useState } from 'react';\nimport { useAIChat } from '@genericsuite/ai-hooks';\n\nconst ChatInterface = () => {\n  const { messages, sendMessage, isLoading } = useAIChat();\n  \n  return (\n    <div className=\"chat-container\">\n      {/* GenericSuite AI chat pattern */}\n    </div>\n  );\n};",
        "source": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Sample-Code/frontend-ai/chat-interface.jsx",
        "score": 0.94
      }
    ]
  }
}
```

## Document Retrieval Examples

### Example 1: Single Document Retrieval

**Agent Tool Call:**
```json
{
  "tool": "retrieve_document_from_local_storage",
  "parameters": {
    "document_path": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Configuration-Guide/Generic-CRUD-Editor-Configuration.md"
  }
}
```

**Tool Response:**
```json
{
  "path": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Configuration-Guide/Generic-CRUD-Editor-Configuration.md",
  "content": "# Generic CRUD Editor Configuration\n\nThis document describes how to configure the Generic CRUD Editor for your tables.\n\n## Table Configuration\n\nThe table configuration defines the structure and behavior of your data tables:\n\n```json\n{\n  \"table_name\": \"users\",\n  \"display_name\": \"User Management\",\n  \"fields\": [\n    {\n      \"name\": \"id\",\n      \"type\": \"integer\",\n      \"primary_key\": true,\n      \"auto_increment\": true\n    },\n    {\n      \"name\": \"name\",\n      \"type\": \"string\",\n      \"required\": true,\n      \"validation\": {\n        \"min_length\": 2,\n        \"max_length\": 100\n      }\n    }\n  ]\n}\n```\n\n## Field Types\n\nSupported field types include:\n- `string`: Text fields with validation\n- `integer`: Numeric fields\n- `email`: Email validation\n- `select`: Dropdown options\n- `boolean`: True/false values\n- `date`: Date picker\n- `datetime`: Date and time picker\n\n## Validation Rules\n\nEach field can have validation rules:\n- `required`: Field must have a value\n- `unique`: Value must be unique in the table\n- `min_length`/`max_length`: String length constraints\n- `min_value`/`max_value`: Numeric constraints\n- `pattern`: Regular expression validation\n\n## UI Configuration\n\nCustomize the user interface:\n- `display_name`: Human-readable table name\n- `description`: Table description\n- `icon`: Table icon\n- `color`: Theme color\n- `permissions`: Access control settings",
  "file_type": "md",
  "size": 2048,
  "last_modified": "2024-01-15T10:30:00Z",
  "encoding": "utf-8",
  "metadata": {
    "title": "Generic CRUD Editor Configuration",
    "section": "Configuration Guide",
    "tags": ["configuration", "crud", "tables"]
  }
}
```

### Example 2: Batch Document Retrieval

**Agent Tool Call:**
```json
{
  "tool": "retrieve_multiple_documents_from_local_storage",
  "parameters": {
    "document_paths": [
      "local_repo_files/genericsuite-basecamp/mkdocs_root/code/genericsuite-configs/frontend/users.json",
      "local_repo_files/genericsuite-basecamp/mkdocs_root/code/genericsuite-configs/frontend/products.json",
      "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Configuration-Guide/field-types.md"
    ]
  }
}
```

**Tool Response:**
```json
{
  "documents": [
    {
      "path": "local_repo_files/genericsuite-basecamp/mkdocs_root/code/genericsuite-configs/frontend/users.json",
      "content": "{\n  \"table_name\": \"users\",\n  \"display_name\": \"User Management\",\n  \"fields\": [\n    {\n      \"name\": \"id\",\n      \"type\": \"integer\",\n      \"primary_key\": true\n    },\n    {\n      \"name\": \"name\",\n      \"type\": \"string\",\n      \"required\": true\n    },\n    {\n      \"name\": \"email\",\n      \"type\": \"email\",\n      \"required\": true,\n      \"unique\": true\n    },\n    {\n      \"name\": \"role\",\n      \"type\": \"select\",\n      \"options\": [\"admin\", \"user\", \"viewer\"]\n    }\n  ]\n}",
      "file_type": "json",
      "size": 512,
      "last_modified": "2024-01-10T14:20:00Z"
    },
    {
      "path": "local_repo_files/genericsuite-basecamp/mkdocs_root/code/genericsuite-configs/frontend/products.json",
      "content": "{\n  \"table_name\": \"products\",\n  \"display_name\": \"Product Catalog\",\n  \"fields\": [\n    {\n      \"name\": \"id\",\n      \"type\": \"integer\",\n      \"primary_key\": true\n    },\n    {\n      \"name\": \"name\",\n      \"type\": \"string\",\n      \"required\": true\n    },\n    {\n      \"name\": \"price\",\n      \"type\": \"decimal\",\n      \"required\": true,\n      \"validation\": {\n        \"min_value\": 0\n      }\n    }\n  ]\n}",
      "file_type": "json",
      "size": 384,
      "last_modified": "2024-01-12T09:15:00Z"
    },
    {
      "path": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Configuration-Guide/field-types.md",
      "content": "# Field Types Reference\n\n## String Fields\n- Basic text input\n- Validation: min_length, max_length, pattern\n- Example: `{\"type\": \"string\", \"validation\": {\"min_length\": 2}}`\n\n## Email Fields\n- Automatic email validation\n- Unique constraint support\n- Example: `{\"type\": \"email\", \"unique\": true}`\n\n## Select Fields\n- Dropdown with predefined options\n- Single or multiple selection\n- Example: `{\"type\": \"select\", \"options\": [\"option1\", \"option2\"]}`",
      "file_type": "md",
      "size": 768,
      "last_modified": "2024-01-08T16:45:00Z"
    }
  ],
  "summary": {
    "total_documents": 3,
    "total_size": 1664,
    "successful_retrievals": 3,
    "failed_retrievals": 0
  }
}
```

### Example 3: Document Metadata Query

**Agent Tool Call:**
```json
{
  "tool": "get_document_metadata",
  "parameters": {
    "document_path": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/large-document.md"
  }
}
```

**Tool Response:**
```json
{
  "path": "local_repo_files/genericsuite-basecamp/mkdocs_root/en/large-document.md",
  "file_type": "md",
  "size": 15728640,
  "last_modified": "2024-01-20T11:30:00Z",
  "exists": true,
  "encoding": "utf-8",
  "metadata": {
    "title": "Comprehensive GenericSuite Guide",
    "estimated_read_time": "45 minutes",
    "sections": 12,
    "code_examples": 28
  }
}
```

## Context-Aware Generation Examples

### Example 1: Automatic Context Detection

**Input:**
```json
{
  "query": "I need to create a table for managing customer orders with order ID, customer info, and items",
  "task_type": "general"
}
```

**Context Determination:**
```json
{
  "context_determination": {
    "detected_code_type": "json",
    "confidence": 0.85,
    "reasoning": "Keywords 'table', 'managing', 'orders' suggest database table configuration",
    "contextual_search_triggered": "examples of how to create a JSON table configuration files in Genericsuite"
  }
}
```

**Generated Response:**
```json
{
  "response": "Based on GenericSuite patterns, here's a customer orders table configuration:\n\n```json\n{\n  \"table_name\": \"customer_orders\",\n  \"display_name\": \"Customer Orders\",\n  \"fields\": [\n    {\n      \"name\": \"order_id\",\n      \"type\": \"string\",\n      \"primary_key\": true,\n      \"auto_generate\": true,\n      \"pattern\": \"ORD-{YYYY}{MM}{DD}-{####}\"\n    },\n    {\n      \"name\": \"customer_name\",\n      \"type\": \"string\",\n      \"required\": true,\n      \"validation\": {\n        \"min_length\": 2,\n        \"max_length\": 100\n      }\n    },\n    {\n      \"name\": \"customer_email\",\n      \"type\": \"email\",\n      \"required\": true\n    },\n    {\n      \"name\": \"items\",\n      \"type\": \"json_array\",\n      \"description\": \"Array of order items\",\n      \"schema\": {\n        \"type\": \"object\",\n        \"properties\": {\n          \"product_id\": {\"type\": \"string\"},\n          \"quantity\": {\"type\": \"integer\", \"minimum\": 1},\n          \"price\": {\"type\": \"number\", \"minimum\": 0}\n        }\n      }\n    },\n    {\n      \"name\": \"order_date\",\n      \"type\": \"datetime\",\n      \"default\": \"now\"\n    },\n    {\n      \"name\": \"status\",\n      \"type\": \"select\",\n      \"options\": [\"pending\", \"processing\", \"shipped\", \"delivered\", \"cancelled\"],\n      \"default\": \"pending\"\n    },\n    {\n      \"name\": \"total_amount\",\n      \"type\": \"decimal\",\n      \"calculated\": true,\n      \"formula\": \"sum(items.quantity * items.price)\"\n    }\n  ],\n  \"ui_config\": {\n    \"list_view\": {\n      \"columns\": [\"order_id\", \"customer_name\", \"order_date\", \"status\", \"total_amount\"],\n      \"sortable\": [\"order_date\", \"total_amount\"],\n      \"filterable\": [\"status\", \"customer_name\"]\n    },\n    \"form_view\": {\n      \"sections\": [\n        {\n          \"title\": \"Customer Information\",\n          \"fields\": [\"customer_name\", \"customer_email\"]\n        },\n        {\n          \"title\": \"Order Details\",\n          \"fields\": [\"items\", \"status\"]\n        }\n      ]\n    }\n  }\n}\n```\n\nThis configuration follows GenericSuite conventions for:\n- Auto-generated order IDs with meaningful patterns\n- Proper field validation and constraints\n- JSON array handling for complex data (items)\n- Calculated fields for derived values\n- UI configuration for optimal user experience",
  "context_used": {
    "contextual_search": "examples of how to create a JSON table configuration files in Genericsuite",
    "documents_retrieved": [
      "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Configuration-Guide/Generic-CRUD-Editor-Configuration.md",
      "local_repo_files/genericsuite-basecamp/mkdocs_root/code/genericsuite-configs/frontend/orders.json"
    ],
    "patterns_applied": [
      "auto_generated_ids",
      "json_array_fields",
      "calculated_fields",
      "ui_configuration"
    ]
  }
}
```

### Example 2: MCP Tool Generation

**Input:**
```json
{
  "query": "Create an MCP tool for file operations",
  "task_type": "python_code",
  "code_type": "mcp"
}
```

**Context-Aware Response:**
```json
{
  "response": "Here's an MCP tool for file operations following GenericSuite patterns:\n\n```python\nfrom typing import Any, Dict, List, Optional\nfrom mcp.server.models import Tool\nfrom mcp.server import Server\nfrom mcp.types import TextContent, ImageContent, EmbeddedResource\nimport os\nimport json\nfrom pathlib import Path\n\nclass FileOperationsTool:\n    \"\"\"MCP tool for file operations following GenericSuite patterns.\"\"\"\n    \n    def __init__(self, server: Server):\n        self.server = server\n        self.allowed_paths = self._get_allowed_paths()\n    \n    def _get_allowed_paths(self) -> List[str]:\n        \"\"\"Get allowed file paths from configuration.\"\"\"\n        return [\n            \"local_repo_files\",\n            \"generated_files\",\n            \"temp_files\"\n        ]\n    \n    def _validate_path(self, file_path: str) -> bool:\n        \"\"\"Validate file path for security.\"\"\"\n        path = Path(file_path).resolve()\n        return any(\n            str(path).startswith(str(Path(allowed).resolve()))\n            for allowed in self.allowed_paths\n        )\n    \n    @self.server.tool()\n    async def read_file(\n        self,\n        file_path: str,\n        encoding: str = \"utf-8\"\n    ) -> List[TextContent]:\n        \"\"\"Read file content with security validation.\n        \n        Args:\n            file_path: Path to the file to read\n            encoding: File encoding (default: utf-8)\n            \n        Returns:\n            File content as text\n        \"\"\"\n        if not self._validate_path(file_path):\n            raise ValueError(f\"Access denied to path: {file_path}\")\n        \n        try:\n            with open(file_path, 'r', encoding=encoding) as f:\n                content = f.read()\n            \n            return [\n                TextContent(\n                    type=\"text\",\n                    text=f\"File: {file_path}\\n\\n{content}\"\n                )\n            ]\n        except Exception as e:\n            return [\n                TextContent(\n                    type=\"text\",\n                    text=f\"Error reading file {file_path}: {str(e)}\"\n                )\n            ]\n    \n    @self.server.tool()\n    async def write_file(\n        self,\n        file_path: str,\n        content: str,\n        encoding: str = \"utf-8\",\n        create_dirs: bool = True\n    ) -> List[TextContent]:\n        \"\"\"Write content to file with security validation.\n        \n        Args:\n            file_path: Path to the file to write\n            content: Content to write\n            encoding: File encoding (default: utf-8)\n            create_dirs: Create parent directories if needed\n            \n        Returns:\n            Success confirmation\n        \"\"\"\n        if not self._validate_path(file_path):\n            raise ValueError(f\"Access denied to path: {file_path}\")\n        \n        try:\n            path = Path(file_path)\n            if create_dirs:\n                path.parent.mkdir(parents=True, exist_ok=True)\n            \n            with open(file_path, 'w', encoding=encoding) as f:\n                f.write(content)\n            \n            return [\n                TextContent(\n                    type=\"text\",\n                    text=f\"Successfully wrote {len(content)} characters to {file_path}\"\n                )\n            ]\n        except Exception as e:\n            return [\n                TextContent(\n                    type=\"text\",\n                    text=f\"Error writing file {file_path}: {str(e)}\"\n                )\n            ]\n    \n    @self.server.tool()\n    async def list_files(\n        self,\n        directory_path: str,\n        pattern: str = \"*\",\n        recursive: bool = False\n    ) -> List[TextContent]:\n        \"\"\"List files in directory with pattern matching.\n        \n        Args:\n            directory_path: Directory to list\n            pattern: File pattern to match (default: *)\n            recursive: Search recursively\n            \n        Returns:\n            List of matching files\n        \"\"\"\n        if not self._validate_path(directory_path):\n            raise ValueError(f\"Access denied to path: {directory_path}\")\n        \n        try:\n            path = Path(directory_path)\n            if not path.exists():\n                return [\n                    TextContent(\n                        type=\"text\",\n                        text=f\"Directory not found: {directory_path}\"\n                    )\n                ]\n            \n            if recursive:\n                files = list(path.rglob(pattern))\n            else:\n                files = list(path.glob(pattern))\n            \n            file_list = []\n            for file in files:\n                if file.is_file():\n                    stat = file.stat()\n                    file_list.append({\n                        \"path\": str(file),\n                        \"size\": stat.st_size,\n                        \"modified\": stat.st_mtime\n                    })\n            \n            return [\n                TextContent(\n                    type=\"text\",\n                    text=f\"Files in {directory_path}:\\n\\n\" +\n                         json.dumps(file_list, indent=2)\n                )\n            ]\n        except Exception as e:\n            return [\n                TextContent(\n                    type=\"text\",\n                    text=f\"Error listing files in {directory_path}: {str(e)}\"\n                )\n            ]\n\n# Tool registration following GenericSuite MCP patterns\ndef register_file_operations_tool(server: Server) -> FileOperationsTool:\n    \"\"\"Register file operations tool with MCP server.\"\"\"\n    tool = FileOperationsTool(server)\n    \n    # Register tool descriptions\n    server.tool_descriptions = {\n        \"read_file\": Tool(\n            name=\"read_file\",\n            description=\"Read file content with security validation\",\n            inputSchema={\n                \"type\": \"object\",\n                \"properties\": {\n                    \"file_path\": {\n                        \"type\": \"string\",\n                        \"description\": \"Path to the file to read\"\n                    },\n                    \"encoding\": {\n                        \"type\": \"string\",\n                        \"description\": \"File encoding\",\n                        \"default\": \"utf-8\"\n                    }\n                },\n                \"required\": [\"file_path\"]\n            }\n        ),\n        \"write_file\": Tool(\n            name=\"write_file\",\n            description=\"Write content to file with security validation\",\n            inputSchema={\n                \"type\": \"object\",\n                \"properties\": {\n                    \"file_path\": {\n                        \"type\": \"string\",\n                        \"description\": \"Path to the file to write\"\n                    },\n                    \"content\": {\n                        \"type\": \"string\",\n                        \"description\": \"Content to write to file\"\n                    },\n                    \"encoding\": {\n                        \"type\": \"string\",\n                        \"description\": \"File encoding\",\n                        \"default\": \"utf-8\"\n                    },\n                    \"create_dirs\": {\n                        \"type\": \"boolean\",\n                        \"description\": \"Create parent directories if needed\",\n                        \"default\": true\n                    }\n                },\n                \"required\": [\"file_path\", \"content\"]\n            }\n        ),\n        \"list_files\": Tool(\n            name=\"list_files\",\n            description=\"List files in directory with pattern matching\",\n            inputSchema={\n                \"type\": \"object\",\n                \"properties\": {\n                    \"directory_path\": {\n                        \"type\": \"string\",\n                        \"description\": \"Directory to list\"\n                    },\n                    \"pattern\": {\n                        \"type\": \"string\",\n                        \"description\": \"File pattern to match\",\n                        \"default\": \"*\"\n                    },\n                    \"recursive\": {\n                        \"type\": \"boolean\",\n                        \"description\": \"Search recursively\",\n                        \"default\": false\n                    }\n                },\n                \"required\": [\"directory_path\"]\n            }\n        )\n    }\n    \n    return tool\n```\n\nThis MCP tool follows GenericSuite patterns for:\n- Security validation with allowed paths\n- Comprehensive error handling\n- Proper tool registration and schema definition\n- Consistent return format using TextContent\n- Configuration-driven allowed paths\n- Path validation to prevent directory traversal",
  "context_used": {
    "contextual_search": "examples of how to create a MCP server tool in Genericsuite",
    "documents_retrieved": [
      "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Sample-Code/mcp-tools/file_operations.py",
      "local_repo_files/genericsuite-basecamp/mkdocs_root/en/Backend-Development/mcp-patterns.md"
    ]
  }
}
```

## Error Handling Examples

### Example 1: Document Not Found

**Agent Tool Call:**
```json
{
  "tool": "retrieve_document_from_local_storage",
  "parameters": {
    "document_path": "local_repo_files/non-existent-file.md"
  }
}
```

**Error Response:**
```json
{
  "error": {
    "error_code": "DOCUMENT_RETRIEVAL_ERROR",
    "error_type": "FILE_NOT_FOUND",
    "message": "Document not found in local storage",
    "details": {
      "requested_path": "local_repo_files/non-existent-file.md",
      "resolved_path": "/absolute/path/to/local_repo_files/non-existent-file.md",
      "directory_exists": true,
      "suggested_alternatives": [
        "local_repo_files/similar-file.md",
        "local_repo_files/another-file.md"
      ]
    }
  }
}
```

### Example 2: Path Traversal Attempt

**Agent Tool Call:**
```json
{
  "tool": "retrieve_document_from_local_storage",
  "parameters": {
    "document_path": "local_repo_files/../../../etc/passwd"
  }
}
```

**Error Response:**
```json
{
  "error": {
    "error_code": "DOCUMENT_RETRIEVAL_ERROR",
    "error_type": "PATH_TRAVERSAL",
    "message": "Path traversal detected in document path",
    "details": {
      "requested_path": "local_repo_files/../../../etc/passwd",
      "security_violation": "directory_traversal",
      "allowed_base_paths": ["local_repo_files"]
    }
  }
}
```

### Example 3: Contextual Search Failure with Fallback

**Request:**
```json
{
  "query": "Create a complex data visualization component",
  "task_type": "frontend_code"
}
```

**Response with Fallback:**
```json
{
  "response": "Here's a data visualization component based on available patterns...",
  "search_results": [
    {
      "content": "React component patterns for data visualization...",
      "source": "local_repo_files/general-frontend-patterns.md",
      "score": 0.78
    }
  ],
  "context_used": {
    "contextual_search_attempted": "examples of how to create frontend code in Genericsuite",
    "contextual_search_status": "failed",
    "fallback_applied": true,
    "fallback_reason": "No specific GenericSuite frontend patterns found for data visualization"
  },
  "warnings": [
    {
      "code": "CONTEXTUAL_SEARCH_FAILED",
      "message": "Contextual search failed, using general patterns",
      "impact": "Generated code may not follow all GenericSuite conventions"
    }
  ]
}
```

## Integration Examples

### Example 1: MCP Client Integration

**MCP Tool Call:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "generate_json_config",
    "arguments": {
      "requirements": "Create a product catalog table with name, description, price, and category",
      "table_name": "products",
      "config_type": "table"
    }
  }
}
```

**MCP Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Generated product catalog configuration:\n\n```json\n{\n  \"table_name\": \"products\",\n  \"display_name\": \"Product Catalog\",\n  \"fields\": [\n    {\n      \"name\": \"id\",\n      \"type\": \"integer\",\n      \"primary_key\": true,\n      \"auto_increment\": true\n    },\n    {\n      \"name\": \"name\",\n      \"type\": \"string\",\n      \"required\": true,\n      \"validation\": {\n        \"min_length\": 2,\n        \"max_length\": 200\n      }\n    },\n    {\n      \"name\": \"description\",\n      \"type\": \"text\",\n      \"required\": false\n    },\n    {\n      \"name\": \"price\",\n      \"type\": \"decimal\",\n      \"required\": true,\n      \"validation\": {\n        \"min_value\": 0,\n        \"decimal_places\": 2\n      }\n    },\n    {\n      \"name\": \"category\",\n      \"type\": \"select\",\n      \"required\": true,\n      \"options\": [\n        \"electronics\",\n        \"clothing\",\n        \"books\",\n        \"home\",\n        \"sports\"\n      ]\n    }\n  ]\n}\n```\n\nThis configuration follows GenericSuite patterns for product management with proper validation and field types."
    }
  ],
  "isError": false,
  "_meta": {
    "context_used": {
      "contextual_search": "examples of how to create a JSON table configuration files in Genericsuite",
      "documents_retrieved": 2,
      "patterns_applied": ["table_structure", "field_validation", "select_options"]
    }
  }
}
```

### Example 2: API Integration with Streaming

**HTTP Request:**
```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Create a user authentication system with JWT",
    "task_type": "backend_code"
  }'
```

**Streaming Response:**
```
data: {"type": "context", "content": "Searching for GenericSuite backend authentication patterns..."}

data: {"type": "search_results", "content": "Found 3 relevant authentication examples"}

data: {"type": "document_retrieval", "content": "Retrieved complete JWT implementation guide"}

data: {"type": "generation_start", "content": "Generating authentication system based on GenericSuite patterns..."}

data: {"type": "code", "content": "from fastapi import FastAPI, HTTPException, Depends\nfrom fastapi.security import HTTPBearer, HTTPAuthorizationCredentials\nimport jwt\nfrom datetime import datetime, timedelta\n\nclass JWTAuthenticator:\n    \"\"\"GenericSuite JWT authentication following established patterns.\"\"\"\n    \n    def __init__(self, secret_key: str, algorithm: str = \"HS256\"):\n        self.secret_key = secret_key\n        self.algorithm = algorithm\n    \n    def create_token(self, user_data: dict, expires_delta: timedelta = None) -> str:\n        \"\"\"Create JWT token with user data.\"\"\"\n        to_encode = user_data.copy()\n        if expires_delta:\n            expire = datetime.utcnow() + expires_delta\n        else:\n            expire = datetime.utcnow() + timedelta(hours=24)\n        \n        to_encode.update({\"exp\": expire})\n        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)"}

data: {"type": "completion", "content": "Authentication system generated successfully with GenericSuite patterns"}
```

### Example 3: Error Recovery and Fallback

**Request with Problematic Context:**
```json
{
  "query": "Create a quantum computing interface",
  "task_type": "frontend_code"
}
```

**Response with Graceful Fallback:**
```json
{
  "response": "I'll create a frontend interface based on available patterns. Since quantum computing interfaces are specialized, I'm using general GenericSuite frontend patterns:\n\n```jsx\nimport React, { useState, useEffect } from 'react';\nimport { Card, Button, Input, Alert } from '@genericsuite/ui-components';\n\nconst QuantumInterface = () => {\n  const [quantumState, setQuantumState] = useState(null);\n  const [isProcessing, setIsProcessing] = useState(false);\n  const [error, setError] = useState(null);\n  \n  // Following GenericSuite error handling patterns\n  const handleQuantumOperation = async (operation) => {\n    setIsProcessing(true);\n    setError(null);\n    \n    try {\n      // Quantum operation logic would go here\n      const result = await performQuantumOperation(operation);\n      setQuantumState(result);\n    } catch (err) {\n      setError(`Quantum operation failed: ${err.message}`);\n    } finally {\n      setIsProcessing(false);\n    }\n  };\n  \n  return (\n    <div className=\"quantum-interface\">\n      <Card title=\"Quantum Computing Interface\">\n        {error && (\n          <Alert type=\"error\" message={error} />\n        )}\n        \n        <div className=\"quantum-controls\">\n          <Button \n            onClick={() => handleQuantumOperation('superposition')}\n            disabled={isProcessing}\n          >\n            Create Superposition\n          </Button>\n          \n          <Button \n            onClick={() => handleQuantumOperation('entanglement')}\n            disabled={isProcessing}\n          >\n            Create Entanglement\n          </Button>\n        </div>\n        \n        {quantumState && (\n          <div className=\"quantum-state\">\n            <h3>Quantum State:</h3>\n            <pre>{JSON.stringify(quantumState, null, 2)}</pre>\n          </div>\n        )}\n      </Card>\n    </div>\n  );\n};\n\nexport default QuantumInterface;\n```\n\nNote: This interface uses GenericSuite UI patterns for error handling, loading states, and component structure, but quantum computing specifics would need domain expertise.",
  "context_used": {
    "contextual_search_attempted": "examples of how to create frontend code in Genericsuite",
    "contextual_search_status": "partial_success",
    "fallback_applied": true,
    "patterns_found": ["error_handling", "loading_states", "ui_components"],
    "patterns_missing": ["quantum_computing_specifics"]
  },
  "recommendations": [
    "Consider consulting quantum computing documentation for domain-specific patterns",
    "Review GenericSuite scientific computing examples if available",
    "Implement proper quantum state visualization components"
  ]
}
```

These examples demonstrate the comprehensive capabilities of the enhanced search system, including dual search functionality, document retrieval, context-aware generation, error handling, and integration patterns.