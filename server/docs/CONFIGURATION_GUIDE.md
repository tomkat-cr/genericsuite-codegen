# Enhanced Search Configuration Guide

This guide covers all configuration options for the enhanced vector search capabilities, including search templates, document retrieval settings, and customization options.

## Table of Contents

1. [Overview](#overview)
2. [Search Template Configuration](#search-template-configuration)
3. [Document Retrieval Configuration](#document-retrieval-configuration)
4. [Environment Variables](#environment-variables)
5. [Advanced Configuration](#advanced-configuration)
6. [Custom Templates](#custom-templates)
7. [Performance Tuning](#performance-tuning)
8. [Security Configuration](#security-configuration)

## Overview

The enhanced search system uses configurable templates and settings to provide context-aware code generation. Configuration is managed through:

- **JSON Configuration Files**: For search templates and structured settings
- **Environment Variables**: For runtime configuration and secrets
- **Code Configuration**: For advanced customization and integration

## Search Template Configuration

### Default Template File

The primary configuration file is located at:
```
server/genericsuite_codegen/config/search_templates.json
```

### Basic Template Structure

```json
{
  "templates": {
    "template_name": {
      "template": "search query template",
      "file_type_filter": "file_extension",
      "priority": 1,
      "enabled": true,
      "description": "Template description"
    }
  },
  "config": {
    "version": "1.0",
    "last_updated": "2024-01-01T00:00:00Z",
    "fallback_enabled": true
  }
}
```

### Default Templates

```json
{
  "templates": {
    "json": {
      "template": "examples of how to create a JSON table configuration files in Genericsuite",
      "file_type_filter": "json",
      "priority": 1,
      "enabled": true,
      "description": "GenericSuite JSON table configuration patterns"
    },
    "langchain": {
      "template": "examples of how to create a Python Langchain Tool in Genericsuite",
      "file_type_filter": "py",
      "priority": 1,
      "enabled": true,
      "description": "GenericSuite LangChain tool implementation patterns"
    },
    "mcp": {
      "template": "examples of how to create a MCP server tool in Genericsuite",
      "file_type_filter": "py",
      "priority": 1,
      "enabled": true,
      "description": "GenericSuite MCP server tool patterns"
    },
    "frontend": {
      "template": "examples of how to create frontend code in Genericsuite",
      "file_type_filter": "jsx",
      "priority": 1,
      "enabled": true,
      "description": "GenericSuite React frontend patterns"
    },
    "frontend_ai": {
      "template": "examples of how to create frontend with AI code in Genericsuite",
      "file_type_filter": "jsx",
      "priority": 1,
      "enabled": true,
      "description": "GenericSuite AI-enhanced frontend patterns"
    },
    "backend": {
      "template": "examples of how to create backend code in Genericsuite",
      "file_type_filter": "py",
      "priority": 1,
      "enabled": true,
      "description": "GenericSuite FastAPI backend patterns"
    },
    "backend_ai": {
      "template": "examples of how to create backend with AI code in Genericsuite",
      "file_type_filter": "py",
      "priority": 1,
      "enabled": true,
      "description": "GenericSuite AI-enhanced backend patterns"
    }
  },
  "config": {
    "version": "1.0",
    "fallback_enabled": true,
    "reload_on_change": true
  }
}
```

### Template Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `template` | string | Yes | The search query template to use for contextual search |
| `file_type_filter` | string | No | File extension filter (e.g., "py", "json", "jsx") |
| `priority` | integer | No | Template priority (higher numbers = higher priority) |
| `enabled` | boolean | No | Whether the template is enabled (default: true) |
| `description` | string | No | Human-readable description of the template |

### Adding Custom Templates

To add a new template for a custom code type:

```json
{
  "templates": {
    "custom_api": {
      "template": "examples of how to create custom API endpoints in Genericsuite",
      "file_type_filter": "py",
      "priority": 2,
      "enabled": true,
      "description": "Custom API endpoint patterns"
    }
  }
}
```

### Template Variables

Templates support variable substitution:

```json
{
  "templates": {
    "framework_specific": {
      "template": "examples of how to create {framework} applications in Genericsuite",
      "file_type_filter": "py",
      "priority": 1,
      "variables": {
        "framework": ["fastapi", "flask", "django"]
      }
    }
  }
}
```

## Document Retrieval Configuration

### Basic Configuration

Document retrieval settings are configured through environment variables and code configuration:

```python
# In your configuration file
DOCUMENT_RETRIEVAL_CONFIG = {
    "local_repo_path": "local_repo_files",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".md", ".py", ".json", ".jsx", ".tsx", ".ts"],
    "encoding_detection": True,
    "cache_enabled": True,
    "cache_ttl": 3600,  # 1 hour
    "security_validation": True
}
```

### Path Configuration

Configure allowed paths for document retrieval:

```python
ALLOWED_PATHS = [
    "local_repo_files",
    "generated_files",
    "temp_files",
    "user_uploads"
]

# Path validation rules
PATH_VALIDATION = {
    "prevent_traversal": True,
    "allowed_base_paths": ALLOWED_PATHS,
    "case_sensitive": True,
    "resolve_symlinks": False
}
```

### File Type Handling

Configure how different file types are handled:

```python
FILE_TYPE_CONFIG = {
    "text_files": {
        "extensions": [".md", ".txt", ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".yaml", ".yml"],
        "encoding_detection": True,
        "max_size": 10 * 1024 * 1024  # 10MB
    },
    "binary_files": {
        "extensions": [".pdf", ".docx", ".xlsx"],
        "allowed": False,
        "error_message": "Binary files are not supported for content retrieval"
    },
    "large_files": {
        "max_size": 50 * 1024 * 1024,  # 50MB
        "streaming": True,
        "chunk_size": 1024 * 1024  # 1MB chunks
    }
}
```

## Environment Variables

### Core Configuration

```bash
# Enhanced Search Configuration
ENHANCED_SEARCH_ENABLED=true
ENHANCED_SEARCH_MAX_CONTEXT_LENGTH=10000
ENHANCED_SEARCH_FALLBACK_ENABLED=true
ENHANCED_SEARCH_DEBUG=false

# Document Retrieval Configuration
LOCAL_REPO_PATH=local_repo_files
DOCUMENT_RETRIEVAL_MAX_SIZE=10485760
DOCUMENT_RETRIEVAL_TIMEOUT=30
DOCUMENT_RETRIEVAL_CACHE_ENABLED=true
DOCUMENT_RETRIEVAL_CACHE_TTL=3600

# Search Template Configuration
SEARCH_TEMPLATES_CONFIG_PATH=server/genericsuite_codegen/config/search_templates.json
SEARCH_TEMPLATES_RELOAD_ENABLED=true
SEARCH_TEMPLATES_FALLBACK_ENABLED=true

# Performance Configuration
DUAL_SEARCH_PARALLEL_ENABLED=true
SEARCH_RESULT_CACHE_ENABLED=true
SEARCH_RESULT_CACHE_TTL=1800
```

### Database Configuration

```bash
# MongoDB Configuration for Vector Search
APP_DB_URI=mongodb://localhost:27017/genericsuite_codegen
MONGODB_VECTOR_COLLECTION=documents
MONGODB_VECTOR_INDEX=vector_index
MONGODB_MAX_POOL_SIZE=10
MONGODB_TIMEOUT=30000
```

### AI Model Configuration

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.1

# Embedding Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=32
```

## Advanced Configuration

### Custom Context Determination

Configure custom context determination rules:

```python
# In server/genericsuite_codegen/config/context_rules.py
CONTEXT_DETERMINATION_RULES = {
    "keywords": {
        "json": ["table", "configuration", "config", "schema", "fields"],
        "langchain": ["tool", "langchain", "agent", "chain"],
        "mcp": ["mcp", "server", "protocol", "tool"],
        "frontend": ["react", "component", "ui", "interface", "jsx"],
        "backend": ["api", "endpoint", "server", "fastapi", "flask"]
    },
    "patterns": {
        "json": r"\b(table|config|schema)\b.*\b(json|configuration)\b",
        "langchain": r"\b(langchain|tool|agent)\b",
        "mcp": r"\b(mcp|server|protocol)\b",
        "frontend": r"\b(react|component|ui|frontend)\b",
        "backend": r"\b(api|endpoint|backend|server)\b"
    },
    "confidence_thresholds": {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    }
}
```

### Search Result Merging

Configure how search results are merged:

```python
SEARCH_MERGE_CONFIG = {
    "strategy": "priority_weighted",  # Options: priority_weighted, score_based, hybrid
    "contextual_priority": 0.7,      # Weight for contextual results
    "user_priority": 0.3,            # Weight for user query results
    "max_results": 10,               # Maximum merged results
    "score_threshold": 0.5,          # Minimum score threshold
    "deduplication": {
        "enabled": True,
        "similarity_threshold": 0.9,
        "prefer": "contextual"       # Options: contextual, user, higher_score
    }
}
```

### Error Handling Configuration

```python
ERROR_HANDLING_CONFIG = {
    "retry_attempts": 3,
    "retry_delay": 1.0,              # Seconds
    "fallback_enabled": True,
    "partial_results_enabled": True,
    "error_logging": {
        "level": "ERROR",
        "include_stack_trace": False,
        "include_request_data": True
    },
    "user_error_messages": {
        "generic": "An error occurred during search. Please try again.",
        "timeout": "Search timed out. Please try a more specific query.",
        "no_results": "No relevant results found. Try different keywords."
    }
}
```

## Custom Templates

### Creating Template Packages

Create reusable template packages for different domains:

```json
{
  "package_name": "ecommerce_templates",
  "version": "1.0.0",
  "description": "Templates for ecommerce applications",
  "templates": {
    "product_catalog": {
      "template": "examples of how to create product catalog configurations in Genericsuite",
      "file_type_filter": "json",
      "priority": 2,
      "tags": ["ecommerce", "products", "catalog"]
    },
    "shopping_cart": {
      "template": "examples of how to create shopping cart components in Genericsuite",
      "file_type_filter": "jsx",
      "priority": 2,
      "tags": ["ecommerce", "cart", "frontend"]
    },
    "payment_processing": {
      "template": "examples of how to create payment processing APIs in Genericsuite",
      "file_type_filter": "py",
      "priority": 2,
      "tags": ["ecommerce", "payments", "backend"]
    }
  }
}
```

### Template Inheritance

Support template inheritance for complex scenarios:

```json
{
  "templates": {
    "base_api": {
      "template": "examples of how to create API endpoints in Genericsuite",
      "file_type_filter": "py",
      "priority": 1,
      "abstract": true
    },
    "crud_api": {
      "extends": "base_api",
      "template": "examples of how to create CRUD API endpoints in Genericsuite",
      "priority": 2,
      "additional_keywords": ["crud", "database", "operations"]
    },
    "auth_api": {
      "extends": "base_api",
      "template": "examples of how to create authentication API endpoints in Genericsuite",
      "priority": 2,
      "additional_keywords": ["auth", "login", "jwt", "security"]
    }
  }
}
```

### Conditional Templates

Create templates that activate based on conditions:

```json
{
  "templates": {
    "ai_enhanced_frontend": {
      "template": "examples of how to create AI-enhanced frontend components in Genericsuite",
      "file_type_filter": "jsx",
      "priority": 3,
      "conditions": {
        "query_contains": ["ai", "artificial intelligence", "machine learning", "llm"],
        "context_type": "frontend",
        "user_preferences": ["ai_features_enabled"]
      }
    }
  }
}
```

## Performance Tuning

### Caching Configuration

```python
CACHE_CONFIG = {
    "search_results": {
        "enabled": True,
        "ttl": 1800,                 # 30 minutes
        "max_size": 1000,            # Maximum cached items
        "key_strategy": "query_hash"
    },
    "document_content": {
        "enabled": True,
        "ttl": 3600,                 # 1 hour
        "max_size": 500,
        "compression": True
    },
    "templates": {
        "enabled": True,
        "ttl": 7200,                 # 2 hours
        "reload_on_change": True
    }
}
```

### Parallel Processing

```python
PARALLEL_CONFIG = {
    "dual_search": {
        "enabled": True,
        "max_workers": 4,
        "timeout": 30
    },
    "document_retrieval": {
        "batch_size": 10,
        "max_workers": 2,
        "timeout": 15
    },
    "embedding_generation": {
        "batch_size": 32,
        "max_workers": 2
    }
}
```

### Resource Limits

```python
RESOURCE_LIMITS = {
    "max_concurrent_searches": 10,
    "max_documents_per_request": 50,
    "max_context_length": 10000,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "request_timeout": 60,               # seconds
    "memory_limit": 512 * 1024 * 1024    # 512MB
}
```

## Security Configuration

### Path Security

```python
SECURITY_CONFIG = {
    "path_validation": {
        "prevent_traversal": True,
        "allowed_base_paths": ["local_repo_files", "generated_files"],
        "blocked_patterns": [r"\.\.\/", r"\/etc\/", r"\/proc\/", r"\/sys\/"],
        "case_sensitive": True
    },
    "file_access": {
        "allowed_extensions": [".md", ".py", ".json", ".jsx", ".tsx", ".ts", ".yaml", ".yml"],
        "blocked_extensions": [".exe", ".bat", ".sh", ".ps1"],
        "max_file_size": 10 * 1024 * 1024,
        "scan_for_malware": False
    },
    "content_filtering": {
        "remove_sensitive_data": True,
        "blocked_patterns": [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]"
        ]
    }
}
```

### API Security

```python
API_SECURITY_CONFIG = {
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 60,
        "burst_limit": 10
    },
    "authentication": {
        "required": False,
        "api_key_header": "X-API-Key",
        "jwt_validation": False
    },
    "request_validation": {
        "max_query_length": 1000,
        "max_file_paths": 20,
        "sanitize_input": True
    }
}
```

## Configuration Examples

### Development Environment

```bash
# .env.development
ENHANCED_SEARCH_ENABLED=true
ENHANCED_SEARCH_DEBUG=true
ENHANCED_SEARCH_FALLBACK_ENABLED=true
LOCAL_REPO_PATH=local_repo_files
DOCUMENT_RETRIEVAL_CACHE_ENABLED=false
SEARCH_TEMPLATES_RELOAD_ENABLED=true
DUAL_SEARCH_PARALLEL_ENABLED=false
```

### Production Environment

```bash
# .env.production
ENHANCED_SEARCH_ENABLED=true
ENHANCED_SEARCH_DEBUG=false
ENHANCED_SEARCH_FALLBACK_ENABLED=true
LOCAL_REPO_PATH=/app/data/repo_files
DOCUMENT_RETRIEVAL_CACHE_ENABLED=true
DOCUMENT_RETRIEVAL_CACHE_TTL=7200
SEARCH_TEMPLATES_RELOAD_ENABLED=false
DUAL_SEARCH_PARALLEL_ENABLED=true
```

### Testing Environment

```bash
# .env.test
ENHANCED_SEARCH_ENABLED=true
ENHANCED_SEARCH_DEBUG=true
LOCAL_REPO_PATH=test_data/repo_files
DOCUMENT_RETRIEVAL_CACHE_ENABLED=false
SEARCH_TEMPLATES_CONFIG_PATH=test_data/test_templates.json
APP_DB_URI=mongodb://localhost:27017/test_genericsuite_codegen
```

## Configuration Validation

The system includes built-in configuration validation:

```python
from genericsuite_codegen.config.config_validator import validate_config

# Validate configuration on startup
config_errors = validate_config()
if config_errors:
    logger.error(f"Configuration validation failed: {config_errors}")
    raise ConfigurationError("Invalid configuration")
```

### Validation Rules

- Template files must be valid JSON
- File paths must exist and be accessible
- Environment variables must have valid values
- Resource limits must be positive integers
- Cache TTL values must be reasonable (> 0, < 86400)

## Troubleshooting Configuration

### Common Issues

1. **Template Loading Failures**
   - Check file permissions on template files
   - Validate JSON syntax
   - Ensure file paths are correct

2. **Document Retrieval Errors**
   - Verify `LOCAL_REPO_PATH` exists
   - Check file permissions
   - Ensure path validation settings are correct

3. **Performance Issues**
   - Adjust cache settings
   - Tune parallel processing limits
   - Check resource limits

4. **Search Quality Issues**
   - Review template queries
   - Adjust context determination rules
   - Fine-tune search merge configuration

### Debug Mode

Enable debug mode for detailed logging:

```bash
ENHANCED_SEARCH_DEBUG=true
LOG_LEVEL=DEBUG
```

This will provide detailed information about:
- Template loading and validation
- Search query execution
- Document retrieval operations
- Context determination decisions
- Error conditions and fallbacks