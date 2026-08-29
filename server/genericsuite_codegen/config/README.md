# Enhanced Search Configuration

This directory contains configuration files and utilities for the GenericSuite CodeGen enhanced search functionality.

## Configuration Files

### Enhanced Search Configuration

- `enhanced_search_config.json` - Default configuration
- `enhanced_search_config.development.json` - Development environment settings
- `enhanced_search_config.production.json` - Production environment settings  
- `enhanced_search_config.docker.json` - Docker deployment settings

### Search Templates Configuration

- `search_templates.json` - Default search templates
- `search_templates.extended.json` - Extended templates with additional code types

## Configuration Structure

### Enhanced Search Config

```json
{
  "enhanced_search": {
    "enabled": true,
    "max_context_length": 10000,
    "fallback_enabled": true,
    "dual_search_enabled": true,
    "context_determination_enabled": true,
    "document_retrieval_enabled": true
  },
  "local_storage": {
    "local_repo_path": "local_repo_files",
    "max_file_size_mb": 10,
    "allowed_file_extensions": [".md", ".py", ".js", "..."],
    "excluded_directories": [".git", "__pycache__", "..."]
  },
  "search_performance": {
    "max_concurrent_searches": 5,
    "search_timeout_seconds": 30,
    "cache_enabled": true,
    "cache_ttl_seconds": 3600
  },
  "context_determination": {
    "confidence_threshold": 0.6,
    "default_context": "generic",
    "context_keywords": {
      "json": ["json", "configuration", "..."],
      "langchain": ["langchain", "tool", "..."]
    }
  },
  "logging": {
    "level": "INFO",
    "log_search_queries": true,
    "log_document_retrieval": true,
    "log_context_determination": true,
    "log_performance_metrics": true
  }
}
```

### Search Templates Config

```json
{
  "templates": {
    "json": {
      "template": "examples of how to create a JSON table configuration files in Genericsuite",
      "file_type_filter": "json",
      "priority": 1,
      "description": "JSON table configuration files for GenericSuite CRUD operations"
    }
  },
  "template_groups": {
    "ai_tools": ["langchain", "mcp", "backend_ai"],
    "frontend": ["frontend", "frontend_ai", "ui_components"]
  }
}
```

## Environment Variables

The following environment variables can override configuration file settings:

### Enhanced Search Settings
- `ENHANCED_SEARCH_ENABLED` - Enable/disable enhanced search (true/false)
- `ENHANCED_SEARCH_MAX_CONTEXT_LENGTH` - Maximum context length (integer)
- `ENHANCED_SEARCH_FALLBACK_ENABLED` - Enable fallback behavior (true/false)
- `ENHANCED_SEARCH_DUAL_SEARCH_ENABLED` - Enable dual search (true/false)
- `ENHANCED_SEARCH_CONTEXT_DETERMINATION_ENABLED` - Enable context determination (true/false)
- `ENHANCED_SEARCH_DOCUMENT_RETRIEVAL_ENABLED` - Enable document retrieval (true/false)

### Local Storage Settings
- `LOCAL_REPO_DIR` - Local repository directory path
- `DOCUMENT_RETRIEVAL_MAX_FILE_SIZE_MB` - Maximum file size for retrieval (integer)

### Performance Settings
- `SEARCH_MAX_CONCURRENT_SEARCHES` - Maximum concurrent searches (integer)
- `SEARCH_TIMEOUT_SECONDS` - Search timeout in seconds (integer)
- `SEARCH_CACHE_ENABLED` - Enable search caching (true/false)
- `SEARCH_CACHE_TTL_SECONDS` - Cache TTL in seconds (integer)

### Context Determination Settings
- `CONTEXT_DETERMINATION_CONFIDENCE_THRESHOLD` - Confidence threshold (0.0-1.0)
- `CONTEXT_DETERMINATION_DEFAULT_CONTEXT` - Default context type

### Logging Settings
- `ENHANCED_SEARCH_LOG_LEVEL` - Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- `ENHANCED_SEARCH_LOG_QUERIES` - Log search queries (true/false)
- `ENHANCED_SEARCH_LOG_DOCUMENT_RETRIEVAL` - Log document retrieval (true/false)
- `ENHANCED_SEARCH_LOG_CONTEXT_DETERMINATION` - Log context determination (true/false)
- `ENHANCED_SEARCH_LOG_PERFORMANCE_METRICS` - Log performance metrics (true/false)

## Configuration Management

### Using the Config Manager CLI

```bash
# Validate all configuration files
python -m genericsuite_codegen.config.config_manager validate

# Validate specific file
python -m genericsuite_codegen.config.config_manager validate --file enhanced_search_config.json

# Load and display configuration
python -m genericsuite_codegen.config.config_manager load

# Load environment-specific configuration
python -m genericsuite_codegen.config.config_manager load --environment production

# Load search templates
python -m genericsuite_codegen.config.config_manager load --templates

# Create new configuration file
python -m genericsuite_codegen.config.config_manager create my_config.json

# Create new templates file
python -m genericsuite_codegen.config.config_manager create --templates my_templates.json
```

### Using the Config Loader in Code

```python
from genericsuite_codegen.config import ConfigLoader

# Load enhanced search configuration
loader = ConfigLoader()
config = loader.load_enhanced_search_config()

# Load environment-specific configuration
config = loader.load_enhanced_search_config(environment="production")

# Load search templates
templates = loader.load_search_templates()

# Reload configuration (clears cache)
loader.reload_config()
```

### Using the Config Validator

```python
from genericsuite_codegen.config import ConfigValidator, ValidationError

validator = ConfigValidator()

try:
    # Validate configuration file
    validator.validate_config_file("enhanced_search_config.json")
    print("Configuration is valid")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Environment-Specific Configurations

### Development
- Higher context length (15000)
- More verbose logging (DEBUG level)
- Longer timeouts for debugging
- Additional file extensions allowed

### Production
- Optimized for performance
- Lower context length (8000)
- Minimal logging (WARNING level)
- Shorter timeouts
- Stricter file size limits

### Docker
- Container-appropriate paths (`/app/local_repo_files`)
- Balanced performance settings
- Container-friendly logging

## Configuration Priority

Configuration values are loaded in the following priority order (highest to lowest):

1. Environment variables
2. Environment-specific configuration file
3. Default configuration file
4. Hardcoded defaults

## Validation Rules

### Enhanced Search Config
- `max_context_length` must be >= 1000
- `confidence_threshold` must be between 0.0 and 1.0
- File extensions must start with '.'
- Timeout values must be positive
- Log level must be valid (DEBUG/INFO/WARNING/ERROR/CRITICAL)

### Search Templates Config
- Each template must have a 'template' field
- Priority must be an integer
- File type filters must be strings or null
- Template groups must reference existing templates

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   - Check file path and permissions
   - Use absolute paths if needed
   - Verify file exists in config directory

2. **Invalid JSON format**
   - Validate JSON syntax
   - Check for trailing commas
   - Ensure proper escaping of strings

3. **Validation errors**
   - Review error messages carefully
   - Check data types match requirements
   - Verify required fields are present

4. **Environment variable not working**
   - Check variable name spelling
   - Ensure proper data type conversion
   - Verify environment is loaded correctly

### Debug Mode

Enable debug logging to troubleshoot configuration issues:

```bash
export ENHANCED_SEARCH_LOG_LEVEL=DEBUG
```

This will provide detailed information about:
- Configuration file loading
- Environment variable processing
- Validation steps
- Default value usage