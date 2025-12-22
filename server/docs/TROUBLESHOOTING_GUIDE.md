# Enhanced Search Troubleshooting Guide

This guide helps diagnose and resolve issues with the enhanced vector search capabilities, including dual search problems, document retrieval errors, and configuration issues.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Error Codes Reference](#error-codes-reference)
4. [Performance Issues](#performance-issues)
5. [Configuration Problems](#configuration-problems)
6. [Debug Mode](#debug-mode)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Recovery Procedures](#recovery-procedures)

## Quick Diagnostics

### Health Check

First, verify the system health:

```bash
# Check API health
curl http://localhost:8000/health

# Check enhanced search components
curl http://localhost:8000/status
```

Expected healthy response:
```json
{
  "status": "healthy",
  "components": {
    "enhanced_search": "healthy",
    "document_retrieval": "healthy",
    "search_templates": "healthy",
    "database": "healthy"
  }
}
```

### Basic Functionality Test

Test basic enhanced search functionality:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "table configuration example",
    "enable_contextual_search": true,
    "code_context": {"code_type": "json"}
  }'
```

### Template Loading Test

Verify search templates are loading correctly:

```bash
# Check if templates are accessible
ls -la server/genericsuite_codegen/config/search_templates.json

# Validate JSON syntax
python -m json.tool server/genericsuite_codegen/config/search_templates.json
```

## Common Issues

### 1. Dual Search Not Working

**Symptoms:**
- Only user query results returned
- No contextual search results
- Missing `context_used` in response

**Diagnosis:**
```bash
# Check if enhanced search is enabled
grep ENHANCED_SEARCH_ENABLED .env

# Check template configuration
cat server/genericsuite_codegen/config/search_templates.json | jq '.templates'
```

**Solutions:**

1. **Enable Enhanced Search:**
```bash
echo "ENHANCED_SEARCH_ENABLED=true" >> .env
```

2. **Fix Template Configuration:**
```json
{
  "templates": {
    "json": {
      "template": "examples of how to create a JSON table configuration files in Genericsuite",
      "file_type_filter": "json",
      "priority": 1,
      "enabled": true
    }
  }
}
```

3. **Check Context Determination:**
```python
# Test context determination
from genericsuite_codegen.agent.context_determination import ContextDeterminationService

service = ContextDeterminationService()
context = service.determine_context("create a table configuration", "json_config")
print(f"Detected context: {context}")
```

### 2. Document Retrieval Failures

**Symptoms:**
- `DOCUMENT_RETRIEVAL_ERROR` errors
- Empty document content
- Path traversal errors

**Diagnosis:**
```bash
# Check local repo path exists
ls -la local_repo_files/

# Check permissions
find local_repo_files/ -type f -name "*.md" | head -5 | xargs ls -la

# Test document retrieval directly
python -c "
from genericsuite_codegen.agent.document_retrieval_tool import DocumentRetrievalTool
tool = DocumentRetrievalTool()
result = tool.retrieve_document('local_repo_files/genericsuite-basecamp/README.md')
print(result)
"
```

**Solutions:**

1. **Fix Path Issues:**
```bash
# Ensure local_repo_files directory exists
mkdir -p local_repo_files

# Set correct permissions
chmod -R 755 local_repo_files/
```

2. **Update Configuration:**
```bash
# Set correct local repo path
echo "LOCAL_REPO_PATH=local_repo_files" >> .env
```

3. **Fix Path Traversal:**
```python
# Ensure paths are within allowed directories
ALLOWED_PATHS = [
    "local_repo_files",
    "generated_files"
]
```

### 3. Context Determination Issues

**Symptoms:**
- Wrong contextual search triggered
- Low confidence scores
- Generic context used instead of specific

**Diagnosis:**
```python
# Test context determination with debug info
from genericsuite_codegen.agent.context_determination import ContextDeterminationService

service = ContextDeterminationService()
context = service.determine_context(
    "create a langchain tool for authentication", 
    "python_code"
)
print(f"Context: {context}")
print(f"Confidence: {context.confidence}")
```

**Solutions:**

1. **Update Context Rules:**
```python
# In context_determination.py, update keyword patterns
CONTEXT_PATTERNS = {
    "langchain": r"\b(langchain|tool|agent|chain)\b",
    "mcp": r"\b(mcp|server|protocol|model context)\b",
    "json": r"\b(table|config|configuration|schema|json)\b"
}
```

2. **Adjust Confidence Thresholds:**
```python
CONFIDENCE_THRESHOLDS = {
    "high": 0.7,    # Reduced from 0.8
    "medium": 0.5,  # Reduced from 0.6
    "low": 0.3      # Reduced from 0.4
}
```

### 4. Search Template Loading Errors

**Symptoms:**
- `TEMPLATE_LOAD_ERROR` errors
- Fallback templates used
- Missing template configurations

**Diagnosis:**
```bash
# Check template file syntax
python -m json.tool server/genericsuite_codegen/config/search_templates.json

# Check file permissions
ls -la server/genericsuite_codegen/config/search_templates.json

# Test template loading
python -c "
from genericsuite_codegen.agent.search_templates import SearchTemplateManager
manager = SearchTemplateManager()
print(manager.get_template('json'))
"
```

**Solutions:**

1. **Fix JSON Syntax:**
```bash
# Validate and fix JSON
jsonlint server/genericsuite_codegen/config/search_templates.json
```

2. **Restore Default Templates:**
```json
{
  "templates": {
    "json": {
      "template": "examples of how to create a JSON table configuration files in Genericsuite",
      "file_type_filter": "json",
      "priority": 1
    },
    "langchain": {
      "template": "examples of how to create a Python Langchain Tool in Genericsuite",
      "file_type_filter": "py",
      "priority": 1
    }
  }
}
```

3. **Set Correct Permissions:**
```bash
chmod 644 server/genericsuite_codegen/config/search_templates.json
```

### 5. Database Connection Issues

**Symptoms:**
- Vector search failures
- Database connection errors
- Empty search results

**Diagnosis:**
```bash
# Check MongoDB connection
mongo --eval "db.adminCommand('ismaster')"

# Test database connection from Python
python -c "
from genericsuite_codegen.database.setup import test_database_connection, initialize_database
db = initialize_database()
result = test_database_connection(db)
print(f'Database connection: {result}')
"
```

**Solutions:**

1. **Fix MongoDB Connection:**
```bash
# Start MongoDB
brew services start mongodb-community
# or
sudo systemctl start mongod

# Check connection string
echo $MONGODB_URI
```

2. **Update Connection String:**
```bash
echo "MONGODB_URI=mongodb://localhost:27017/genericsuite_codegen" >> .env
```

3. **Verify Collection Setup:**
```python
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client.genericsuite_codegen
collections = db.list_collection_names()
print(f"Collections: {collections}")
```

## Error Codes Reference

### Enhanced Search Errors

| Error Code | Description | Common Causes | Solutions |
|------------|-------------|---------------|-----------|
| `DUAL_SEARCH_ERROR` | Dual search operation failed | Template loading failure, database connection | Check templates and database |
| `CONTEXT_DETERMINATION_ERROR` | Context determination failed | Invalid query, missing patterns | Update context rules |
| `SEARCH_MERGE_ERROR` | Search result merging failed | Conflicting results, invalid scores | Check merge configuration |

### Document Retrieval Errors

| Error Code | Description | Common Causes | Solutions |
|------------|-------------|---------------|-----------|
| `FILE_NOT_FOUND` | Document not found | Wrong path, missing file | Check file path and existence |
| `PATH_TRAVERSAL` | Path traversal detected | Security violation | Use allowed paths only |
| `BINARY_FILE` | Binary file not supported | Trying to read binary content | Use text files only |
| `ENCODING_ERROR` | File encoding issue | Invalid encoding, corrupted file | Check file encoding |
| `SIZE_LIMIT_EXCEEDED` | File too large | File exceeds size limit | Increase limit or use smaller files |

### Template Errors

| Error Code | Description | Common Causes | Solutions |
|------------|-------------|---------------|-----------|
| `TEMPLATE_LOAD_ERROR` | Template loading failed | Invalid JSON, missing file | Fix template file |
| `TEMPLATE_VALIDATION_ERROR` | Template validation failed | Invalid template structure | Update template format |
| `TEMPLATE_NOT_FOUND` | Template not found | Missing template for code type | Add missing template |

## Performance Issues

### Slow Search Response

**Symptoms:**
- Search requests taking > 10 seconds
- Timeouts on complex queries
- High CPU usage

**Diagnosis:**
```bash
# Check search performance
time curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'

# Monitor system resources
top -p $(pgrep -f "genericsuite_codegen")
```

**Solutions:**

1. **Enable Caching:**
```bash
echo "SEARCH_RESULT_CACHE_ENABLED=true" >> .env
echo "SEARCH_RESULT_CACHE_TTL=1800" >> .env
```

2. **Optimize Database:**
```javascript
// Create vector index in MongoDB
db.documents.createIndex({"embedding": "2dsphere"})
```

3. **Tune Parallel Processing:**
```bash
echo "DUAL_SEARCH_PARALLEL_ENABLED=true" >> .env
echo "MAX_CONCURRENT_SEARCHES=5" >> .env
```

### High Memory Usage

**Symptoms:**
- Memory usage > 2GB
- Out of memory errors
- System slowdown

**Solutions:**

1. **Limit Document Size:**
```bash
echo "DOCUMENT_RETRIEVAL_MAX_SIZE=5242880" >> .env  # 5MB
```

2. **Enable Compression:**
```python
CACHE_CONFIG = {
    "document_content": {
        "compression": True,
        "max_size": 100
    }
}
```

3. **Reduce Batch Sizes:**
```bash
echo "EMBEDDING_BATCH_SIZE=16" >> .env
```

### Database Performance

**Symptoms:**
- Slow vector searches
- Database timeouts
- High database CPU

**Solutions:**

1. **Optimize Indexes:**
```javascript
// MongoDB optimization
db.documents.createIndex({"metadata.file_type": 1})
db.documents.createIndex({"metadata.source": 1})
```

2. **Tune Connection Pool:**
```bash
echo "MONGODB_MAX_POOL_SIZE=20" >> .env
echo "MONGODB_TIMEOUT=15000" >> .env
```

## Configuration Problems

### Environment Variables Not Loading

**Diagnosis:**
```bash
# Check .env file exists
ls -la .env

# Check environment variables
env | grep ENHANCED_SEARCH
```

**Solutions:**

1. **Create .env File:**
```bash
cp .env.example .env
```

2. **Load Environment:**
```bash
source .env
# or
export $(cat .env | xargs)
```

### Template Configuration Issues

**Diagnosis:**
```python
# Test template configuration loading
from genericsuite_codegen.config.config_loader import load_config
config = load_config()
print(config)
```

**Solutions:**

1. **Reset to Defaults:**
```bash
cp server/genericsuite_codegen/config/examples/minimal_config.json \
   server/genericsuite_codegen/config/search_templates.json
```

2. **Validate Configuration:**
```python
from genericsuite_codegen.config.config_validator import validate_config
errors = validate_config()
if errors:
    print(f"Configuration errors: {errors}")
```

## Debug Mode

### Enable Debug Logging

```bash
# Enable debug mode
echo "ENHANCED_SEARCH_DEBUG=true" >> .env
echo "LOG_LEVEL=DEBUG" >> .env

# Restart the server
make restart
```

### Debug Output Examples

**Context Determination Debug:**
```
DEBUG:context_determination:Analyzing query: "create a table configuration"
DEBUG:context_determination:Keywords found: ['table', 'configuration']
DEBUG:context_determination:Pattern match for 'json': True
DEBUG:context_determination:Confidence score: 0.85
DEBUG:context_determination:Selected context: json
```

**Document Retrieval Debug:**
```
DEBUG:document_retrieval:Retrieving document: local_repo_files/example.md
DEBUG:document_retrieval:Path validation passed
DEBUG:document_retrieval:File exists: True
DEBUG:document_retrieval:File size: 2048 bytes
DEBUG:document_retrieval:Encoding detected: utf-8
DEBUG:document_retrieval:Document retrieved successfully
```

**Search Template Debug:**
```
DEBUG:search_templates:Loading templates from: search_templates.json
DEBUG:search_templates:Template 'json' loaded successfully
DEBUG:search_templates:Template query: "examples of how to create a JSON table configuration files in Genericsuite"
DEBUG:search_templates:File filter: json
```

### Debug Tools

**Interactive Debug Session:**
```python
# Start Python REPL with debug environment
python -c "
import os
os.environ['ENHANCED_SEARCH_DEBUG'] = 'true'

from genericsuite_codegen.agent.enhanced_search import EnhancedVectorSearch
from genericsuite_codegen.agent.tools import KnowledgeBaseTool
from genericsuite_codegen.agent.search_templates import SearchTemplateManager

# Initialize components
kb_tool = KnowledgeBaseTool()
template_manager = SearchTemplateManager()
enhanced_search = EnhancedVectorSearch(kb_tool, template_manager)

# Test dual search
import asyncio
result = asyncio.run(enhanced_search.dual_search(
    'create a table configuration',
    {'code_type': 'json', 'framework': None, 'confidence': 0.9}
))
print(result)
"
```

## Monitoring and Logging

### Log File Locations

```bash
# Server logs
tail -f server/logs/genericsuite_codegen.log

# Enhanced search specific logs
tail -f server/logs/enhanced_search.log

# Document retrieval logs
tail -f server/logs/document_retrieval.log
```

### Key Metrics to Monitor

1. **Search Performance:**
   - Average search response time
   - Dual search success rate
   - Context determination accuracy

2. **Document Retrieval:**
   - Document retrieval success rate
   - Average file size retrieved
   - Cache hit rate

3. **System Health:**
   - Memory usage
   - CPU usage
   - Database connection status

### Monitoring Commands

```bash
# Monitor search performance
grep "search_duration" server/logs/enhanced_search.log | tail -10

# Monitor error rates
grep "ERROR" server/logs/genericsuite_codegen.log | wc -l

# Monitor document retrieval
grep "document_retrieved" server/logs/document_retrieval.log | tail -10
```

## Recovery Procedures

### Reset Enhanced Search Configuration

```bash
# Backup current configuration
cp server/genericsuite_codegen/config/search_templates.json \
   server/genericsuite_codegen/config/search_templates.json.backup

# Reset to defaults
cp server/genericsuite_codegen/config/examples/minimal_config.json \
   server/genericsuite_codegen/config/search_templates.json

# Restart server
make restart
```

### Clear Caches

```bash
# Clear search result cache
redis-cli FLUSHDB  # If using Redis
# or
rm -rf /tmp/search_cache/*

# Clear document cache
rm -rf /tmp/document_cache/*

# Restart server
make restart
```

### Database Recovery

```bash
# Backup database
mongodump --db genericsuite_codegen --out backup/

# Drop and recreate collections
mongo genericsuite_codegen --eval "db.documents.drop()"

# Reinitialize database
python -c "
from genericsuite_codegen.database.setup import initialize_database
initialize_database()
"

# Re-ingest documents
make update-knowledge-base
```

### Complete System Reset

```bash
# Stop all services
make down

# Clean up data
rm -rf local_mongodb_data/data/*
rm -rf /tmp/*cache*

# Reset configuration
cp .env.example .env
cp server/genericsuite_codegen/config/examples/minimal_config.json \
   server/genericsuite_codegen/config/search_templates.json

# Restart system
make init-app-environment
make run
```

## Getting Help

### Log Analysis

When reporting issues, include:

1. **Error logs:**
```bash
grep -A 5 -B 5 "ERROR" server/logs/genericsuite_codegen.log | tail -20
```

2. **Configuration:**
```bash
cat .env | grep -v "API_KEY\|SECRET"
cat server/genericsuite_codegen/config/search_templates.json
```

3. **System information:**
```bash
python --version
pip list | grep -E "(pydantic|fastapi|pymongo)"
```

### Common Support Scenarios

1. **"Enhanced search not working"**
   - Check if `ENHANCED_SEARCH_ENABLED=true`
   - Verify template configuration
   - Test context determination

2. **"Document retrieval failing"**
   - Check file permissions
   - Verify path configuration
   - Test with simple file

3. **"Poor search quality"**
   - Review template queries
   - Check context determination rules
   - Analyze search result merging

4. **"Performance issues"**
   - Enable caching
   - Check database indexes
   - Monitor resource usage

### Debug Checklist

Before reporting issues, verify:

- [ ] Enhanced search is enabled
- [ ] Template configuration is valid JSON
- [ ] Local repo files exist and are accessible
- [ ] Database connection is working
- [ ] Environment variables are loaded
- [ ] Debug logging is enabled
- [ ] System has sufficient resources
- [ ] No conflicting configurations