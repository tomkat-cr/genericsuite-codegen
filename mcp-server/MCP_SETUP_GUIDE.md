# MCP Server Setup Guide

## Quick Setup for Empty Configuration

If your `.kiro/settings/mcp.json` file is empty or missing, follow these steps to set up the GenericSuite CodeGen MCP server.

## Step 1: Verify MCP Configuration File

Check if your MCP configuration file exists and has content:

```bash
# Check if file exists
ls -la .kiro/settings/mcp.json

# View current content
cat .kiro/settings/mcp.json
```

If the file is empty or contains only `{"mcpServers": {}}`, proceed with the setup.

## Step 2: Get Absolute Path

Get the absolute path to your GenericSuite CodeGen project:

```bash
# From the project root directory
pwd
# Example output: /Users/username/projects/genericsuite-codegen
```

## Step 3: Configure MCP Server

### Option A: Using Shell Script (Recommended)

Replace the content of `.kiro/settings/mcp.json` with:

```json
{
  "mcpServers": {
    "genericsuite-codegen": {
      "command": "sh",
      "args": ["/REPLACE/WITH/YOUR/PATH/genericsuite-codegen/mcp-server/run_mcp_server.sh"],
      "env": {
        "MCP_API_KEY": "ag-api-key-genericsuite-codegen",
        "MCP_TRANSPORT": "stdio"
      },
      "disabled": false,
      "autoApprove": [
        "search_knowledge_base",
        "get_knowledge_base_stats"
      ]
    }
  }
}
```

### Option B: Using Poetry Directly

```json
{
  "mcpServers": {
    "genericsuite-codegen": {
      "command": "poetry",
      "args": ["run", "python", "start_mcp_server.py"],
      "cwd": "/REPLACE/WITH/YOUR/PATH/genericsuite-codegen/mcp-server",
      "env": {
        "MCP_API_KEY": "ag-api-key-genericsuite-codegen",
        "MCP_SERVER_PORT": "8070",
        "MCP_DEBUG": "0"
      },
      "disabled": false,
      "autoApprove": [
        "search_knowledge_base",
        "get_knowledge_base_stats"
      ]
    }
  }
}
```

**Important**: Replace `/REPLACE/WITH/YOUR/PATH` with the actual absolute path from Step 2.

## Step 4: Verify MCP Server Dependencies

Ensure the MCP server dependencies are installed:

```bash
cd mcp-server
poetry install
```

## Step 5: Test MCP Server

Test the MCP server independently:

```bash
cd mcp-server
make test
```

## Step 6: Restart Kiro

After updating the configuration:

1. Restart Kiro completely
2. Or use the command palette: "MCP: Restart Servers"
3. Check the MCP Server view in Kiro to verify connection

## Step 7: Verify Integration

Test the integration by trying one of the auto-approved tools:

1. Open Kiro
2. Try using the `search_knowledge_base` tool
3. Search for "GenericSuite table configuration"

## Troubleshooting

### Server Not Starting

1. **Check paths**: Ensure all paths in the configuration are absolute and correct
2. **Check permissions**: Ensure the shell script is executable:
   ```bash
   chmod +x mcp-server/run_mcp_server.sh
   ```
3. **Check dependencies**: Verify Poetry and Python are available in the path

### Connection Issues

1. **Check logs**: Look at Kiro's MCP server logs
2. **Test manually**: Run the MCP server command manually to check for errors
3. **Verify environment**: Ensure all required environment variables are set

### Tool Execution Errors

1. **Check API keys**: Verify OpenAI API key is set in the environment
2. **Check database**: Ensure MongoDB is running if using local database
3. **Check permissions**: Verify file access permissions for local document retrieval

## Configuration Options

### Auto-Approve Settings

Add tools to `autoApprove` that you want to use without manual confirmation:

```json
"autoApprove": [
  "search_knowledge_base",
  "get_knowledge_base_stats",
  "generate_json_config",
  "generate_python_code"
]
```

### Environment Variables

Common environment variables you can set in the `env` section:

- `MCP_API_KEY`: API key for authentication (if enabled)
- `MCP_DEBUG`: Set to "1" for debug logging
- `MCP_SERVER_PORT`: Port for HTTP transport (default: 8070)
- `OPENAI_API_KEY`: OpenAI API key for AI functionality
- `LLM_MODEL_NAME`: LLM model to use (default: gpt-4o-mini)

## Next Steps

Once configured, you can:

1. Use the knowledge base search tools to find GenericSuite documentation
2. Generate JSON configurations for tables and forms
3. Create Python tools (LangChain, MCP) following GenericSuite patterns
4. Generate frontend and backend code with GenericSuite conventions
5. Access complete documents from the local knowledge base

For more detailed usage information, see [MCP_USAGE.md](./MCP_USAGE.md).