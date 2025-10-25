#!/usr/bin/env bash
set -euo pipefail
: "${TAVILY_API_KEY:?TAVILY_API_KEY is required}"


# Run Tavily remote MCP using npx
npx -y mcp-remote "https://mcp.tavily.com/mcp/?tavilyApiKey=${TAVILY_API_KEY}"


