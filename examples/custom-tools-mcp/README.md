# examples/custom-tools-mcp

The `--custom-tools <path>` flag is used to provide additional tools to the LLM agents.

```bash
ra-aid --custom-tools tools/custom_tools.py
```

Any custom tool function name, docstring, and signature (e.g. arguments and return type) will be included in the LLM text prompt so that the agents can choose to invoke any provided tools.

The `--custom-tools` argument should point to one file in your project folder that exports a symbol named `tools`.

```python
# tools/custom_tools.py
from langchain_core.tools import tool

@tool
def custom_add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

tools = [
    custom_add,
]
```

Tools can also be dynamically added by configuring MCP servers.

```python
# tools/custom_tools.py
from ra_aid.utils.mcp_client import MultiServerMCPClient_Sync

mcp_client = MultiServerMCPClient_Sync({
    "mcp-example-server": {
        "transport": "stdio",
        "command": "python",
        "args": ["./examples/custom-tools-mcp/mcp_server.py"],
    },
    "mcp-weather": {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@mcp-examples/weather",
        ]
    },
})

mcp_tools = mcp_client.get_tools_sync()

tools = mcp_tools
```