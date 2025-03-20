# Using Custom Tools with RA.Aid

LLM tool calling allows language models to invoke external functions during their execution. This enables the model to:
- Access external data sources
- Perform complex computations
- Interact with systems and APIs
- Execute code

RA.Aid supports custom tools that can be invoked by the LLM during task execution. These tools are defined in Python and can be used to extend the capabilities of the AI agent.

## Adding Custom Tools

The `--custom-tools <path>` flag is used to provide additional tools to the LLM agents.

```bash
ra-aid --custom-tools tools/custom_tools.py
```

Any custom tool function name, docstring, and signature (e.g. arguments and return type) will be included in the LLM text prompt so that the agents can choose to invoke any provided tools.

## Creating Custom Tools

The `--custom-tools` argument should point to one file in your project folder that exports a symbol named `tools`.

Here's an example custom tools file:

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

## Using MCP for Advanced Tools

Tools can also be dynamically added by configuring MCP servers. Here's an example using the MultiServerMCPClient_Sync:

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

This allows you to integrate complex tools and services while keeping your custom tools file clean and maintainable.
