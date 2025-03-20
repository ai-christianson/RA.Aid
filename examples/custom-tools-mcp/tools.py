from langchain_core.tools import tool
from ra_aid.utils.mcp_client import MultiServerMCPClient_Sync

@tool
def custom_add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

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

local_tools = [
    custom_add,
]

mcp_tools = mcp_client.get_tools_sync()

tools = local_tools + mcp_tools