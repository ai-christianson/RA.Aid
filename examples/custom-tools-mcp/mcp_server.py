from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("example")

@mcp.tool()
def multiply(a: int, b: int) -> str:
    """Multiplies two numbers together."""
    return a * b

if __name__ == "__main__":
    mcp.run()
