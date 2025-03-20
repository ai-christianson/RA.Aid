from mcp.server.fastmcp import FastMCP

mcp = FastMCP("example")

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b

if __name__ == "__main__":
    mcp.run()
