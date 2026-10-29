"""Serve the placeholder store catalog over MCP (stdio), using the MCP SDK's FastMCP."""

from mcp.server.fastmcp import FastMCP

from examples.store_catalog import build_catalog

mcp = FastMCP("store")
for tool in build_catalog():
    mcp.add_tool(tool.func, name=tool.name, description=tool.description)

if __name__ == "__main__":
    mcp.run()
