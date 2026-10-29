"""Load the 170-tool catalog from an MCP server and run an agent on it.

    uv run --extra examples python -m examples.mcp_example "Rename supplier sup-1 to Acme"

Needs an OpenAI or Azure OpenAI key in the environment, see examples/model.py.
Run with `uv run --env-file .env ...` to load it from .env.
"""

import asyncio
import sys

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from examples.model import make_model
from langgraph_dynamic_tool_binding import DynamicToolBindingMiddleware


async def main(question: str) -> None:
    client = MultiServerMCPClient(
        {
            "store": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "examples.fastmcp_server"],
            }
        }
    )
    tools = await client.get_tools()
    print(f"loaded {len(tools)} tools over MCP")

    agent = create_agent(
        make_model(),
        middleware=[DynamicToolBindingMiddleware(tools)],
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    print("bound after run:", result.get("selected_tool_names", []))
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "List the first page of suppliers"))
