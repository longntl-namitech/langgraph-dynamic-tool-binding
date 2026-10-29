"""Run an agent over the 170-tool store catalog with search-then-bind.

    uv run --extra examples python -m examples.quickstart "Create a coupon called SUMMER10"

Needs an OpenAI or Azure OpenAI key in the environment, see examples/model.py.
Run with `uv run --env-file .env ...` to load it from .env.
"""

import sys

from langchain.agents import create_agent

from examples.model import make_model
from examples.store_catalog import build_catalog
from langgraph_dynamic_tool_binding import DynamicToolBindingMiddleware


def main(question: str) -> None:
    agent = create_agent(
        make_model(),
        system_prompt="You manage a store through tools. Use search_tools to find them.",
        middleware=[DynamicToolBindingMiddleware(build_catalog())],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    for message in result["messages"]:
        for tool_call in getattr(message, "tool_calls", []):
            print(f"-> {tool_call['name']}({tool_call['args']})")
    print("bound after run:", result.get("selected_tool_names", []))
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "Create a coupon called SUMMER10")
