import json
import sys

import pytest
from conftest import call, script
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from examples.store_catalog import ACTIONS, ENTITIES, build_catalog
from langgraph_dynamic_tool_binding import SEARCH_TOOL_NAME, DynamicToolBindingMiddleware


def test_catalog_has_170_unique_tools():
    names = [t.name for t in build_catalog()]
    assert len(names) == len(ENTITIES) * len(ACTIONS) == 170
    assert len(set(names)) == 170


def test_catalog_is_deterministic():
    tool = {t.name: t for t in build_catalog()}["supplier_update"]
    out = json.loads(tool.invoke({"id": "sup-1", "name": "Acme"}))
    assert out == {"id": "sup-1", "type": "supplier", "name": "Acme", "notes": "", "updated": True}


def test_full_catalog_is_never_bound_whole():
    model = script(
        AIMessage("", tool_calls=[call(SEARCH_TOOL_NAME, "1", query="coupon create")]),
        AIMessage("", tool_calls=[call("coupon_create", "2", name="SUMMER10")]),
        AIMessage("done"),
    )
    agent = create_agent(model, middleware=[DynamicToolBindingMiddleware(build_catalog())])

    agent.invoke({"messages": [HumanMessage("create coupon SUMMER10")]})

    assert model.bound[0] == [SEARCH_TOOL_NAME]
    assert "coupon_create" in model.bound[1]
    assert all(len(names) <= 128 for names in model.bound)


@pytest.mark.asyncio
async def test_catalog_over_mcp():
    client_module = pytest.importorskip("langchain_mcp_adapters.client")
    client = client_module.MultiServerMCPClient(
        {
            "store": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "examples.fastmcp_server"],
            }
        }
    )
    tools = await client.get_tools()
    assert len(tools) == 170
    result = await {t.name: t for t in tools}["supplier_get"].ainvoke({"id": "sup-1"})
    assert '"type": "supplier"' in str(result)
