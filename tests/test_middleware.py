import pytest
from conftest import call, script
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_dynamic_tool_binding import (
    SEARCH_TOOL_NAME,
    DynamicToolBindingMiddleware,
    merge_names,
)


def test_merge_names_is_ordered_union():
    assert merge_names(["a", "b"], ["b", "c", "a", "d"]) == ["a", "b", "c", "d"]
    assert merge_names(None, None) == []


def test_binds_only_search_tool_then_discovered_tools(catalog):
    model = script(
        AIMessage("", tool_calls=[call(SEARCH_TOOL_NAME, "1", query="order create")]),
        AIMessage("", tool_calls=[call("order_create", "2", id="o1")]),
        AIMessage("done"),
    )
    agent = create_agent(model, middleware=[DynamicToolBindingMiddleware(catalog)])

    result = agent.invoke({"messages": [HumanMessage("create an order")]})

    assert model.bound[0] == [SEARCH_TOOL_NAME]
    assert "order_create" in model.bound[1]
    assert "supplier_list" not in model.bound[1]  # matches neither "order" nor "create"
    assert "order_create" in result["selected_tool_names"]
    assert result["messages"][-1].content == "done"
    tool_outputs = [m.content for m in result["messages"] if m.type == "tool"]
    assert "order_create ok" in tool_outputs


def test_parallel_searches_are_merged(catalog):
    model = script(
        AIMessage(
            "",
            tool_calls=[
                call(SEARCH_TOOL_NAME, "1", query="order"),
                call(SEARCH_TOOL_NAME, "2", query="supplier"),
            ],
        ),
        AIMessage("done"),
    )
    agent = create_agent(model, middleware=[DynamicToolBindingMiddleware(catalog)])

    result = agent.invoke({"messages": [HumanMessage("orders and suppliers")]})

    selected = result["selected_tool_names"]
    assert any(n.startswith("order_") for n in selected)
    assert any(n.startswith("supplier_") for n in selected)


def test_follow_up_turn_keeps_discovered_tools(catalog):
    model = script(
        AIMessage("", tool_calls=[call(SEARCH_TOOL_NAME, "1", query="customer get")]),
        AIMessage("first"),
        AIMessage("second"),
    )
    agent = create_agent(
        model,
        middleware=[DynamicToolBindingMiddleware(catalog)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t1"}}

    agent.invoke({"messages": [HumanMessage("find customer")]}, config)
    agent.invoke({"messages": [HumanMessage("again")]}, config)

    assert "customer_get" in model.bound[-1]


def test_max_bound_keeps_most_recent(catalog):
    model = script(
        AIMessage("", tool_calls=[call(SEARCH_TOOL_NAME, "1", query="order")]),
        AIMessage("", tool_calls=[call(SEARCH_TOOL_NAME, "2", query="supplier")]),
        AIMessage("done"),
    )
    agent = create_agent(model, middleware=[DynamicToolBindingMiddleware(catalog, max_bound=5)])

    agent.invoke({"messages": [HumanMessage("go")]})

    catalog_bound = [n for n in model.bound[-1] if n != SEARCH_TOOL_NAME]
    assert len(catalog_bound) == 5
    assert all(n.startswith("supplier_") for n in catalog_bound)


def test_other_agent_tools_stay_bound(catalog):
    extra = catalog[0].model_copy(update={"name": "always_on"})
    model = script(AIMessage("done"))
    agent = create_agent(model, tools=[extra], middleware=[DynamicToolBindingMiddleware(catalog)])

    agent.invoke({"messages": [HumanMessage("hi")]})

    assert sorted(model.bound[0]) == sorted(["always_on", SEARCH_TOOL_NAME])


def test_no_match_leaves_state_unchanged(catalog):
    model = script(
        AIMessage("", tool_calls=[call(SEARCH_TOOL_NAME, "1", query="weather")]),
        AIMessage("done"),
    )
    agent = create_agent(model, middleware=[DynamicToolBindingMiddleware(catalog)])

    result = agent.invoke({"messages": [HumanMessage("weather?")]})

    assert result.get("selected_tool_names", []) == []
    assert model.bound[-1] == [SEARCH_TOOL_NAME]


@pytest.mark.asyncio
async def test_async_path(catalog):
    model = script(
        AIMessage("", tool_calls=[call(SEARCH_TOOL_NAME, "1", query="product list")]),
        AIMessage("", tool_calls=[call("product_list", "2", id="p")]),
        AIMessage("done"),
    )
    agent = create_agent(model, middleware=[DynamicToolBindingMiddleware(catalog)])

    result = await agent.ainvoke({"messages": [HumanMessage("list products")]})

    assert "product_list" in model.bound[1]
    assert result["messages"][-1].content == "done"
