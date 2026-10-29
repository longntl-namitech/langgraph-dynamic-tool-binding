# langgraph-dynamic-tool-binding

Search-then-bind tool loading for LangChain v1 agents with large tool catalogs.

An agent that talks to a big API (an MCP server with 150+ tools, say) runs into two problems when every tool is bound to every model call:

- OpenAI and Azure OpenAI reject requests with more than 128 tools: `Invalid 'tools': array too long. Expected an array with maximum length 128`.
- Below that limit, every tool schema is still sent on every call, so most of the prompt is tools the model will not use.

This library gives the agent one `search_tools` meta-tool instead. The model searches for what it needs, and only the matching tools are bound on the next call. The whole catalog stays registered, so any tool can still run.

## Install

```bash
pip install "git+https://github.com/longntl-namitech/langgraph-dynamic-tool-binding"
```

Python 3.11+, `langchain>=1`, `langgraph>=1`.

## Usage

```python
from langchain.agents import create_agent
from langgraph_dynamic_tool_binding import DynamicToolBindingMiddleware

agent = create_agent(
    "openai:gpt-5-mini",
    middleware=[DynamicToolBindingMiddleware(catalog)],  # catalog: list of BaseTool
)
agent.invoke({"messages": [{"role": "user", "content": "Create a coupon called SUMMER10"}]})
```

The middleware registers the catalog itself, so do not also pass it to `create_agent(tools=...)`. Tools passed there are always bound, which is useful for a few core tools. It also works with deepagents, which builds on the same middleware API.

Options:

| Argument | Default | Meaning |
|---|---|---|
| `matcher` | `KeywordToolMatcher()` | Ranks the catalog for a query. Any object with `search(query, tools) -> list[BaseTool]`. |
| `max_bound` | `100` | Most catalog tools bound to one call. The most recently discovered win. |
| `search_tool_description` | generic | Description of `search_tools`. Put domain query examples here. |

`KeywordToolMatcher(top_k=25, stopwords=...)` scores token overlap: 3 per query token found in the tool name, 2 for a partial name match, 1 for a description match. It works best with structured names such as `<entity>_<action>`.

## How it works

```
user ──▶ model (bound: search_tools + discovered tools)
           │ search_tools("coupon create")
           ▼
         matcher ──▶ ToolMessage(content=list, artifact=[names])
           │
         wrap_tool_call: names ──▶ state.selected_tool_names (union reducer)
           │
         wrap_model_call: next call binds search_tools + coupon_create, ...
           ▼
         model ──▶ coupon_create(name="SUMMER10")
```

- `wrap_model_call` removes catalog tools that have not been discovered from the request.
- `wrap_tool_call` reads the names a `search_tools` call returned (from the `ToolMessage` artifact) and writes them to `selected_tool_names`.
- `selected_tool_names` uses an order-preserving union reducer. The model can call `search_tools` several times in one step, and LangGraph rejects concurrent writes to a plain state key.
- The discovered set is part of the agent state, so with a checkpointer a follow-up turn starts with the tools earlier turns found.

## Benchmark

`benchmark/` compares binding the whole catalog with search-then-bind on a placeholder store-management API (`examples/store_catalog.py`: 34 entities x list/get/create/update/delete = 170 tools, mock JSON responses).

- 30 hand-written tasks: 12 single-entity, 10 multi-entity, 8 two-turn follow-ups. A task passes when every expected tool is called with the expected arguments.
- `bind_all_120` binds all tools of a 120-tool subset, the largest catalog that fits under the 128-tool limit. `search_bind_120` uses the same 120 tools through the middleware. `search_bind_170` uses the full catalog.
- Model: `gpt-5.4-nano` on Azure OpenAI, one run per task, 2026-10-29. Raw results: `benchmark/results/full-n30.json`.

| Arm | Task success | Mean input tokens per task | Mean LLM calls | Mean latency | Matcher recall | Mean tools bound |
|---|---|---|---|---|---|---|
| bind_all_120 | 30/30 | 12,482 | 2.7 | 6.2 s | - | 120 |
| search_bind_120 | 28/30 | 4,019 | 3.8 | 6.1 s | 1.00 | 20.7 |
| search_bind_170 | 28/30 | 3,951 | 3.8 | 5.9 s | 0.97 | 20.0 |
| bind_all_170 | request rejected (400, more than 128 tools) | | | | | |

On the 120-tool catalog, search-then-bind used **3.1x fewer input tokens** per task (3.4x on single-entity tasks, 2.7x on multi-entity, 3.2x on follow-ups), at the cost of about one extra model call per task. With 170 tools it kept the same cost, where binding everything is not possible.

What went wrong in the 4 failures:

- Twice the matcher returned the right tool but the model picked another one: for "Rename warehouse war-3 to North Hub" it searched "warehouse rename", got `warehouse_update`, and called `warehouse_list` instead. With every tool bound it chose correctly.
- Twice the model searched for only one of two entities in a multi-entity request ("order get" for "Get order ord-9 and its invoice inv-9") and stopped after the first.

So the trade is fewer tokens for a small drop in task success with a small model on multi-step requests. A stronger model, a hint in the system prompt to search once per entity, or domain examples in `search_tool_description` are the obvious things to try.

These numbers come from mock tools with short schemas (about 75 tokens each). Real API tools usually have longer descriptions and more parameters, which makes binding everything more expensive and the saving larger.

Run it yourself (about 0.6M tokens for the full set):

```bash
cp .env.example .env   # OpenAI or Azure OpenAI credentials
uv run --env-file .env --extra examples python -m benchmark.run --limit 5   # try 5 tasks first
uv run --env-file .env --extra examples python -m benchmark.run
```

## Examples

```bash
uv run --env-file .env --extra examples python -m examples.quickstart "Create a coupon called SUMMER10"
uv run --env-file .env --extra examples python -m examples.mcp_example "Rename supplier sup-1 to Acme"
```

`mcp_example.py` serves the same 170 tools from an MCP server (`examples/fastmcp_server.py`, stdio) and loads them with `langchain-mcp-adapters`.

### Custom StateGraph

The middleware targets `create_agent`. In a hand-written graph, use a matcher directly and bind per node:

```python
from langgraph_dynamic_tool_binding import KeywordToolMatcher

matcher = KeywordToolMatcher(top_k=10)

def agent_node(state):
    tools = matcher.search(state["task"], catalog)
    return {"messages": [model.bind_tools(tools).invoke(state["messages"])]}
```

### Returning tool errors to the model

Not part of this library, but it pairs well with it: a tool that raises ends the run unless the error goes back to the model as a message.

```python
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def tool_errors_to_model(request, handler):
    try:
        return handler(request)
    except Exception as exc:
        call = request.tool_call
        return ToolMessage(f"Error calling {call['name']}: {exc}", tool_call_id=call["id"], status="error")
```

## Development

```bash
uv sync --extra examples
uv run pytest        # no API key needed
uv run ruff check .
```

## License

MIT
