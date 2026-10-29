"""Search-then-bind middleware for agents with a large tool catalog.

The whole catalog is registered on the agent, so the tool node can run any of it,
but the model is never bound to the whole catalog. Each model call is bound to the
other agent tools, a ``search_tools`` meta-tool, and only the catalog tools that
``search_tools`` has returned so far in this thread. That keeps the request under
provider limits (OpenAI accepts at most 128 tools) and keeps unused tool schemas
out of the prompt.

Two hooks do the work:

- ``wrap_model_call`` trims the request's tool list to the discovered set.
- ``wrap_tool_call`` reads the names a ``search_tools`` call returned and merges
  them into ``selected_tool_names`` in the agent state.

The discovered set is a union across calls and is checkpointed with the thread,
so a follow-up turn starts with the tools earlier turns found.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command

from langgraph_dynamic_tool_binding.matcher import KeywordToolMatcher, ToolMatcher

logger = logging.getLogger(__name__)

SEARCH_TOOL_NAME = "search_tools"

DEFAULT_SEARCH_DESCRIPTION = (
    "Find the tools available for a task. Call this with a short query naming the "
    "entity and the action you need (for example 'order create' or 'supplier list'). "
    "It returns matching tool names and descriptions; the returned tools become "
    "callable on your next step. Call it again for another entity: earlier matches "
    "stay available."
)


def merge_names(current: list[str] | None, new: list[str] | None) -> list[str]:
    """Order-preserving union. Used as the state reducer for ``selected_tool_names``.

    It has to be a reducer: the model may call ``search_tools`` several times in one
    step, and LangGraph rejects concurrent writes to a last-value channel.
    """
    merged = list(current or [])
    for name in new or []:
        if name not in merged:
            merged.append(name)
    return merged


class ToolBindingState(AgentState):
    selected_tool_names: NotRequired[Annotated[list[str], merge_names]]


def _tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name") or tool.get("function", {}).get("name")
    return getattr(tool, "name", None)


class DynamicToolBindingMiddleware(AgentMiddleware):
    """Bind ``search_tools`` plus the discovered catalog tools, never the whole catalog.

    Args:
        tools: The catalog. The middleware registers these tools itself, so do not
            pass them to ``create_agent(tools=...)`` as well. Tools passed there are
            always bound.
        matcher: Ranks the catalog for a query. Defaults to ``KeywordToolMatcher()``.
        max_bound: Most catalog tools bound to one model call. When more have been
            discovered, the most recent ones win.
        search_tool_description: Overrides the ``search_tools`` description, for
            example to give domain-specific query examples.
    """

    state_schema = ToolBindingState

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        matcher: ToolMatcher | None = None,
        max_bound: int = 100,
        search_tool_description: str = DEFAULT_SEARCH_DESCRIPTION,
    ):
        super().__init__()
        self.catalog = list(tools)
        self.matcher = matcher or KeywordToolMatcher()
        self.max_bound = max_bound
        self._catalog_names = {t.name for t in self.catalog}
        self.search_tool = StructuredTool.from_function(
            func=self._search,
            name=SEARCH_TOOL_NAME,
            description=search_tool_description,
            response_format="content_and_artifact",
        )
        self.tools = [self.search_tool, *self.catalog]

    def _search(self, query: str) -> tuple[str, list[str]]:
        matches = self.matcher.search(query, self.catalog)
        names = [t.name for t in matches]
        logger.debug("search_tools(%r) -> %s", query, names)
        if not matches:
            return "No matching tools. Try other words for the entity or action.", []
        lines = [f"- {t.name}: {(t.description or '').strip()[:200]}" for t in matches]
        return "Matching tools:\n" + "\n".join(lines), names

    # Model call: trim the bound tools.

    def bound_tools(self, request: ModelRequest) -> list[BaseTool | dict[str, Any]]:
        selected = request.state.get("selected_tool_names") or []
        selected = set(selected[-self.max_bound :])
        return [
            tool
            for tool in request.tools
            if _tool_name(tool) not in self._catalog_names or _tool_name(tool) in selected
        ]

    def wrap_model_call(self, request: ModelRequest, handler: Callable) -> Any:
        return handler(request.override(tools=self.bound_tools(request)))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]
    ) -> Any:
        return await handler(request.override(tools=self.bound_tools(request)))

    # Tool call: record what search_tools discovered.

    def _capture(self, request: Any, result: ToolMessage | Command) -> ToolMessage | Command:
        if request.tool_call.get("name") != SEARCH_TOOL_NAME or not isinstance(result, ToolMessage):
            return result
        names = [n for n in (result.artifact or []) if n in self._catalog_names]
        if not names:
            return result
        # Only the new names: the reducer unions them, also across parallel calls.
        return Command(update={"selected_tool_names": names, "messages": [result]})

    def wrap_tool_call(self, request: Any, handler: Callable) -> ToolMessage | Command:
        return self._capture(request, handler(request))

    async def awrap_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        return self._capture(request, await handler(request))
