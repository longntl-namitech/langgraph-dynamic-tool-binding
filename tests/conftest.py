from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool


class RecordingFakeModel(GenericFakeChatModel):
    """Scripted chat model that records the tool names bound to each call."""

    bound: list[list[str]] = []

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        self.bound.append([t["name"] if isinstance(t, dict) else t.name for t in tools])
        return self


def script(*messages: AIMessage) -> RecordingFakeModel:
    return RecordingFakeModel(messages=iter(messages), bound=[])


def call(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def make_tool(name: str) -> BaseTool:
    entity, action = name.rsplit("_", 1)
    return StructuredTool.from_function(
        func=lambda id: f"{name} ok",
        name=name,
        description=f"{action.capitalize()} a {entity}.",
    )


@pytest.fixture
def catalog() -> list[BaseTool]:
    names = [
        f"{entity}_{action}"
        for entity in ["order", "supplier", "customer", "product"]
        for action in ["list", "get", "create", "update", "delete"]
    ]
    return [make_tool(n) for n in names]
