"""Tool matchers: rank a catalog of tools against a short natural-language query."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from langchain_core.tools import BaseTool

DEFAULT_STOPWORDS = frozenset(
    {
        "a", "all", "an", "and", "for", "find", "get", "give", "in", "me", "my", "of",
        "on", "or", "please", "show", "the", "to", "tool", "tools", "want", "with",
    }
)  # fmt: skip


@runtime_checkable
class ToolMatcher(Protocol):
    """Anything that can pick the tools relevant to a query."""

    def search(self, query: str, tools: Sequence[BaseTool]) -> list[BaseTool]:
        """Return the best matches for ``query``, best first."""
        ...


def tokenize(text: str, stopwords: Iterable[str] = DEFAULT_STOPWORDS) -> list[str]:
    """Lowercase word tokens, splitting snake_case and camelCase, minus stopwords."""
    stop = set(stopwords)
    tokens: list[str] = []
    for part in re.split(r"[^a-zA-Z0-9]+", text or ""):
        for word in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", part):
            word = word.lower()
            if word not in stop:
                tokens.append(word)
    return tokens


class KeywordToolMatcher:
    """Deterministic token-overlap matcher over tool names and descriptions.

    Works best when tool names are structured (``<entity>_<action>``): a query like
    "update supplier" then scores ``supplier_update`` far above the rest. Each query
    token scores 3 for an exact name-token hit, 2 for a partial name-token hit and 1
    for a description hit.
    """

    def __init__(self, top_k: int = 25, stopwords: Iterable[str] = DEFAULT_STOPWORDS):
        self.top_k = top_k
        self.stopwords = frozenset(stopwords)
        self._index: dict[str, tuple[set[str], set[str]]] = {}

    def _tokens(self, tool: BaseTool) -> tuple[set[str], set[str]]:
        if tool.name not in self._index:
            self._index[tool.name] = (
                set(tokenize(tool.name, self.stopwords)),
                set(tokenize(tool.description or "", self.stopwords)),
            )
        return self._index[tool.name]

    def _score(self, query_tokens: list[str], tool: BaseTool) -> int:
        name_tokens, desc_tokens = self._tokens(tool)
        score = 0
        for qt in query_tokens:
            if qt in name_tokens:
                score += 3
            elif any(qt in nt or nt in qt for nt in name_tokens):
                score += 2
            elif qt in desc_tokens:
                score += 1
        return score

    def search(self, query: str, tools: Sequence[BaseTool]) -> list[BaseTool]:
        query_tokens = tokenize(query, self.stopwords)
        scored = [(s, t) for t in tools if (s := self._score(query_tokens, t)) > 0]
        # Stable sort keeps catalog order among equal scores, so results are deterministic.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [t for _, t in scored[: self.top_k]]
