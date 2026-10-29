"""A placeholder tool catalog for a store-management API.

34 entities x 5 actions = 170 tools named ``<entity>_<action>``. Every tool returns
deterministic mock JSON, so examples and the benchmark run without a backend.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.tools import BaseTool, StructuredTool

ENTITIES = [
    "product", "category", "brand", "supplier", "purchase_order", "inventory",
    "warehouse", "stock_transfer", "customer", "customer_group", "loyalty_account",
    "order", "order_item", "invoice", "payment", "refund", "return_request",
    "shipment", "carrier", "coupon", "promotion", "price_list", "tax_rate", "store",
    "employee", "shift", "cash_register", "gift_card", "review", "wishlist", "cart",
    "notification", "report", "webhook",
]  # fmt: skip

ACTIONS = ["list", "get", "create", "update", "delete"]


def _label(entity: str) -> str:
    return entity.replace("_", " ")


def _record(entity: str, record_id: str, name: str | None = None) -> dict:
    return {"id": record_id, "type": entity, "name": name or f"{_label(entity)} {record_id}"}


def _make_tools(entity: str) -> list[BaseTool]:
    label = _label(entity)

    def list_(query: str = "", limit: int = 10, page: int = 1) -> str:
        items = [_record(entity, f"{entity[:3]}-{i}") for i in range(1, 3)]
        return json.dumps({"items": items, "page": page, "limit": limit, "query": query})

    def get(id: str) -> str:
        return json.dumps(_record(entity, id))

    def create(name: str, notes: str = "") -> str:
        return json.dumps({**_record(entity, f"{entity[:3]}-new", name), "notes": notes})

    def update(id: str, name: str = "", notes: str = "") -> str:
        return json.dumps({**_record(entity, id, name or None), "notes": notes, "updated": True})

    def delete(id: str, reason: str = "") -> str:
        return json.dumps({"id": id, "type": entity, "deleted": True, "reason": reason})

    specs = {
        "list": (list_, f"List {label} records, with optional text search and paging."),
        "get": (get, f"Get one {label} by its id."),
        "create": (create, f"Create a new {label}. Returns the created record."),
        "update": (update, f"Update the name or notes of an existing {label} by id."),
        "delete": (delete, f"Delete a {label} by id, with an optional reason."),
    }
    return [
        StructuredTool.from_function(func=fn, name=f"{entity}_{action}", description=desc)
        for action, (fn, desc) in specs.items()
    ]


def build_catalog(entities: Sequence[str] | None = None) -> list[BaseTool]:
    """Return the catalog, optionally limited to some entities."""
    return [tool for entity in (entities or ENTITIES) for tool in _make_tools(entity)]
