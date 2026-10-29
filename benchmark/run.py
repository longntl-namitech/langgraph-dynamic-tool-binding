"""Benchmark: bind the whole catalog vs search-then-bind.

    uv run --env-file .env --extra examples python -m benchmark.run --limit 5

Arms:
- bind_all_120: every tool of a 120-tool catalog is bound to every model call.
- search_bind_120: the same 120 tools behind DynamicToolBindingMiddleware.
- search_bind_170: the full 170-tool catalog behind the middleware. Binding all
  170 is not possible on OpenAI (128-tool limit); --probe-170 records that error.

Each task is scored automatically: every expected tool must be called with the
expected arguments. Token counts come from the model's usage metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import date
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from examples.model import make_model, model_name
from examples.store_catalog import ENTITIES, build_catalog
from langgraph_dynamic_tool_binding import DynamicToolBindingMiddleware

HERE = Path(__file__).parent
SYSTEM_PROMPT = "You manage a store through tools. Do what the user asks, then reply briefly."
ARMS = {
    "bind_all_120": (ENTITIES[:24], False),
    "search_bind_120": (ENTITIES[:24], True),
    "search_bind_170": (ENTITIES, True),
}


def load_tasks(limit: int | None) -> list[dict[str, Any]]:
    lines = (HERE / "tasks.jsonl").read_text().splitlines()
    tasks = [json.loads(line) for line in lines if line.strip()]
    return tasks[:limit] if limit else tasks


def args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(
        str(actual.get(k, "")).strip().lower() == str(v).strip().lower()
        for k, v in expected.items()
    )


def build_agent(entities: list[str], search: bool):
    catalog = build_catalog(entities)
    if search:
        return create_agent(
            make_model(),
            system_prompt=SYSTEM_PROMPT,
            middleware=[DynamicToolBindingMiddleware(catalog)],
            checkpointer=InMemorySaver(),
        )
    return create_agent(
        make_model(), tools=catalog, system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver()
    )


async def run_task(arm: str, task: dict[str, Any]) -> dict[str, Any]:
    entities, search = ARMS[arm]
    agent = build_agent(entities, search)
    config = {"configurable": {"thread_id": f"{arm}-{task['id']}"}}
    record: dict[str, Any] = {"arm": arm, "task_id": task["id"], "type": task["type"]}
    started = time.perf_counter()
    try:
        result: dict[str, Any] = {}
        for turn in task["turns"]:
            result = await agent.ainvoke({"messages": [HumanMessage(turn)]}, config)
    except Exception as exc:  # recorded, not raised: one failure must not stop the run
        record.update(error=f"{type(exc).__name__}: {str(exc)[:300]}", success=False)
        return record
    record["latency_s"] = round(time.perf_counter() - started, 2)

    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    calls = [c for m in ai_messages for c in m.tool_calls]
    usage = [m.usage_metadata or {} for m in ai_messages]
    record.update(
        input_tokens=sum(u.get("input_tokens", 0) for u in usage),
        output_tokens=sum(u.get("output_tokens", 0) for u in usage),
        llm_calls=len(ai_messages),
        tool_calls=[c["name"] for c in calls],
        success=all(
            any(c["name"] == e["tool"] and args_match(c["args"], e["args"]) for c in calls)
            for e in task["expected"]
        ),
    )
    if search:
        selected = set(result.get("selected_tool_names", []))
        expected_tools = {e["tool"] for e in task["expected"]}
        record["recall"] = len(expected_tools & selected) / len(expected_tools)
        record["bound_tools"] = len(selected)
        record["search_queries"] = [
            c["args"].get("query") for c in calls if c["name"] == "search_tools"
        ]
    return record


async def probe_bind_all_170() -> str:
    agent = build_agent(ENTITIES, search=False)
    config = {"configurable": {"thread_id": "probe-170"}}
    try:
        await agent.ainvoke({"messages": [HumanMessage("List the shipments.")]}, config)
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:300]}"
    return "no error"


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in ARMS:
        rows = [r for r in runs if r["arm"] == arm]
        ok = [r for r in rows if "error" not in r]
        if not rows:
            continue
        summary[arm] = {
            "tasks": len(rows),
            "errors": len(rows) - len(ok),
            "success_rate": round(sum(r["success"] for r in rows) / len(rows), 3),
            "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in ok))
            if ok
            else None,
            "total_input_tokens": sum(r["input_tokens"] for r in ok),
            "total_output_tokens": sum(r["output_tokens"] for r in ok),
            "mean_llm_calls": round(statistics.mean(r["llm_calls"] for r in ok), 2) if ok else None,
            "mean_latency_s": round(statistics.mean(r["latency_s"] for r in ok), 2) if ok else None,
        }
        recalls = [r["recall"] for r in ok if "recall" in r]
        if recalls:
            summary[arm]["mean_recall"] = round(statistics.mean(recalls), 3)
            summary[arm]["mean_bound_tools"] = round(
                statistics.mean(r["bound_tools"] for r in ok), 1
            )
    base, ours = summary.get("bind_all_120"), summary.get("search_bind_120")
    if base and ours and base["mean_input_tokens"] and ours["mean_input_tokens"]:
        summary["input_token_ratio_120"] = round(
            base["mean_input_tokens"] / ours["mean_input_tokens"], 2
        )
    return summary


def print_table(summary: dict[str, Any]) -> None:
    cols = ["tasks", "errors", "success_rate", "mean_input_tokens", "mean_llm_calls",
            "mean_latency_s", "mean_recall", "mean_bound_tools"]  # fmt: skip
    print("| arm | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for arm in ARMS:
        if arm in summary:
            print(f"| {arm} | " + " | ".join(str(summary[arm].get(c, "")) for c in cols) + " |")
    if "input_token_ratio_120" in summary:
        print(
            f"\ninput tokens, bind_all_120 / search_bind_120: {summary['input_token_ratio_120']}x"
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="run only the first N tasks")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-probe-170", action="store_true")
    parser.add_argument("--name", default=None, help="results file name")
    args = parser.parse_args()

    tasks = load_tasks(args.limit)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(arm: str, task: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            record = await run_task(arm, task)
            status = "ok" if record.get("success") else ("error" if "error" in record else "fail")
            print(f"{arm:16} {task['id']:4} {status:5} {record.get('input_tokens', '-')}")
            return record

    runs = await asyncio.gather(*(guarded(a, t) for a in args.arms for t in tasks))
    summary = summarize(runs)
    if not args.no_probe_170:
        summary["probe_bind_all_170"] = await probe_bind_all_170()

    meta = {"model": model_name(), "date": date.today().isoformat(), "limit": args.limit}
    name = args.name or f"{meta['date']}-{model_name()}-n{len(tasks)}"
    out = HERE / "results" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "runs": runs, "summary": summary}, indent=2))
    print()
    print_table(summary)
    print(f"\nprobe bind_all_170: {summary.get('probe_bind_all_170')}")
    print(f"saved {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
