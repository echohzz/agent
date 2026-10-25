"""Command-line entry point for the minimal Agent."""

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import indent

from agent import Agent, AgentError


def print_trace_detail(value, padding="  "):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            if key == "arguments" and isinstance(item, str):
                try:
                    item = json.loads(item)
                except ValueError:
                    pass
            print(f"{padding}{key}:")
            print_trace_detail(item, padding + "  ")
    elif isinstance(value, list) and value:
        for index, item in enumerate(value, 1):
            print(f"{padding}[{index}]")
            print_trace_detail(item, padding + "  ")
    else:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        print(indent(text, padding))


def trace_groups(events):
    """Group contiguous events into one readable Agent turn."""
    groups = []
    pending_summary = []
    current = None

    def new_group(created_at):
        return {"started_at": created_at, "summary": [], "steps": [], "closed": False}

    for created_at, kind, raw_detail in events:
        try:
            detail = json.loads(raw_detail)
        except (TypeError, ValueError):
            detail = raw_detail
        if kind == "summary":
            if current is not None and not current["closed"]:
                current["summary"].append(detail)
            else:
                pending_summary.append(detail)
            continue
        if kind == "model":
            starts_turn = current is None or current["closed"] or (
                isinstance(detail, dict) and detail.get("user_input") is not None)
            if starts_turn:
                current = new_group(created_at)
                current["summary"].extend(pending_summary)
                pending_summary.clear()
                groups.append(current)
            if isinstance(detail, dict) and detail.get("user_input") is not None:
                current["user_input"] = detail["user_input"]
            current["steps"].append((kind, detail))
            if isinstance(detail, dict) and not detail.get("tool_calls") and detail.get("answer"):
                current["closed"] = True
            continue
        if current is None or current["closed"]:
            current = new_group(created_at)
            current["summary"].extend(pending_summary)
            pending_summary.clear()
            groups.append(current)
            current["closed"] = True
        current["steps"].append((kind, detail))
    if pending_summary:
        current = new_group("")
        current["summary"].extend(pending_summary)
        groups.append(current)
    return groups


def print_trace_group(index, group):
    print(f"\n[{index}] {group['started_at']} | agent turn")
    if group.get("user_input") is not None:
        print("  user_input:")
        print_trace_detail(group["user_input"], "    ")
    if group["summary"]:
        print("  context_summary:")
        for detail in group["summary"]:
            print_trace_detail(detail, "    ")
    for step_index, (kind, detail) in enumerate(group["steps"], 1):
        print(f"  step_{step_index} ({kind}):")
        if kind == "model" and isinstance(detail, dict) and "user_input" in detail:
            detail = {key: value for key, value in detail.items() if key != "user_input"}
        print_trace_detail(detail, "    ")


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="最小可用 Agent（百炼 kimi-k3）")
    parser.add_argument("--user", default="local", help="用户 ID")
    parser.add_argument("--session", default="default", help="会话 ID；不同窗口使用不同 ID")
    parser.add_argument("--db", type=Path, default=Path("agent.sqlite3"), help="本地会话数据库")
    parser.add_argument("--message", help="执行一次后退出；默认进入交互模式")
    args = parser.parse_args()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        parser.error("请先设置 DASHSCOPE_API_KEY 环境变量")
    agent = Agent(api_key, args.db)
    try:
        if args.message is not None:
            try:
                print(agent.run(args.user, args.session, args.message))
            except (AgentError, ValueError) as exc:
                parser.exit(1, f"错误: {exc}\n")
            return
        print(f"会话 {args.user}/{args.session}；输入 /trace 查看执行日志，/quit 退出。")
        while True:
            try:
                text = input("你> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.strip() == "/quit":
                break
            if text.strip() == "/trace":
                for index, group in enumerate(
                        trace_groups(agent.store.recent_events(args.user, args.session)), 1):
                    print_trace_group(index, group)
                continue
            try:
                print("Agent>", agent.run(args.user, args.session, text))
            except (AgentError, ValueError) as exc:
                print("错误:", exc)
    finally:
        agent.store.db.close()


if __name__ == "__main__":
    main()
