import io
import json
import os
import sys
import unittest
from unittest.mock import patch

import main


class FakeDatabase:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeStore:
    def __init__(self):
        self.db = FakeDatabase()
        self.trace_requested = False

    def recent_events(self, user, session):
        self.trace_requested = (user, session)
        return iter([("now", "model", "{}")])


class FakeAgent:
    instance = None

    def __init__(self, api_key, db_path):
        self.api_key = api_key
        self.db_path = db_path
        self.store = FakeStore()
        self.messages = []
        FakeAgent.instance = self

    def run(self, user, session, text):
        self.messages.append((user, session, text))
        return "fake answer"


class CliTests(unittest.TestCase):
    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
                sys, "argv", ["main.py", "--message", "hello"]):
            with self.assertRaises(SystemExit) as error:
                main.main()
        self.assertEqual(error.exception.code, 2)

    def test_message_mode_runs_agent_and_closes_store(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "key"}, clear=True), patch.object(
                sys, "argv", ["main.py", "--db", "test.db", "--user", "u", "--session", "s",
                                "--message", "hello"]), patch.object(main, "Agent", FakeAgent):
            main.main()
        self.assertEqual(FakeAgent.instance.messages, [("u", "s", "hello")])
        self.assertTrue(FakeAgent.instance.store.db.closed)

    def test_trace_command_reads_current_session(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "key"}, clear=True), patch.object(
                sys, "argv", ["main.py", "--user", "u", "--session", "s"]), patch.object(
                main, "Agent", FakeAgent), patch("builtins.input", side_effect=["/trace", "/quit"]):
            main.main()
        self.assertEqual(FakeAgent.instance.store.trace_requested, ("u", "s"))
        self.assertTrue(FakeAgent.instance.store.db.closed)

    def test_trace_formats_multiline_text_arguments_and_nested_results(self):
        rows = [
            ("before", "summary", json.dumps({"messages": 4, "chars": 402})),
            ("now", "model", json.dumps({"user_input": "3 * (5 + 5)",
                "reasoning_content": "第一行\n第二行", "tool_calls": [
                {"id": "call_1", "name": "calculator", "arguments": "{\"expression\": \"2+3\"}"},
                {"id": "call_2", "name": "todo", "arguments": "{\"action\": \"add\", \"text\": \"记录结果\"}"},
            ]})),
            ("later", "tool", json.dumps({
                "name": "todo", "arguments": json.dumps({"action": "list"}),
                "result": {"items": [{"text": "写周报\n检查内容", "done": False}],
                           "next_after_id": None, "path": r"C:\new\notes"}})),
            ("finish", "model", json.dumps({"reasoning_content": "完成", "tool_calls": [],
                                               "answer": "已完成"})),
            ("last", "tool", json.dumps({"arguments": "{broken", "items": [], "extra": {}})),
            ("end", "legacy", "旧日志\n保留原文"),
        ]
        output = io.StringIO()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "key"}, clear=True), patch.object(
                sys, "argv", ["main.py"]), patch.object(main, "Agent", FakeAgent), patch.object(
                FakeStore, "recent_events", return_value=iter(rows)), patch(
                "builtins.input", side_effect=["/trace", "/quit"]), patch.object(sys, "stdout", output):
            main.main()
        text = output.getvalue()
        self.assertIn("[1] now | agent turn", text)
        self.assertIn("user_input:\n    3 * (5 + 5)", text)
        self.assertIn("context_summary:\n    messages:\n      4", text)
        self.assertIn("step_1 (model):\n    reasoning_content:\n      第一行\n      第二行", text)
        self.assertIn("name:\n          calculator", text)
        self.assertIn("name:\n          todo", text)
        self.assertIn("expression:\n            2+3", text)
        self.assertIn("step_2 (tool):\n    name:\n      todo", text)
        self.assertIn("text:\n            写周报\n            检查内容", text)
        self.assertIn("done:\n            false", text)
        self.assertIn("next_after_id:\n        null", text)
        self.assertIn("step_3 (model):\n    reasoning_content:\n      完成", text)
        self.assertIn("answer:\n      已完成", text)
        self.assertIn(r"C:\new\notes", text)
        self.assertIn("\n[2] last | agent turn", text)
        self.assertIn("step_1 (tool):\n    arguments:\n      {broken", text)
        self.assertIn("items:\n      []", text)
        self.assertIn("extra:\n      {}", text)
        self.assertIn("\n[3] end | agent turn", text)
        self.assertIn("step_1 (legacy):\n    旧日志\n    保留原文", text)
        self.assertTrue(FakeAgent.instance.store.db.closed)


if __name__ == "__main__":
    unittest.main()
