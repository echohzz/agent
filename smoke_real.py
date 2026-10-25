"""Opt-in real Bailian acceptance tests. Never touches the user's database."""

import argparse
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent import Agent
from agent.model import MODEL, call_model


class RealAgentTests(unittest.TestCase):
    records = []

    def setUp(self):
        self.started = time.perf_counter()
        self.conversations = []
        self.api_calls = 0
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "acceptance.sqlite3"
        self.agent = Agent(os.environ["DASHSCOPE_API_KEY"], self.db_path, self.client)
        self.addCleanup(lambda: self.agent.store.db.close())

    def client(self, key, messages, tools):
        self.api_calls += 1
        return call_model(key, messages, tools)

    def ask(self, text, user="demo", session="main"):
        entry = {"user": user, "session": session, "input": text}
        self.conversations.append(entry)
        answer = self.agent.run(user, session, text)
        entry["answer"] = answer
        self.assertNotIn("最大执行轮次", answer)
        return answer

    def tools(self):
        return [json.loads(row[0]) for row in self.agent.store.db.execute(
            "SELECT detail FROM events WHERE kind='tool' ORDER BY id")]

    def todos(self, user="demo", session="main"):
        return self.agent.store.todo({"action": "list"}, user, session)["items"]

    def tearDown(self):
        self.records.append({
            "case": self._testMethodName,
            "checks": self.shortDescription(),
            "seconds": round(time.perf_counter() - self.started, 2),
            "api_calls": self.api_calls,
            "conversations": self.conversations,
            "tool_events": self.tools(),
        })

    def test_01_direct_answer(self):
        """Greeting produces a nonempty answer without invoking tools."""
        self.assertTrue(self.ask("你好，请用一句话打个招呼。").strip())
        self.assertEqual(self.tools(), [])

    def test_02_calculator(self):
        """Arithmetic uses calculator and its executed result equals 42."""
        answer = self.ask("请计算 17 * 3 - 9。")
        self.assertTrue(any(event["name"] == "calculator" and event["result"].get("result") == 42
                            for event in self.tools()))
        self.assertIn("42", answer)
        self.assertEqual(self.todos(), [])

    def test_03_calculate_then_add(self):
        """Calculator result 8 precedes one todo containing that result."""
        self.ask("请先计算 (18+6)/3，再把计算结果写成一条待办，格式为：核对结果：计算出的数字。")
        events = self.tools()
        names = [event["name"] for event in events]
        self.assertIn("calculator", names)
        self.assertIn("todo", names)
        self.assertLess(names.index("calculator"), names.index("todo"))
        self.assertEqual(next(event for event in events if event["name"] == "calculator")["result"]["result"], 8)
        self.assertEqual(len(self.todos()), 1)
        self.assertIn("8", self.todos()[0]["text"])

    def test_04_missing_todo_text(self):
        """Missing todo content causes a clarification and no successful write."""
        answer = self.ask("帮我添加一条待办，内容稍后再告诉你，先不要替我编内容。")
        self.assertEqual(self.todos(), [])
        self.assertFalse(any(event["name"] == "todo" and "error" not in event["result"]
                             and json.loads(event["arguments"]).get("action") == "add"
                             for event in self.tools()))
        self.assertRegex(answer, "内容|什么|具体|告诉")

    def test_05_tool_failure_retry(self):
        """After one injected calculator failure, the real model retries and gets 25."""
        original = self.agent.tools["calculator"]
        attempts = []

        def fail_once(args, user, session):
            attempts.append(args)
            if len(attempts) == 1:
                raise ValueError("临时计算失败，请使用新的调用 ID 重试一次")
            return original.handler(args, user, session)

        self.agent.tools["calculator"] = replace(original, handler=fail_once)
        answer = self.ask("请计算 5*5；如果工具提示临时失败，换一个新的调用 ID 重试一次。")
        results = [event["result"] for event in self.tools() if event["name"] == "calculator"]
        self.assertGreaterEqual(len(attempts), 2)
        self.assertIn("error", results[0])
        self.assertEqual(results[-1].get("result"), 25)
        self.assertIn("25", answer)

    def test_06_followup_completes_todo(self):
        """A natural follow-up completes exactly the todo added in the preceding turn."""
        self.ask("请添加一条待办：整理面试笔记。")
        self.assertEqual(len(self.todos()), 1)
        item_id = self.todos()[0]["id"]
        self.ask("请把刚才那条待办标记完成。")
        self.assertEqual(len(self.todos()), 1)
        self.assertEqual(self.todos()[0]["id"], item_id)
        self.assertTrue(self.todos()[0]["done"])
        self.assertTrue(any(event["name"] == "todo" and event["result"].get("done") is True
                            for event in self.tools()))

    def test_07_user_and_session_isolation(self):
        """Other user and session list empty todos and never echo the private marker."""
        self.ask("请添加待办：PRIVATE-4827。")
        self.assertEqual(len(self.todos()), 1)
        for user, session in (("demo", "other"), ("other-user", "main")):
            answer = self.ask("请查询当前会话的待办。", user, session)
            self.assertEqual(self.todos(user, session), [])
            self.assertNotIn("PRIVATE-4827", answer)
            events = [json.loads(row[0]) for row in self.agent.store.db.execute(
                "SELECT detail FROM events WHERE kind='tool' AND user_id=? AND session_id=?",
                (user, session))]
            self.assertTrue(any(event["name"] == "todo" and event["result"].get("items") == []
                                for event in events))

    def test_08_restart_persistence(self):
        """Reopening SQLite retains the original todo ID and the model queries it."""
        self.ask("请添加待办：重启后检查。")
        self.assertEqual(len(self.todos()), 1)
        item_id = self.todos()[0]["id"]
        self.agent.store.db.close()
        self.agent = Agent(os.environ["DASHSCOPE_API_KEY"], self.db_path, self.client)
        answer = self.ask("请查询我的待办。")
        self.assertEqual(self.todos()[0]["id"], item_id)
        self.assertIn("重启后检查", answer)
        self.assertTrue(any(event["result"].get("items") for event in self.tools()
                            if event["name"] == "todo"))

    def test_09_summary_followup(self):
        """After actual fifth-turn compaction, summary and answer retain BLUE-731."""
        self.ask("记住本次项目代号是 BLUE-731，后面我会问你。现在只回复收到。")
        for text in ("交付物是技术说明。", "阅读对象是研发同事。", "说明应保持简短。"):
            self.ask(text + "现在只回复收到。")
        answer = self.ask("本次项目代号是什么？请简短回答。")
        summary, _ = self.agent.store.load("demo", "main")
        self.assertIn("BLUE-731", summary)
        self.assertIn("BLUE-731", answer)
        self.assertGreater(self.agent.store.db.execute(
            "SELECT COUNT(*) FROM events WHERE kind='summary'").fetchone()[0], 0)

    def test_10_search_scope(self):
        """Search cites local material; a live-weather request states the offline limitation."""
        answer = self.ask("请查询本地资料，解释 Agent 的工具调用是怎么执行的。")
        hits = [event["result"] for event in self.tools() if event["name"] == "search"]
        self.assertTrue(any("本地" in hit.get("source", "") and hit.get("hits") for hit in hits))
        self.assertIn("本地", answer)
        weather = self.ask("请联网查询今天上海的实时天气。")
        self.assertRegex(weather, "不能|无法|不支持|不具备|没有.*(联网|实时)")
        self.assertRegex(weather, "联网|实时|网页|网络")
        self.assertEqual(self.todos(), [])


def main():
    parser = argparse.ArgumentParser(description="真实百炼 Agent 验收（会产生 API 费用）")
    parser.add_argument("--report", type=Path, default=Path("smoke-report.json"), help="JSON 验收报告")
    names = unittest.defaultTestLoader.getTestCaseNames(RealAgentTests)
    parser.add_argument("--case", nargs="+", choices=names, help="只运行指定用例，用于补测")
    args = parser.parse_args()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        parser.error("请先设置 DASHSCOPE_API_KEY")
    RealAgentTests.records.clear()
    suite = unittest.TestSuite(RealAgentTests(name) for name in (args.case or names))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed = {test._testMethodName for test, _ in result.failures + result.errors}
    for record in RealAgentTests.records:
        record["passed"] = record["case"] not in failed
    report = {
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "model": MODEL,
        "total": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "cases": RealAgentTests.records,
        "failures": [{"case": test._testMethodName, "detail": detail}
                     for test, detail in result.failures + result.errors],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"验收报告：{args.report.resolve()}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
