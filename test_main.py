import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from agent import Agent, AgentError, MAX_ROUNDS
from agent.runtime import MAX_CONTEXT_CHARS, MAX_SUMMARY_CHARS, MAX_TOOL_RESULT_CHARS
from agent.storage import TODO_PAGE_SIZE
from agent.tools import Tool, calculate, search


def tool_call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.contexts = []

    def __call__(self, _key, context, tools):
        self.contexts.append(context)
        if tools:
            assert {item["function"]["name"] for item in tools} == {"calculator", "search", "todo"}
        return next(self.responses)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "agent.db"
        self.agents = []

    def agent(self, model):
        agent = Agent("key", self.db, model)
        self.agents.append(agent)
        return agent

    def tearDown(self):
        for agent in self.agents:
            agent.store.db.close()
        self.temp.cleanup()

    def test_tool_loop_and_followup(self):
        model = ScriptedModel([
            {"tool_calls": [tool_call("calculator", {"expression": "2+3"})], "reasoning_content": "private"},
            {"content": "结果是 5"},
            {"content": "刚才算的是 2+3"},
        ])
        agent = self.agent(model)
        self.assertEqual(agent.run("u", "s", "算 2+3"), "结果是 5")
        self.assertEqual(json.loads(model.contexts[1][-1]["content"]), {"result": 5})
        self.assertEqual(agent.run("u", "s", "刚才算了什么？"), "刚才算的是 2+3")
        self.assertEqual(model.contexts[2][-2]["content"], "结果是 5")
        self.assertNotIn("private", str(model.contexts[2]))
        event = agent.store.db.execute(
            "SELECT detail FROM events WHERE kind='model' ORDER BY id LIMIT 1").fetchone()
        event_detail = json.loads(event[0])
        self.assertEqual(event_detail["reasoning_content"], "private")
        self.assertEqual(event_detail["tool_calls"][0]["name"], "calculator")
        self.assertEqual(json.loads(event_detail["tool_calls"][0]["arguments"]),
                         {"expression": "2+3"})

    def test_model_trace_lists_all_tool_calls_and_empty_final_answer_calls(self):
        calls = [tool_call("calculator", {"expression": "2+3"}, "calc"),
                 tool_call("search", {"query": "agent"}, "search")]
        agent = self.agent(ScriptedModel([{"tool_calls": calls}, {"content": "完成"}]))
        self.assertEqual(agent.run("u", "s", "计算并搜索"), "完成")
        events = [json.loads(row[0]) for row in agent.store.db.execute(
            "SELECT detail FROM events WHERE kind='model' ORDER BY id")]
        self.assertEqual(events[0]["tool_calls"], [
            {"id": "calc", "name": "calculator", "arguments": calls[0]["function"]["arguments"]},
            {"id": "search", "name": "search", "arguments": calls[1]["function"]["arguments"]},
        ])
        self.assertEqual(events[1]["tool_calls"], [])

    def test_session_and_user_isolation_survive_reopen(self):
        agent = self.agent(ScriptedModel([]))
        agent.store.todo({"action": "add", "text": "窗口一"}, "u", "one")
        agent.store.todo({"action": "add", "text": "另一个用户"}, "v", "one")
        agent.store.save("u", "one", "", [{"role": "user", "content": "私有历史"}])
        agent.store.db.close()
        model = ScriptedModel([{"content": "继续记录"}])
        reopened = self.agent(model)
        self.assertEqual(reopened.store.todo({"action": "list"}, "u", "one")["items"][0]["text"], "窗口一")
        self.assertEqual(reopened.run("u", "one", "继续"), "继续记录")
        self.assertIn("私有历史", str(model.contexts[0]))
        self.assertEqual(reopened.store.todo({"action": "list"}, "u", "two")["items"], [])
        with self.assertRaisesRegex(ValueError, "没有这个待办"):
            reopened.store.todo({"action": "complete", "id": 1}, "u", "two")

    def test_todo_add_list_complete_lifecycle(self):
        agent = self.agent(ScriptedModel([]))
        added = agent.store.todo({"action": "add", "text": "提交面试题"}, "u", "s")
        self.assertEqual(added["done"], False)
        self.assertEqual(agent.store.todo({"action": "list"}, "u", "s")["items"],
                         [{"id": added["id"], "text": "提交面试题", "done": False}])
        self.assertEqual(agent.store.todo({"action": "complete", "id": added["id"]}, "u", "s"),
                         {"id": added["id"], "done": True})
        self.assertTrue(agent.store.todo({"action": "list"}, "u", "s")["items"][0]["done"])

    def test_todo_session_numbers_work_with_existing_global_ids(self):
        with closing(sqlite3.connect(self.db)) as db:
            db.executescript("""
                CREATE TABLE todos (
                    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    text TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0
                );
                INSERT INTO todos VALUES (1, 'demo', 'smoke', '示例一', 0);
                INSERT INTO todos VALUES (2, 'demo', 'smoke2', '示例二', 0);
                INSERT INTO todos VALUES (3, 'local', 'default', '健身', 0);
                INSERT INTO todos VALUES (4, 'local', 'default', '确认计算结果 8', 0);
            """)
        model = ScriptedModel([
            {"tool_calls": [tool_call("todo", {"action": "list"}, "list_now")]},
            {"tool_calls": [tool_call("todo", {"action": "complete", "id": 2}, "complete")]},
            {"content": "已完成确认计算结果 8"},
        ])
        agent = self.agent(model)
        agent.run("local", "default", "完成确认计算结果 8")
        self.assertEqual(json.loads(model.contexts[1][-1]["content"])["items"], [
            {"id": 1, "text": "健身", "done": False},
            {"id": 2, "text": "确认计算结果 8", "done": False},
        ])
        self.assertEqual(json.loads(model.contexts[2][-1]["content"]), {"id": 2, "done": True})
        self.assertEqual(agent.store.db.execute("SELECT id, done FROM todos ORDER BY id").fetchall(),
                         [(1, 0), (2, 0), (3, 0), (4, 1)])
        for user, session in (("other", "default"), ("local", "other")):
            self.assertEqual(agent.store.todo({"action": "add", "text": "独立待办"}, user, session)["id"], 1)
        agent.store.db.close()
        reopened = self.agent(ScriptedModel([]))
        self.assertEqual(reopened.store.todo({"action": "list"}, "local", "default")["items"], [
            {"id": 1, "text": "健身", "done": False},
            {"id": 2, "text": "确认计算结果 8", "done": True},
        ])
        self.assertEqual(reopened.store.todo({"action": "add", "text": "新增"}, "local", "default")["id"], 3)
        for item_id in (0, -1, True, "1", 1.5, 4):
            with self.subTest(item_id=item_id), self.assertRaises(ValueError):
                reopened.store.todo({"action": "complete", "id": item_id}, "local", "default")

    def test_invalid_tool_arguments_return_error_to_model(self):
        model = ScriptedModel([
            {"tool_calls": [tool_call("todo", {"action": "add"})]},
            {"tool_calls": [{"id": "call_2", "type": "function", "function": {
                "name": "todo", "arguments": "{broken"}}]},
            {"content": "请提供待办内容"},
        ])
        agent = self.agent(model)
        self.assertEqual(agent.run("u", "s", "添加待办"), "请提供待办内容")
        self.assertIn("error", json.loads(model.contexts[1][-1]["content"]))
        self.assertIn("error", json.loads(model.contexts[2][-1]["content"]))
        self.assertEqual(agent.store.todo({"action": "list"}, "u", "s")["items"], [])

    def test_context_compaction_keeps_recent_turns(self):
        model = ScriptedModel([
            *[{"content": str(i)} for i in range(4)],
            {"content": "问题0的摘要"}, {"content": "4"},
            {"content": "问题0和问题1的摘要"}, {"content": "5"},
        ])
        agent = self.agent(model)
        for i in range(6):
            agent.run("u", "s", f"问题{i}")
        summary, raw = agent.store.db.execute("SELECT summary, messages FROM sessions").fetchone()
        self.assertIn("问题0", summary)
        self.assertNotIn("问题0", raw)
        self.assertIn("问题5", raw)
        first = json.loads(model.contexts[4][1]["content"])
        self.assertEqual(first["previous_summary"], "")
        self.assertEqual(first["messages"][0]["content"], "问题0")
        second = json.loads(model.contexts[6][1]["content"])
        self.assertEqual(second["previous_summary"], "问题0的摘要")
        self.assertIn("问题0的摘要", model.contexts[5][0]["content"])
        self.assertEqual([m["content"] for m in model.contexts[7] if m["role"] == "user"],
                         ["问题2", "问题3", "问题4", "问题5"])

    def test_search_result_enters_agent_loop(self):
        model = ScriptedModel([
            {"tool_calls": [tool_call("search", {"query": "工具调用"})]},
            {"content": "工具通过 Schema 选择并执行。"},
        ])
        agent = self.agent(model)
        self.assertEqual(agent.run("u", "s", "工具是怎么调用的？"), "工具通过 Schema 选择并执行。")
        search_result = json.loads(model.contexts[1][-1]["content"])
        self.assertEqual(search_result["source"], "本地示例资料（非实时网页）")
        self.assertEqual(search_result["hits"][0]["title"], "工具调用")

    def test_summary_preserves_constraints_and_database_todos_for_followup(self):
        original = "不要出现客户名字"
        model = ScriptedModel([
            {"content": "要求：不要出现客户名字；任务：继续周报；待办以查询为准。",
             "reasoning_content": "not-for-context"},
            {"tool_calls": [tool_call("todo", {"action": "list"}, "list_now")]},
            {"content": "已按要求继续周报，待办已完成。"},
        ])
        agent = self.agent(model)
        todo = agent.store.todo({"action": "add", "text": "写周报"}, "u", "s")
        agent.store.todo({"action": "complete", "id": todo["id"]}, "u", "s")
        old = [{"role": "user", "content": "前置文字" * 150 + original},
               {"role": "assistant", "content": "", "tool_calls": [tool_call(
                   "todo", {"action": "add", "text": "写周报"})], "reasoning_content": "private"},
               {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(todo)},
               {"role": "assistant", "content": "已添加"}]
        recent = [{"role": "user", "content": f"近期{i}"} for i in range(3)]
        agent.store.save("u", "s", "原摘要", old + recent)
        agent.run("u", "s", "按之前要求继续，并查一下刚才的待办")
        data = json.loads(model.contexts[0][1]["content"])
        self.assertIn(original, str(data))
        self.assertEqual(data["previous_summary"], "原摘要")
        self.assertEqual(len(data["messages"]), len(old))
        self.assertNotIn("reasoning_content", str(data))
        self.assertIn(original, model.contexts[1][0]["content"])
        self.assertNotIn("not-for-context", str(model.contexts[1]))
        result = json.loads(model.contexts[2][-1]["content"])
        self.assertTrue(result["items"][0]["done"])
        self.assertEqual(result["items"][0]["id"], todo["id"])
        agent.store.db.close()
        reopened = self.agent(ScriptedModel([]))
        self.assertIn(original, reopened.store.load("u", "s")[0])
        self.assertTrue(reopened.store.todo({"action": "list"}, "u", "s")["items"][0]["done"])

    def test_summary_failure_keeps_original_history(self):
        invalid = [AgentError("offline"), {}, {"content": " "},
                   {"content": "x" * (MAX_SUMMARY_CHARS + 1)},
                   {"content": "摘要", "tool_calls": [tool_call("todo", {"action": "list"})]}]
        history = [{"role": "user", "content": f"问题{i}"} for i in range(4)]
        for index, response in enumerate(invalid):
            with self.subTest(response=response):
                session = str(index)
                def model(_key, _context, tools):
                    self.assertEqual(tools, [])
                    if isinstance(response, Exception):
                        raise response
                    return response
                agent = self.agent(model)
                agent.store.save("u", session, "原摘要", history)
                with self.assertRaisesRegex(AgentError, "压缩失败"):
                    agent.run("u", session, "继续")
                self.assertEqual(agent.store.load("u", session),
                                 ("原摘要", [*history, {"role": "user", "content": "继续"}]))
                events = list(agent.store.recent_events("u", session))
                self.assertEqual(events[-1][1], "summary_error")

    def test_context_size_triggers_summary_and_guards_single_turn(self):
        model = ScriptedModel([{"content": "早先要求"}, {"content": "好的"}])
        agent = self.agent(model)
        agent.store.save("u", "s", "", [{"role": "user", "content": "x" * MAX_CONTEXT_CHARS},
                                        {"role": "assistant", "content": "收到"}])
        self.assertEqual(agent.run("u", "s", "继续"), "好的")
        self.assertLessEqual(len(json.dumps(model.contexts[1], ensure_ascii=False)), MAX_CONTEXT_CHARS)
        agent.store.save("u", "large", "", [{"role": "user", "content": "x" * MAX_CONTEXT_CHARS},
                                            {"role": "tool", "content": "y" * MAX_CONTEXT_CHARS}])
        before = agent.store.load("u", "large")
        with self.assertRaisesRegex(AgentError, "当前轮上下文过长"):
            agent._context("u", "large", *before)
        self.assertEqual(agent.store.load("u", "large"), before)

    def test_todo_pagination_and_next_page_followup(self):
        model = ScriptedModel([
            {"tool_calls": [tool_call("todo", {"action": "list"})]},
            {"content": "先展示前 10 条，还有后续。"},
            {"tool_calls": [tool_call("todo", {"action": "list", "after_id": TODO_PAGE_SIZE})]},
            {"content": "这是剩余待办。"},
        ])
        agent = self.agent(model)
        for i in range(23):
            agent.store.todo({"action": "add", "text": "交错的其他会话待办"}, "u", "other")
            agent.store.todo({"action": "add", "text": f"{i}" + "长" * 190}, "u", "s")
        agent.store.todo({"action": "add", "text": "另一个窗口"}, "u", "other")
        seen, cursor = [], 0
        while True:
            page = agent.store.todo({"action": "list", "after_id": cursor}, "u", "s")
            self.assertLessEqual(len(page["items"]), TODO_PAGE_SIZE)
            seen.extend(item["id"] for item in page["items"])
            if not page["has_more"]:
                self.assertIsNone(page["next_after_id"])
                break
            self.assertIn("未展示完整", page["notice"])
            cursor = page["next_after_id"]
        self.assertEqual(seen, list(range(1, 24)))
        for cursor in (-1, True, "1", 1.5):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                agent.store.todo({"action": "list", "after_id": cursor}, "u", "s")
        agent.run("u", "s", "查看待办")
        first = json.loads(model.contexts[1][-1]["content"])
        self.assertTrue(first["has_more"])
        self.assertEqual(first["next_after_id"], TODO_PAGE_SIZE)
        agent.run("u", "s", "继续下一页")
        second = json.loads(model.contexts[3][-1]["content"])
        self.assertEqual(second["items"][0]["id"], TODO_PAGE_SIZE + 1)
        self.assertLessEqual(len(json.dumps(model.contexts[3], ensure_ascii=False)), MAX_CONTEXT_CHARS)

    def test_large_tool_results_are_bounded_and_full_trace_is_preserved(self):
        agent = self.agent(ScriptedModel([]))
        for index, text in enumerate(("x" * 20000, "\n\"\\" * 10000)):
            name = f"large_{index}"
            result = {"text": text}
            agent.register(Tool(name, "large result", {"type": "object", "properties": {}}, lambda *_: result))
            message = agent._execute("u", "s", tool_call(name, {}))
            self.assertEqual(message["tool_call_id"], "call_1")
            self.assertLessEqual(len(message["content"]), MAX_TOOL_RESULT_CHARS)
            partial = json.loads(message["content"])
            self.assertTrue(partial["truncated"])
            self.assertIn("仅展示部分", partial["notice"])
            self.assertEqual(partial["original_chars"], len(json.dumps(result, ensure_ascii=False)))
            event = list(agent.store.recent_events("u", "s"))[-1]
            self.assertEqual(json.loads(event[2])["result"], result)

    def test_round_limit_and_calculator_safety(self):
        model = ScriptedModel([{"tool_calls": [tool_call("search", {"query": "agent"}, f"c{i}")]}
                               for i in range(MAX_ROUNDS)])
        agent = self.agent(model)
        self.assertIn("最大执行轮次", agent.run("u", "s", "一直查"))
        self.assertEqual(len(model.contexts), MAX_ROUNDS)
        for expression in ("__import__('os')", "2**100", "1/0"):
            with self.assertRaises((ValueError, ZeroDivisionError)):
                calculate({"expression": expression}, "u", "s")

    def test_api_failure_preserves_user_message(self):
        def failing(_key, _context, _tools):
            raise AgentError("network unavailable")

        agent = self.agent(failing)
        with self.assertRaises(AgentError):
            agent.run("u", "s", "稍后再试")
        row = agent.store.db.execute("SELECT messages FROM sessions").fetchone()
        self.assertIn("稍后再试", row[0])

    def test_pending_tool_result_is_recovered_without_duplicate_write(self):
        call = tool_call("todo", {"action": "add", "text": "恢复任务"})
        agent = self.agent(ScriptedModel([]))
        agent.store.save("u", "s", "", [
            {"role": "user", "content": "添加恢复任务"},
            {"role": "assistant", "content": "", "tool_turn_id": "turn-1", "tool_calls": [call]},
        ])
        agent.client = ScriptedModel([{"content": "继续"}])
        self.assertEqual(agent.run("u", "s", "继续"), "继续")
        self.assertEqual(len(agent.store.todo({"action": "list"}, "u", "s")["items"]), 1)
        _, messages = agent.store.load("u", "s")
        self.assertEqual([item["role"] for item in messages[-2:]], ["user", "assistant"])
        self.assertNotIn("tool_turn_id", json.dumps(agent.client.contexts))

    def test_failed_tool_commit_rolls_back_and_recovers_after_reopen(self):
        call = tool_call("todo", {"action": "add", "text": "恢复任务"})
        for index, phase in enumerate(("trace", "ledger", "history", "interrupt")):
            with self.subTest(phase=phase):
                session = f"fault-{index}"
                agent = self.agent(ScriptedModel([{"tool_calls": [call]}]))
                original_event = agent.store.event
                original_save = agent.store.save

                def fail_event(user, current_session, kind, detail):
                    if kind == "tool":
                        if phase == "interrupt":
                            raise KeyboardInterrupt()
                        raise sqlite3.OperationalError("simulated trace failure")
                    original_event(user, current_session, kind, detail)

                def fail_save(user, current_session, summary, messages):
                    if messages[-1]["role"] == "tool":
                        raise sqlite3.OperationalError("simulated history failure")
                    original_save(user, current_session, summary, messages)

                expected = KeyboardInterrupt if phase == "interrupt" else AgentError
                with patch.object(agent.store, "event", side_effect=fail_event if phase in (
                        "trace", "interrupt") else original_event), patch.object(
                        agent.store, "save", side_effect=fail_save if phase == "history" else original_save):
                    if phase == "ledger":
                        with patch.object(agent.store, "save_tool_result", side_effect=sqlite3.OperationalError(
                                "simulated result failure")), self.assertRaises(expected):
                            agent.run("u", session, "添加恢复任务")
                    else:
                        with self.assertRaises(expected):
                            agent.run("u", session, "添加恢复任务")
                self.assertEqual(agent.store.todo({"action": "list"}, "u", session)["items"], [])
                self.assertEqual(agent.store.db.execute(
                    "SELECT COUNT(*) FROM tool_results WHERE session_id=?", (session,)).fetchone()[0], 0)
                self.assertEqual(agent.store.load("u", session)[1][-1]["role"], "assistant")
                agent.store.db.close()
                reopened = self.agent(ScriptedModel([{"content": "恢复完成"}]))
                self.assertEqual(reopened.run("u", session, "继续"), "恢复完成")
                self.assertEqual(len(reopened.store.todo({"action": "list"}, "u", session)["items"]), 1)
                self.assertEqual(reopened.client.contexts[0][-2]["role"], "tool")

    def test_partly_executed_batch_recovers_only_missing_results(self):
        calls = [tool_call("todo", {"action": "add", "text": text}, text) for text in ("first", "second")]
        agent = self.agent(ScriptedModel([{"tool_calls": calls}]))
        original = agent._execute

        def interrupt_second(user, session, call, turn_id=None):
            if call["id"] == "second":
                raise KeyboardInterrupt()
            return original(user, session, call, turn_id)

        with patch.object(agent, "_execute", side_effect=interrupt_second), self.assertRaises(KeyboardInterrupt):
            agent.run("u", "s", "添加两个待办")
        self.assertEqual(len(agent.store.todo({"action": "list"}, "u", "s")["items"]), 1)
        agent.store.db.close()
        reopened = self.agent(ScriptedModel([{"content": "已恢复"}]))
        reopened.run("u", "s", "继续")
        self.assertEqual([item["text"] for item in reopened.store.todo({"action": "list"}, "u", "s")["items"]],
                         ["first", "second"])
        replies = [message["tool_call_id"] for message in reopened.client.contexts[0] if message["role"] == "tool"]
        self.assertEqual(replies, ["first", "second"])

    def test_hard_process_exit_does_not_commit_a_partial_tool_write(self):
        script = """
import json, os, sys
from agent import Agent
call = {"id": "crash", "type": "function", "function": {
    "name": "todo", "arguments": json.dumps({"action": "add", "text": "recover me"})}}
agent = Agent("fake", sys.argv[1], lambda *_: {"tool_calls": [call]})
original = agent.store.event
def crash(user, session, kind, detail):
    if kind == "tool":
        os._exit(17)
    original(user, session, kind, detail)
agent.store.event = crash
agent.run("u", "s", "add")
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.db)],
                                cwd=Path(__file__).parent, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 17, result.stderr.decode(errors="replace"))
        agent = self.agent(ScriptedModel([{"content": "recovered"}]))
        self.assertEqual(agent.store.todo({"action": "list"}, "u", "s")["items"], [])
        self.assertEqual(agent.store.db.execute("SELECT COUNT(*) FROM tool_results").fetchone()[0], 0)
        self.assertEqual(agent.run("u", "s", "continue"), "recovered")
        self.assertEqual(len(agent.store.todo({"action": "list"}, "u", "s")["items"]), 1)

    def test_legacy_pending_write_is_not_replayed(self):
        agent = self.agent(ScriptedModel([{"content": "请确认当前状态"}]))
        call = tool_call("todo", {"action": "add", "text": "旧任务"})
        agent.store.todo({"action": "add", "text": "旧任务"}, "u", "s")
        agent.store.save("u", "s", "", [{"role": "user", "content": "添加"},
            {"role": "assistant", "content": "", "tool_calls": [call]}])
        agent.run("u", "s", "继续")
        self.assertEqual(len(agent.store.todo({"action": "list"}, "u", "s")["items"]), 1)
        self.assertIn("结果未知", agent.client.contexts[0][-2]["content"])

    def test_repeated_tool_call_id_uses_cached_result(self):
        call = tool_call("todo", {"action": "add", "text": "只添加一次"})
        model = ScriptedModel([{"tool_calls": [call]}, {"tool_calls": [call]}, {"content": "完成"}])
        agent = self.agent(model)
        self.assertEqual(agent.run("u", "s", "添加"), "完成")
        self.assertEqual(len(agent.store.todo({"action": "list"}, "u", "s")["items"]), 1)
        self.assertEqual(agent.store.db.execute("SELECT COUNT(*) FROM tool_results").fetchone()[0], 1)

    def test_call_ids_can_be_reused_across_user_turns(self):
        call = tool_call("todo", {"action": "add", "text": "相同任务"})
        agent = self.agent(ScriptedModel([{"tool_calls": [call]}, {"content": "已添加"},
                                         {"tool_calls": [call]}, {"content": "又添加一次"}]))
        agent.run("u", "s", "添加")
        agent.run("u", "s", "再添加一个一样的")
        self.assertEqual(len(agent.store.todo({"action": "list"}, "u", "s")["items"]), 2)

    def test_conflicting_reused_id_is_rejected_before_any_new_batch_write(self):
        first = tool_call("todo", {"action": "add", "text": "original"}, "same")
        conflict = tool_call("todo", {"action": "add", "text": "changed"}, "same")
        unrelated = tool_call("todo", {"action": "add", "text": "must not execute"}, "new")
        agent = self.agent(ScriptedModel([{"tool_calls": [first]}, {"tool_calls": [unrelated, conflict]}]))
        with self.assertRaisesRegex(AgentError, "ID 复用"):
            agent.run("u", "s", "添加")
        self.assertEqual([item["text"] for item in agent.store.todo({"action": "list"}, "u", "s")["items"]],
                         ["original"])
        self.assertEqual(agent.store.load("u", "s")[1][-1]["role"], "tool")

    def test_persisted_replay_is_scoped_and_large_results_stay_bounded(self):
        agent = self.agent(ScriptedModel([]))
        call = tool_call("todo", {"action": "add", "text": "same"})
        first = agent._execute("u", "s", call, "turn")
        agent.store.db.close()
        reopened = self.agent(ScriptedModel([]))
        reordered = {**call, "function": {**call["function"], "arguments": '{"text":"same","action":"add"}'}}
        self.assertEqual(reopened._execute("u", "s", reordered, "turn"), first)
        for user, session in (("u", "other"), ("other", "s")):
            reopened._execute(user, session, call, "turn")
            self.assertEqual(len(reopened.store.todo({"action": "list"}, user, session)["items"]), 1)
        self.assertEqual(len(reopened.store.todo({"action": "list"}, "u", "s")["items"]), 1)
        large = {"text": "x" * 20000}
        reopened.register(Tool("large", "large", {"properties": {}}, lambda *_: large))
        large_call = tool_call("large", {})
        first = reopened._execute("u", "s", large_call, "large-turn")
        self.assertEqual(reopened._execute("u", "s", large_call, "large-turn"), first)
        self.assertLessEqual(len(first["content"]), MAX_TOOL_RESULT_CHARS)

    def test_truncated_model_answer_is_rejected(self):
        for reason in ("length", "content_filter", "unknown"):
            with self.subTest(reason=reason):
                agent = self.agent(ScriptedModel([{"content": "半句", "finish_reason": reason}]))
                with self.assertRaisesRegex(AgentError, "截断"):
                    agent.run("u", reason, "回答")
                self.assertEqual(agent.store.load("u", reason)[1], [{"role": "user", "content": "回答"}])
                event = json.loads(list(agent.store.recent_events("u", reason))[-1][2])
                self.assertNotIn("answer", event)
                self.assertEqual(event["partial_content"], "半句")

    def test_truncated_tool_batch_does_not_write_and_summary_keeps_history(self):
        call = tool_call("todo", {"action": "add", "text": "不应执行"})
        agent = self.agent(ScriptedModel([{"tool_calls": [call], "finish_reason": "length"}]))
        with self.assertRaises(AgentError):
            agent.run("u", "s", "添加")
        self.assertEqual(agent.store.todo({"action": "list"}, "u", "s")["items"], [])
        history = [{"role": "user", "content": f"旧问题{i}"} for i in range(4)]
        agent.store.save("u", "summary", "原摘要", history)
        agent.client = ScriptedModel([{"content": "半段摘要", "finish_reason": "length"}])
        with self.assertRaisesRegex(AgentError, "压缩失败"):
            agent.run("u", "summary", "继续")
        self.assertEqual(agent.store.load("u", "summary"),
                         ("原摘要", [*history, {"role": "user", "content": "继续"}]))

    def test_invalid_model_tool_call_list(self):
        for index, calls in enumerate(({"bad": "shape"}, {}, "", False, 0)):
            with self.subTest(calls=calls):
                agent = self.agent(ScriptedModel([{"tool_calls": calls}]))
                with self.assertRaisesRegex(AgentError, "工具调用列表"):
                    agent.run("u", str(index), "测试")

    def test_malformed_tool_batch_preserves_history_and_allows_retry(self):
        valid = tool_call("todo", {"action": "add", "text": "提交周报"}, "call_2")
        malformed_calls = [None, "bad-shape", {},
                           {key: value for key, value in valid.items() if key != "function"}]
        for field, values in (("id", (None, "", " ", 1, [])),
                              ("type", (None, "", "other")),
                              ("function", (None, "bad-shape", [], {}))):
            malformed_calls.extend({**valid, field: value} for value in values)
        for field, values in (("name", (None, "", " ", 1, [])),
                              ("arguments", (None, {}, [], 1))):
            malformed_calls.extend({**valid, "function": {**valid["function"], field: value}}
                                   for value in values)
        malformed_calls.extend([
            {**valid, "function": {"name": "todo"}},
            {**valid, "function": {"arguments": "{}"}},
        ])
        valid = tool_call("todo", {"action": "add", "text": "提交周报"})
        malformed_calls.append(valid)  # Duplicate call IDs also make the batch invalid.
        for index, malformed in enumerate(malformed_calls):
            with self.subTest(call=malformed):
                session = str(index)
                model = ScriptedModel([
                    {"content": "已有回复"},
                    {"tool_calls": [valid, malformed]},
                    {"tool_calls": [valid]},
                    {"content": "已记录"},
                ])
                agent = self.agent(model)
                agent.run("u", session, "之前的问题")
                _, previous = agent.store.load("u", session)
                with self.assertRaisesRegex(AgentError, "无效工具调用"):
                    agent.run("u", session, "记待办")
                _, history = agent.store.load("u", session)
                self.assertEqual(history, [*previous, {"role": "user", "content": "记待办"}])
                self.assertEqual(agent.store.todo({"action": "list"}, "u", session)["items"], [])
                self.assertEqual(agent.run("u", session, "重试"), "已记录")
                self.assertEqual(model.contexts[2][1:-1], history)
                self.assertEqual(model.contexts[3][-1]["role"], "tool")
                self.assertEqual(model.contexts[3][-1]["tool_call_id"], valid["id"])
                self.assertEqual(len(agent.store.todo({"action": "list"}, "u", session)["items"]), 1)

    def test_search_handles_natural_queries_and_no_match(self):
        hit = search({"query": "长对话怎么压缩上下文？"}, "u", "s")
        self.assertEqual(hit["hits"][0]["title"], "会话记忆")
        self.assertIn("本地", hit["source"])
        self.assertEqual(search({"query": "今天北京天气"}, "u", "s")["hits"], [])


if __name__ == "__main__":
    unittest.main()
