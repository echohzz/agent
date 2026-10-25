"""SQLite storage for sessions, todos, and execution events."""

import json
import sqlite3
import uuid
from contextlib import contextmanager


TODO_PAGE_SIZE = 10


class Store:
    def __init__(self, db_path):
        self.db = sqlite3.connect(db_path, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                summary TEXT NOT NULL, messages TEXT NOT NULL,
                PRIMARY KEY (user_id, session_id)
            );
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                text TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                kind TEXT NOT NULL, detail TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS tool_results (
                user_id TEXT NOT NULL, session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                call_id TEXT NOT NULL, signature TEXT NOT NULL,
                result TEXT NOT NULL,
                PRIMARY KEY (user_id, session_id, turn_id, call_id)
            );
        """)

    @contextmanager
    def transaction(self):
        # Savepoints keep nested todo/event writes inside the caller's transaction.
        name = f"agent_write_{uuid.uuid4().hex}"
        self.db.execute(f"SAVEPOINT {name}")
        try:
            yield
            self.db.execute(f"RELEASE {name}")
        except BaseException:
            self.db.execute(f"ROLLBACK TO {name}")
            self.db.execute(f"RELEASE {name}")
            raise

    def load(self, user, session):
        row = self.db.execute("SELECT summary, messages FROM sessions WHERE user_id=? AND session_id=?",
                              (user, session)).fetchone()
        return (row[0], json.loads(row[1])) if row else ("", [])

    def save(self, user, session, summary, messages):
        with self.transaction():
            self.db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, session_id) DO UPDATE SET summary=excluded.summary, messages=excluded.messages",
                (user, session, summary, json.dumps(messages, ensure_ascii=False)))

    def event(self, user, session, kind, detail):
        with self.transaction():
            self.db.execute("INSERT INTO events(user_id, session_id, kind, detail) VALUES (?, ?, ?, ?)",
                            (user, session, kind, json.dumps(detail, ensure_ascii=False)))

    def recent_events(self, user, session):
        rows = self.db.execute(
            "SELECT created_at, kind, detail FROM events WHERE user_id=? AND session_id=? "
            "ORDER BY id DESC LIMIT 20", (user, session)).fetchall()
        return reversed(rows)

    def tool_result(self, user, session, turn_id, call_id):
        row = self.db.execute(
            "SELECT signature, result FROM tool_results "
            "WHERE user_id=? AND session_id=? AND turn_id=? AND call_id=?",
            (user, session, turn_id, call_id)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def save_tool_result(self, user, session, turn_id, call_id, signature, result):
        with self.transaction():
            self.db.execute(
                "INSERT INTO tool_results VALUES (?, ?, ?, ?, ?, ?)",
                (user, session, turn_id, call_id, signature, json.dumps(result, ensure_ascii=False)))

    def todo(self, args, user, session):
        action = args["action"]
        if action == "add":
            text = args.get("text", "").strip()
            if not text or len(text) > 200:
                raise ValueError("待办内容必须为 1 至 200 字")
            with self.transaction():
                self.db.execute(
                    "INSERT INTO todos(user_id, session_id, text) VALUES (?, ?, ?)",
                    (user, session, text))
                item_id = self.db.execute(
                    "SELECT COUNT(*) FROM todos WHERE user_id=? AND session_id=?",
                    (user, session)).fetchone()[0]
            return {"id": item_id, "text": text, "done": False}
        if action == "complete":
            item_id = args.get("id")
            if type(item_id) is not int or item_id <= 0:
                raise ValueError("完成待办需要正整数 id")
            with self.transaction():
                cursor = self.db.execute(
                    "UPDATE todos SET done=1 WHERE id=(SELECT id FROM todos "
                    "WHERE user_id=? AND session_id=? ORDER BY id LIMIT 1 OFFSET ?)",
                    (user, session, item_id - 1))
            if not cursor.rowcount:
                raise ValueError("当前会话中没有这个待办")
            return {"id": item_id, "done": True}
        after_id = args.get("after_id", 0)
        if type(after_id) is not int or after_id < 0:
            raise ValueError("分页游标 after_id 必须是非负整数")
        # shortcut: ordinal IDs assume no deletion; persist a session ID before adding delete support.
        rows = self.db.execute(
            "SELECT item_id, text, done FROM ("
            "SELECT ROW_NUMBER() OVER (ORDER BY id) AS item_id, text, done "
            "FROM todos WHERE user_id=? AND session_id=?) WHERE item_id>? ORDER BY item_id LIMIT ?",
            (user, session, after_id, TODO_PAGE_SIZE + 1)).fetchall()
        has_more = len(rows) > TODO_PAGE_SIZE
        rows = rows[:TODO_PAGE_SIZE]
        result = {"items": [{"id": row[0], "text": row[1], "done": bool(row[2])} for row in rows],
                  "has_more": has_more, "next_after_id": rows[-1][0] if has_more else None}
        if has_more:
            result["notice"] = "结果未展示完整，请用 next_after_id 作为 after_id 继续查询。"
        return result
