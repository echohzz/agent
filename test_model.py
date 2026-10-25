import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from agent.model import AgentError, call_model


def response(payload):
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


class ModelAdapterTests(unittest.TestCase):
    def test_parses_message_from_successful_response(self):
        payload = {"choices": [{"message": {
            "role": "assistant", "content": "好的", "reasoning_content": "先回答"}}]}
        with patch("agent.model.urllib.request.urlopen", return_value=response(payload)) as request:
            result = call_model("key", [{"role": "user", "content": "你好"}], [])
        self.assertEqual(result["content"], "好的")
        request.assert_called_once()

    def test_preserves_finish_reason(self):
        payload = {"choices": [{"finish_reason": "length", "message": {"content": "未完成"}}]}
        with patch("agent.model.urllib.request.urlopen", return_value=response(payload)):
            result = call_model("key", [], [])
        self.assertEqual(result["finish_reason"], "length")

    def test_summary_request_does_not_enable_tools(self):
        with patch("agent.model.urllib.request.urlopen", return_value=response(
                {"choices": [{"message": {"content": "摘要"}}]})) as request:
            result = call_model("key", [{"role": "user", "content": "总结旧对话"}], [])
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(result["content"], "摘要")
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_translates_http_errors(self):
        for status in (401, 429, 500):
            with self.subTest(status=status):
                error = urllib.error.HTTPError(
                    "https://example.test", status, "error", {},
                    io.BytesIO(b'{"error":{"message":"upstream failure"}}'))
                with patch("agent.model.urllib.request.urlopen", side_effect=error):
                    with self.assertRaisesRegex(AgentError, f"HTTP {status}"):
                        call_model("key", [], [])

    def test_translates_network_and_json_errors(self):
        with patch("agent.model.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("offline")):
            with self.assertRaisesRegex(AgentError, "连接失败"):
                call_model("key", [], [])
        with patch("agent.model.urllib.request.urlopen", return_value=io.BytesIO(b"{broken")):
            with self.assertRaisesRegex(AgentError, "无效 JSON"):
                call_model("key", [], [])

    def test_rejects_incomplete_api_responses(self):
        for payload in ({}, {"choices": []}, {"choices": [{"message": None}]}):
            with self.subTest(payload=payload):
                with patch("agent.model.urllib.request.urlopen", return_value=response(payload)):
                    with self.assertRaisesRegex(AgentError, "无效响应"):
                        call_model("key", [], [])


if __name__ == "__main__":
    unittest.main()
