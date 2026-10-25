"""Offline checks for acceptance selection and truthful report generation."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import smoke_real
from agent import AgentError


class AcceptanceRunnerTests(unittest.TestCase):
    def test_selected_case_writes_success_or_failure_without_real_api(self):
        for should_fail in (False, True):
            with self.subTest(should_fail=should_fail), tempfile.TemporaryDirectory() as folder:
                report_path = Path(folder) / "report.json"
                argv = ["smoke_real.py", "--case", "test_01_direct_answer", "--report", str(report_path)]
                response = {"content": "你好", "finish_reason": "stop"}
                with (
                    patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-runner-key"}),
                    patch.object(sys, "argv", argv),
                    patch("smoke_real.call_model", return_value=response,
                          side_effect=AgentError("offline") if should_fail else None) as client,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    code = smoke_real.main()
                self.assertEqual(code, int(should_fail))
                client.assert_called_once()
                raw = report_path.read_text(encoding="utf-8")
                report = json.loads(raw)
                self.assertEqual(report["total"], 1)
                self.assertEqual(report["passed"], int(not should_fail))
                self.assertEqual(report["cases"][0]["passed"], not should_fail)
                self.assertEqual(len(report["failures"]), int(should_fail))
                self.assertNotIn("fake-runner-key", raw)

    def test_missing_key_stops_before_api_calls(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["smoke_real.py"]), patch(
                "smoke_real.call_model") as client, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                smoke_real.main()
        self.assertEqual(error.exception.code, 2)
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
