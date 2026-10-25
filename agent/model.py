"""Bailian Chat Completions adapter."""

import json
import urllib.error
import urllib.request


MODEL = "kimi-k3"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class AgentError(Exception):
    pass


def call_model(api_key, messages, tools):
    payload = {"model": MODEL, "messages": messages}
    if tools:
        payload.update(tools=tools, tool_choice="auto")
    request = urllib.request.Request(
        API_URL, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            error = json.load(exc)
            detail = error.get("error", {}).get("message") or error.get("message", "")
        except (ValueError, AttributeError):
            detail = ""
        raise AgentError(f"百炼 API HTTP {exc.code}: {detail[:300]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AgentError(f"百炼 API 连接失败: {exc.reason if hasattr(exc, 'reason') else exc}") from exc
    except ValueError as exc:
        raise AgentError("API 返回了无效 JSON") from exc
    try:
        choice = data["choices"][0]
        message = choice["message"]
        if not isinstance(message, dict):
            raise TypeError("message must be an object")
        if "finish_reason" in choice:
            message = {**message, "finish_reason": choice["finish_reason"]}
        return message
    except (KeyError, IndexError, TypeError) as exc:
        raise AgentError("API 返回了无效响应") from exc
