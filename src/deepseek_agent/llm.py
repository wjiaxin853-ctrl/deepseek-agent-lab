"""LLM backends for DeepSeek Agent Lab.

Two backends share one interface (`chat(messages) -> str`):

* ``DeepSeekClient`` - talks to the official DeepSeek API, which is
  OpenAI-compatible (POST https://api.deepseek.com/chat/completions).
  Implemented with the standard library only, so there is nothing to
  install. Set the ``DEEPSEEK_API_KEY`` environment variable to use it.
* ``MockLLM`` - a deterministic, fully offline stand-in used by the demo
  and the test-suite. It follows the same ReAct protocol, so the whole
  agent loop can be exercised (and CI-tested) without an API key.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"  # or "deepseek-reasoner" for DeepSeek-R1


class DeepSeekClient:
    """Minimal client for the DeepSeek chat-completions API."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 base_url: str = DEEPSEEK_BASE_URL, timeout: int = 60):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing API key: set DEEPSEEK_API_KEY or use --mock for the offline demo."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


class MockLLM:
    """Deterministic offline LLM that speaks the ReAct protocol.

    Rules (enough to demonstrate tool use end-to-end):
    * a question containing an arithmetic expression -> use `calculator`
    * a question about the current date/time        -> use `now`
    * a question asking for a unit conversion        -> use `unit_convert`
    * anything else                                  -> answer directly
    Once an Observation is present it wraps it into a Final Answer.
    """

    _EXPR = re.compile(r"\(?\d+(?:\.\d+)?(?:\s*[-+*/]\s*\(?\d+(?:\.\d+)?\)?)+")
    _CONVERT = re.compile(r"(\d+(?:\.\d+)?)\s*(km|mi|kg|lb|°?C|°?F)\b.*?to\s*(km|mi|kg|lb|°?C|°?F)", re.I)

    def chat(self, messages: list[dict]) -> str:
        history = "\n".join(m.get("content", "") for m in messages)
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        obs = re.findall(r"Observation:\s*(.+)", history)
        if obs:
            return f"Final Answer: 根据工具返回的结果：{obs[-1]}"

        question = last_user
        m = self._CONVERT.search(question)
        if m:
            return (f"Thought: 这是单位换算问题，调用 unit_convert 工具。\n"
                    f"Action: unit_convert\n"
                    f"Action Input: {m.group(1)} {m.group(2)} to {m.group(3)}")
        if any(k in question for k in ("几点", "日期", "星期", "今天", "now", "time", "date")):
            return ("Thought: 需要获取当前时间，调用 now 工具。\n"
                    "Action: now\nAction Input: Asia/Shanghai")
        m = self._EXPR.search(question)
        if m and any(ch.isdigit() for ch in question):
            expr = m.group(0).strip()
            return (f"Thought: 问题包含算术表达式 {expr}，调用 calculator 工具计算。\n"
                    f"Action: calculator\nAction Input: {expr}")
        return ("Final Answer: 这是一个离线演示回答。设置 DEEPSEEK_API_KEY 后，"
                "Agent 会调用真实的 DeepSeek 模型进行推理，并可按需使用工具。")
