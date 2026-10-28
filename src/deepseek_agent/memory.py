"""Conversation memory with optional JSON-file persistence."""

from __future__ import annotations

import json
import os


class ConversationMemory:
    """Keeps the last ``max_turns`` user/assistant turns.

    If ``path`` is given, the history is loaded from / saved to that JSON
    file, so a CLI session can continue where the previous one stopped.
    """

    def __init__(self, path: str | None = None, max_turns: int = 20):
        self.path = path
        self.max_turns = max_turns
        self._messages: list[dict] = []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self._messages = json.load(f)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self._messages = self._messages[-2 * self.max_turns:]
        if self.path:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._messages, f, ensure_ascii=False, indent=2)

    def as_messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
