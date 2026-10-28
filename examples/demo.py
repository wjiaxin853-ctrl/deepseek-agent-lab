"""Offline demo: run the agent on three questions that exercise three tools.

Usage:  python examples/demo.py            # offline MockLLM
        DEEPSEEK_API_KEY=sk-... python examples/demo.py --real
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deepseek_agent import DeepSeekClient, MockLLM, ReActAgent  # noqa: E402

QUESTIONS = [
    "请计算 (12+8) * 3.5 等于多少？",
    "请把 10 km to mi 换算一下",
    "现在北京时间的日期和星期是？",
]


def main() -> int:
    real = "--real" in sys.argv
    llm = DeepSeekClient() if real else MockLLM()
    agent = ReActAgent(llm)
    print(f"=== DeepSeek Agent Lab demo ({'DeepSeek API' if real else 'offline mock'}) ===")
    for q in QUESTIONS:
        print(f"\nQ: {q}")
        result = agent.run(q)
        for s in result.steps:
            print(f"  Thought: {s.thought}")
            print(f"  Action: {s.action}({s.action_input}) -> {s.observation}")
        print(f"A: {result.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
