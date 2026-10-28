"""Command line entry point.

Examples
--------
Offline demo (no API key needed)::

    python -m deepseek_agent --mock "请计算 (12+8) * 3.5 等于多少？"

Real DeepSeek model::

    export DEEPSEEK_API_KEY=sk-...
    python -m deepseek_agent "现在北京时间是几点？"
"""

from __future__ import annotations

import argparse

from .agent import ReActAgent
from .llm import DeepSeekClient, MockLLM
from .memory import ConversationMemory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepseek-agent", description="DeepSeek ReAct Agent")
    parser.add_argument("question", help="要向 Agent 提问的问题")
    parser.add_argument("--mock", action="store_true", help="使用离线 MockLLM（无需 API Key）")
    parser.add_argument("--model", default="deepseek-chat",
                        help="DeepSeek 模型：deepseek-chat 或 deepseek-reasoner")
    parser.add_argument("--memory", default=None, help="对话记忆 JSON 文件路径（可选）")
    parser.add_argument("--verbose", action="store_true", help="打印完整的 Thought/Action/Observation 过程")
    args = parser.parse_args(argv)

    llm = MockLLM() if args.mock else DeepSeekClient(model=args.model)
    agent = ReActAgent(llm, memory=ConversationMemory(path=args.memory))
    result = agent.run(args.question, verbose=args.verbose)
    if not args.verbose:
        for s in result.steps:
            print(f"[工具] {s.action}({s.action_input}) -> {s.observation}")
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
