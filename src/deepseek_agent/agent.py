"""ReAct agent loop.

Implements the Thought -> Action -> Observation cycle from the ReAct paper
(Yao et al., 2022) on top of any backend exposing ``chat(messages) -> str``
(the real DeepSeek API or the offline MockLLM). The LLM either calls one
tool per step or finishes with ``Final Answer:``. A step budget guarantees
the loop always terminates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .memory import ConversationMemory
from .tools import Tool, default_tools

SYSTEM_TEMPLATE = """你是一个会使用工具的智能助手（Agent）。你可以调用以下工具：

{tool_lines}

请严格按 ReAct 格式输出，每次只走一步：
Thought: <你的思考>
Action: <工具名>
Action Input: <工具输入>

当信息足够时，直接输出：
Final Answer: <最终回答>
"""


@dataclass
class Step:
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)


class ReActAgent:
    def __init__(self, llm, tools: dict[str, Tool] | None = None,
                 memory: ConversationMemory | None = None, max_steps: int = 6):
        self.llm = llm
        self.tools = tools if tools is not None else default_tools()
        self.memory = memory if memory is not None else ConversationMemory()
        self.max_steps = max_steps

    # ------------------------------------------------------------------ run
    def run(self, question: str, verbose: bool = False) -> AgentResult:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(tool_lines=tool_lines)}]
        messages += self.memory.as_messages()
        messages.append({"role": "user", "content": question})

        steps: list[Step] = []
        scratchpad = question
        answer = ""
        for _ in range(self.max_steps):
            reply = self.llm.chat(messages + [{"role": "user", "content": scratchpad}])
            if verbose:
                print(reply)
            final = re.search(r"Final Answer:\s*(.+)", reply, re.S)
            if final:
                answer = final.group(1).strip()
                break
            action = re.search(r"Action:\s*(\w+)", reply)
            action_input = re.search(r"Action Input:\s*(.+)", reply)
            if not action:
                answer = reply.strip()  # model answered without the protocol
                break
            name = action.group(1)
            arg = action_input.group(1).strip() if action_input else ""
            tool = self.tools.get(name)
            observation = tool(arg) if tool else f"Error: 未知工具 {name}，可用工具: {list(self.tools)}"
            if verbose:
                print(f"Observation: {observation}")
            thought = re.search(r"Thought:\s*(.+)", reply)
            steps.append(Step(thought.group(1).strip() if thought else "", name, arg, observation))
            scratchpad += f"\n{reply}\nObservation: {observation}"
        else:
            answer = "已达到最大步数，未能得出最终答案。最后的观察结果：" + (
                steps[-1].observation if steps else "无")

        self.memory.add("user", question)
        self.memory.add("assistant", answer)
        return AgentResult(answer=answer, steps=steps)
