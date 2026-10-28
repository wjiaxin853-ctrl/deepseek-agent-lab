# DeepSeek Agent Lab

> 基于 **DeepSeek** 大模型的轻量级 **ReAct Agent 框架** —— 零第三方依赖、开箱即跑、自带离线演示与完整单元测试。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)](#-测试)

在调研了 GitHub 上的主流开源 Agent 方案（LangChain / LangGraph 的图式编排、AutoGen 的多智能体对话、Agently 的事件驱动流、PySpur 的可视化工作流）之后，本项目选择**从零实现一条完整、可解释的 Agent 链路**：自己实现 ReAct 推理循环、工具注册与安全执行、对话记忆持久化，并通过 DeepSeek 官方的 OpenAI 兼容接口接入 `deepseek-chat` / `deepseek-reasoner` 模型。这样每一层都看得见、改得动，适合学习 Agent 原理与二次开发。

## ✨ 功能特性

- **ReAct 推理循环**：Thought → Action → Observation，带最大步数保护，保证必然终止
- **DeepSeek 原生接入**：官方 API（OpenAI 兼容协议），仅用 Python 标准库实现，无需安装任何包
- **安全的工具系统**：计算器基于 `ast` 白名单求值，杜绝 LLM 生成代码被直接 `eval` 执行的风险
- **内置工具**：安全计算器、时区时钟、单位换算（长度/质量/温度）、文本统计；工具注册只需一个函数 + 一行描述
- **对话记忆**：多轮上下文保持，可选 JSON 文件持久化，CLI 重启可续聊
- **离线模式**：内置确定性 `MockLLM`，无 API Key 也能完整跑通 Agent 链路，测试与 CI 零成本
- **完整测试**：10 个单元测试覆盖工具安全性、换算正确性与 Agent 决策路径

## 🏗️ 架构

```
用户问题
   │
   ▼
┌─────────────┐   Thought / Action    ┌──────────────┐
│  ReActAgent │ ────────────────────▶ │  LLM Backend │
│   (agent.py)│ ◀──────────────────── │ DeepSeek API │
└──────┬──────┘     Final Answer      │  或 MockLLM  │
       │ Action Input                 └──────────────┘
       ▼
┌─────────────┐   Observation   ┌──────────────────┐
│ Tool 注册表 │ ──────────────▶ │ calculator / now │
│  (tools.py) │                 │ unit_convert ... │
└─────────────┘                 └──────────────────┘
       │
       ▼
ConversationMemory（memory.py，可选 JSON 持久化）
```

| 模块 | 职责 |
|---|---|
| `llm.py` | DeepSeek API 客户端（标准库实现）+ 离线 MockLLM，两者接口一致可互换 |
| `agent.py` | ReAct 循环：解析模型输出、调度工具、组装 Observation、步数保护 |
| `tools.py` | 工具注册表与 4 个内置工具；计算器用 AST 白名单防代码注入 |
| `memory.py` | 多轮对话记忆，支持 JSON 文件持久化与截断 |
| `cli.py` | 命令行入口，支持 `--mock / --model / --memory / --verbose` |

## 🚀 快速开始

```bash
git clone https://github.com/wjiaxin853-ctrl/deepseek-agent-lab.git
cd deepseek-agent-lab

# 1) 离线演示（不需要 API Key，不需要 pip install 任何东西）
python examples/demo.py
PYTHONPATH=src python -m deepseek_agent --mock "请计算 (12+8) * 3.5 等于多少？"

# 2) 接入真实 DeepSeek 模型
export DEEPSEEK_API_KEY=sk-你的key        # 在 https://platform.deepseek.com 申请
PYTHONPATH=src python -m deepseek_agent "现在北京时间是几点？" --verbose
PYTHONPATH=src python -m deepseek_agent --model deepseek-reasoner "请把 36.5 C to F 换算一下"

# 3) 当作库使用
```

```python
import sys; sys.path.insert(0, "src")
from deepseek_agent import ReActAgent, DeepSeekClient, MockLLM

agent = ReActAgent(MockLLM())            # 或 DeepSeekClient()
result = agent.run("请计算 1024 * 768 等于多少？")
print(result.answer)                     # 根据工具返回的结果：1024 * 768 = 786432
for step in result.steps:                # 完整的推理轨迹，可解释、可调试
    print(step.thought, step.action, step.action_input, step.observation)
```

也可以正式安装：`pip install -e .` 后直接使用 `deepseek-agent` 命令。

## 🧪 测试

```bash
python -m unittest discover -s tests -v   # 10 个测试，全部离线运行
```

测试覆盖：计算器正确性与**代码注入拦截**、单位换算（含量纲错误检测）、Agent 对计算/换算/闲聊三类问题的工具选择、记忆写入。

## 🛠️ 扩展：加一个自己的工具只要三步

```python
from deepseek_agent import ReActAgent, MockLLM
from deepseek_agent.tools import Tool, default_tools

tools = default_tools()
tools["word_count"] = Tool("word_count", "统计文本字数", lambda s: f"共 {len(s)} 字")
agent = ReActAgent(MockLLM(), tools=tools)
```

## 📁 项目结构

```
deepseek-agent-lab/
├── src/deepseek_agent/
│   ├── llm.py       # DeepSeek 客户端 + MockLLM
│   ├── agent.py     # ReAct 推理循环
│   ├── tools.py     # 工具注册表与内置工具
│   ├── memory.py    # 对话记忆（JSON 持久化）
│   └── cli.py       # 命令行入口
├── examples/demo.py # 三个问题的完整演示
├── tests/test_agent.py
└── pyproject.toml
```

## 🗺️ 路线图

- [ ] 接入 DeepSeek 原生 Function Calling 协议
- [ ] 增加网页检索 / 本地知识库（RAG）工具
- [ ] 多 Agent 协作（规划者 + 执行者）
- [ ] FastAPI 服务化与流式输出

## 📄 License

MIT © 王佳鑫 (Jiaxin Wang)
