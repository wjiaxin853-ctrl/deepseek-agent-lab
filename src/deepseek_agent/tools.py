"""Tool registry for the agent.

A tool is a plain callable ``str -> str`` plus a one-line description that
is shown to the LLM in the system prompt. Tools never use ``eval`` on raw
input: the calculator parses the expression with :mod:`ast` and only
allows arithmetic nodes, so LLM-generated input cannot execute code.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass
class Tool:
    name: str
    description: str
    func: object  # Callable[[str], str]

    def __call__(self, arg: str) -> str:
        try:
            return str(self.func(arg.strip()))
        except Exception as exc:  # tools report errors, never crash the loop
            return f"Error: {exc}"


# ---------------------------------------------------------------- calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"不允许的表达式节点: {type(node).__name__}")


def calculator(expr: str) -> str:
    """Safely evaluate an arithmetic expression, e.g. ``(12+8) * 3.5``."""
    result = _eval_node(ast.parse(expr, mode="eval"))
    return f"{expr} = {result:g}" if isinstance(result, float) else f"{expr} = {result}"


# --------------------------------------------------------------------- clock
def now(tz: str = "Asia/Shanghai") -> str:
    """Return the current date/time in the given IANA timezone."""
    try:
        zone = ZoneInfo(tz or "Asia/Shanghai")
    except Exception:
        zone = ZoneInfo("Asia/Shanghai")
    t = datetime.now(zone)
    return t.strftime("%Y-%m-%d %H:%M:%S %A (%Z)")


# ------------------------------------------------------------- unit convert
_FACTORS = {  # to a common base unit per dimension
    "km": ("length", 1000.0), "mi": ("length", 1609.344), "m": ("length", 1.0),
    "kg": ("mass", 1.0), "lb": ("mass", 0.45359237), "g": ("mass", 0.001),
}


def unit_convert(spec: str) -> str:
    """Convert ``<value> <unit> to <unit>``, e.g. ``10 km to mi``, ``36.5 C to F``."""
    parts = spec.replace("°", "").split()
    if len(parts) != 4 or parts[2].lower() != "to":
        raise ValueError("格式应为: <数值> <单位> to <单位>，如 '10 km to mi'")
    value, src, dst = float(parts[0]), parts[1], parts[3]
    if src in ("C", "F") or dst in ("C", "F"):
        celsius = value if src == "C" else (value - 32) * 5 / 9
        out = celsius if dst == "C" else celsius * 9 / 5 + 32
        return f"{value:g}°{src} = {out:.2f}°{dst}"
    if src not in _FACTORS or dst not in _FACTORS:
        raise ValueError(f"暂不支持的单位: {src} / {dst}（支持 km, mi, m, kg, lb, g, C, F）")
    if _FACTORS[src][0] != _FACTORS[dst][0]:
        raise ValueError(f"单位量纲不同，无法换算: {src} -> {dst}")
    out = value * _FACTORS[src][1] / _FACTORS[dst][1]
    return f"{value:g} {src} = {out:.4f} {dst}"


def text_stats(text: str) -> str:
    """Count characters / words / lines of a piece of text."""
    return (f"字符数: {len(text)}，词数: {len(text.split())}，"
            f"行数: {text.count(chr(10)) + 1 if text else 0}")


def default_tools() -> dict[str, Tool]:
    return {
        "calculator": Tool("calculator", "计算算术表达式，如 (12+8)*3.5", calculator),
        "now": Tool("now", "查询某个时区的当前日期时间，输入时区名如 Asia/Shanghai", now),
        "unit_convert": Tool("unit_convert", "单位换算，输入如 '10 km to mi' 或 '36.5 C to F'", unit_convert),
        "text_stats": Tool("text_stats", "统计一段文本的字符数、词数和行数", text_stats),
    }
