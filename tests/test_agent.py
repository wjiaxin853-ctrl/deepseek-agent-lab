"""Unit tests - run with:  python -m unittest discover -s tests -v
All tests are offline (MockLLM), so CI needs no API key or network.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deepseek_agent import MockLLM, ReActAgent  # noqa: E402
from deepseek_agent.tools import calculator, text_stats, unit_convert  # noqa: E402


class TestTools(unittest.TestCase):
    def test_calculator(self):
        self.assertIn("70", calculator("(12+8) * 3.5"))

    def test_calculator_rejects_code(self):
        with self.assertRaises(Exception):
            calculator("__import__('os').system('echo hacked')")

    def test_unit_convert_length(self):
        self.assertIn("6.2137", unit_convert("10 km to mi"))

    def test_unit_convert_temperature(self):
        self.assertIn("97.70", unit_convert("36.5 C to F"))

    def test_unit_convert_dimension_mismatch(self):
        with self.assertRaises(Exception):
            unit_convert("10 km to kg")

    def test_text_stats(self):
        self.assertIn("字符数: 11", text_stats("hello world"))


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.agent = ReActAgent(MockLLM())

    def test_agent_uses_calculator(self):
        result = self.agent.run("请计算 (12+8) * 3.5 等于多少？")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].action, "calculator")
        self.assertIn("70", result.answer)

    def test_agent_uses_unit_convert(self):
        result = self.agent.run("请把 10 km to mi 换算一下")
        self.assertEqual(result.steps[0].action, "unit_convert")
        self.assertIn("6.2137", result.answer)

    def test_agent_direct_answer(self):
        result = self.agent.run("你好，介绍一下你自己")
        self.assertEqual(result.steps, [])
        self.assertTrue(result.answer)

    def test_memory_records_turn(self):
        self.agent.run("请计算 1 + 2 等于多少？")
        msgs = self.agent.memory.as_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
