"""5.2.1 Prompt工程与生成策略：TestRAGPrompt。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import unittest
from src.generation.prompt_template import build_rag_messages


class TestRAGPrompt(unittest.TestCase):
    """核验消息角色、参数填入和文档原文保留，不能据此推断模型答案质量。"""


    def test_empty_context_has_explicit_notice(self):
        for context in ("", " \n\t"):
            with self.subTest(context=context):
                messages = build_rag_messages("有什么相关文献？", context)
                self.assertIn("当前知识库中未找到相关文档。", messages[1].content)
        self.assertIn("当前知识库中未找到相关文档。", build_rag_messages("问题")[1].content)


    def test_braces_formula_and_markdown_are_not_reformatted(self):
        context = "[参考文档1 - 来源: math.md；行1–3]\n$x_{i}={a}+{context}$\n```json\n{\"k\": 5}\n```"
        question = "How is {question} related to $x_{i}$?"
        human = build_rag_messages(question, context)[1].content
        self.assertIn(f"【检索上下文】\n{context}\n\n【用户问题】\n{question}", human)
        self.assertIn("对应引用编号（如[参考文档1]）", human)


if __name__ == "__main__":
    unittest.main()
