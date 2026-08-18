"""核心逻辑的回归测试:参数渲染与工作流声明校验。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfyagent.core.registry import ParamError, load_workflows  # noqa: E402
from comfyagent.core.render import MissingParam, referenced_params, render  # noqa: E402


class TestRender(unittest.TestCase):
    def test_whole_placeholder_preserves_type(self):
        """整值占位必须保留原始类型 —— ComfyUI 对 seed/steps 的类型敏感,
        传成字符串会在节点校验阶段被拒,而报错指向节点、很难追回模板层。"""
        out = render({"a": "{{seed}}", "b": "{{ratio}}"}, {"seed": 42, "ratio": 0.5})
        self.assertIsInstance(out["a"], int)
        self.assertEqual(out["a"], 42)
        self.assertIsInstance(out["b"], float)

    def test_inline_interpolation(self):
        self.assertEqual(render("a {{x}} cat", {"x": "red"}), "a red cat")

    def test_nested_structures(self):
        out = render({"n": [{"v": "{{x}}"}, "{{x}}"]}, {"x": 7})
        self.assertEqual(out, {"n": [{"v": 7}, 7]})

    def test_missing_param_raises(self):
        with self.assertRaises(MissingParam):
            render({"a": "{{nope}}"}, {})

    def test_non_string_passthrough(self):
        self.assertEqual(render({"a": 1, "b": None, "c": True}, {}),
                         {"a": 1, "b": None, "c": True})

    def test_referenced_params(self):
        self.assertEqual(referenced_params({"a": "{{x}}", "b": ["{{y}}"]}), {"x", "y"})


class TestRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wfs = load_workflows(ROOT / "workflows")

    def test_workflows_load(self):
        self.assertIn("sdxl_turbo", self.wfs)

    def test_defaults_filled(self):
        _, resolved = self.wfs["sdxl_turbo"].build({"prompt": "cat"})
        self.assertEqual(resolved["width"], 512)
        self.assertEqual(resolved["steps"], 1)

    def test_seed_randomized_but_recorded(self):
        """省略 seed 时随机,但必须记进 resolved —— 否则「复现上次那张」做不到。"""
        _, a = self.wfs["sdxl_turbo"].build({"prompt": "cat"})
        _, b = self.wfs["sdxl_turbo"].build({"prompt": "cat"})
        self.assertIsInstance(a["seed"], int)
        self.assertNotEqual(a["seed"], b["seed"])

    def test_explicit_seed_respected(self):
        graph, resolved = self.wfs["sdxl_turbo"].build({"prompt": "cat", "seed": 99})
        self.assertEqual(resolved["seed"], 99)
        self.assertEqual(graph["sampler"]["inputs"]["seed"], 99)

    def test_validation_errors_name_the_param(self):
        """报错信息是给 agent 读的,必须点名参数,它才能自己纠正后重试。"""
        cases = [
            ({}, "prompt"),
            ({"prompt": "x", "steps": 99}, "steps"),
            ({"prompt": "x", "width": 999}, "width"),
            ({"prompt": "x", "bogus": 1}, "bogus"),
        ]
        for given, expect in cases:
            with self.subTest(given=given):
                with self.assertRaises(ParamError) as cm:
                    self.wfs["sdxl_turbo"].build(given)
                self.assertIn(expect, str(cm.exception))

    def test_bool_not_accepted_as_integer(self):
        with self.assertRaises(ParamError):
            self.wfs["sdxl_turbo"].build({"prompt": "x", "steps": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
