"""worker handler 的回归测试。

重点覆盖两处从上游继承来、在 fork / 上游最新 / art_ai 三份代码里都存在的缺陷:
  1. WS 监听循环没有终止出口 —— 丢一次完成事件就空转到平台 execution timeout
  2. validate_input 对非 dict 的 images 元素抛 TypeError 而不是返回友好错误
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# handler.py 依赖的第三方包在测试环境里不一定装得全,注入替身即可 import。
for _m in ("runpod", "runpod.serverless", "runpod.serverless.utils", "boto3", "botocore", "botocore.config"):
    sys.modules.setdefault(_m, MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
import handler  # noqa: E402


class TestHistorySettled(unittest.TestCase):
    """_history_settled 是 WS 丢事件时唯一的出口,它的兜底方向决定了失败模式。"""

    def _probe(self, history_value=None, raises=None):
        with patch.object(handler, "get_history") as gh:
            if raises is not None:
                gh.side_effect = raises
            else:
                gh.return_value = history_value
            return handler._history_settled("pid-1")

    def test_completed(self):
        settled, errs = self._probe({"pid-1": {"status": {"completed": True}}})
        self.assertTrue(settled)
        self.assertEqual(errs, [])

    def test_error_with_detail(self):
        settled, errs = self._probe({"pid-1": {"status": {
            "status_str": "error",
            "messages": [["execution_error", {
                "node_type": "KSampler", "node_id": "3",
                "exception_message": "CUDA out of memory",
            }]],
        }}})
        self.assertTrue(settled)
        self.assertEqual(len(errs), 1)
        self.assertIn("KSampler", errs[0])
        self.assertIn("CUDA out of memory", errs[0])

    def test_error_without_detail_still_reports(self):
        """报错但拿不到细节时也必须终结,并给出可辨识的兜底文案。"""
        settled, errs = self._probe({"pid-1": {"status": {"status_str": "error", "messages": []}}})
        self.assertTrue(settled)
        self.assertEqual(len(errs), 1)

    def test_still_running_is_not_settled(self):
        settled, errs = self._probe({"pid-1": {"status": {"completed": False}}})
        self.assertFalse(settled)
        self.assertEqual(errs, [])

    def test_prompt_absent_is_not_settled(self):
        settled, _ = self._probe({})
        self.assertFalse(settled)

    def test_probe_failure_never_fabricates_terminal_state(self):
        """最关键的一条:探测失败必须回落到「还没结束」。

        反过来(失败当成已完成)会把仍在采样的任务判成结束,丢掉真实产物,
        而且这种错误是静默的 —— 比空转难查一个数量级。
        """
        settled, errs = self._probe(raises=RuntimeError("connection refused"))
        self.assertFalse(settled)
        self.assertEqual(errs, [])

    def test_legacy_comfyui_without_status_field(self):
        """老版本 ComfyUI 的 history 条目没有 status,只能靠 outputs 判断。"""
        settled, _ = self._probe({"pid-1": {"outputs": {"9": {"images": [{}]}}}})
        self.assertTrue(settled)
        settled, _ = self._probe({"pid-1": {"outputs": {}}})
        self.assertFalse(settled)


class TestValidateInput(unittest.TestCase):
    def test_accepts_wellformed_images(self):
        data, err = handler.validate_input({"workflow": {"1": {}}, "images": [{"name": "a.png", "image": "b64"}]})
        self.assertIsNone(err)
        self.assertIsNotNone(data)

    def test_rejects_malformed_without_crashing(self):
        """[123] / [None] 以前会抛 TypeError 冒泡到框架,而不是返回友好错误。"""
        for bad in ([123], [None], ["str"], [{"name": "a"}], "notalist"):
            with self.subTest(images=bad):
                data, err = handler.validate_input({"workflow": {"1": {}}, "images": bad})
                self.assertIsNone(data)
                self.assertIsInstance(err, str)

    def test_missing_workflow(self):
        data, err = handler.validate_input({})
        self.assertIsNone(data)
        self.assertIn("workflow", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
