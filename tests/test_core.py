"""核心逻辑的回归测试:参数渲染与工作流声明校验。"""

import base64
import json
import sys
import tempfile
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


class TestImageParam(unittest.TestCase):
    def setUp(self):
        self.wfs = load_workflows(ROOT / "workflows")
        self.img = Path(self.enterContext(tempfile.TemporaryDirectory())) / "frame.png"
        # 最小合法 PNG(1x1)
        self.img.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg"
            "YGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="))

    def test_payload_carries_upload(self):
        payload, resolved = self.wfs["h3_i2v"].build_payload(
            {"image": str(self.img), "prompt": "x"})
        self.assertEqual(len(payload["images"]), 1)
        up = payload["images"][0]
        self.assertTrue(up["name"].endswith(".png"))
        self.assertEqual(base64.b64decode(up["image"]), self.img.read_bytes())
        # 节点图里 LoadImage 引用的是上传名而不是本地路径
        self.assertEqual(payload["workflow"]["114"]["inputs"]["image"], up["name"])
        # 任务记录里保留本地路径,供复现
        self.assertEqual(resolved["image"], str(self.img))

    def test_missing_file_is_param_error(self):
        with self.assertRaises(ParamError) as cm:
            self.wfs["h3_i2v"].build_payload({"image": "/no/such.png", "prompt": "x"})
        self.assertIn("不存在", str(cm.exception))

    def test_bad_extension(self):
        bad = self.img.with_suffix(".txt")
        bad.write_bytes(b"nope")
        with self.assertRaises(ParamError):
            self.wfs["h3_i2v"].build_payload({"image": str(bad), "prompt": "x"})

    def test_non_image_workflow_payload_has_no_images_key(self):
        payload, _ = self.wfs["sdxl_turbo"].build_payload({"prompt": "cat"})
        self.assertNotIn("images", payload)


class TestImagesParam(unittest.TestCase):
    """images 类型(可变张数参考图):按张数克隆槽位节点并接线。"""

    def setUp(self):
        self.wfs = load_workflows(ROOT / "workflows")
        self.tmp = tempfile.TemporaryDirectory()
        self.a = Path(self.tmp.name) / "a.png"
        self.b = Path(self.tmp.name) / "b.png"
        self.a.write_bytes(b"\x89PNG\r\n\x1a\nAAA")
        self.b.write_bytes(b"\x89PNG\r\n\x1a\nBBB")

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, paths):
        return self.wfs["h3_r2v"].build_payload(
            {"ref_images": [str(p) for p in paths], "prompt": "<Picture 1> x"})

    def test_slot_count_matches_image_count(self):
        for n, paths in ((1, [self.a]), (2, [self.a, self.b])):
            with self.subTest(n=n):
                payload, _ = self._build(paths)
                g = payload["workflow"]
                keys = [k for k in g["136"]["inputs"] if k.startswith("ref_images.")]
                self.assertEqual(len(keys), n)
                self.assertEqual(len(payload["images"]), n)
                # 每个槽位都接到一个真实存在的 LoadImage 节点
                for k in keys:
                    node_id = g["136"]["inputs"][k][0]
                    self.assertEqual(g[node_id]["class_type"], "LoadImage")

    def test_no_dangling_refs_or_leftover_template(self):
        g = self._build([self.a, self.b])[0]["workflow"]
        ids = set(g)
        for nid, node in g.items():
            for k, v in node["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    self.assertIn(v[0], ids, f"{nid}.{k} 指向不存在的节点")
        self.assertNotIn("REF_IMAGE_SLOT_TEMPLATE", json.dumps(g))

    def test_same_image_uploaded_once(self):
        payload, _ = self._build([self.a, self.b, self.a])
        self.assertEqual(len(payload["images"]), 2)          # 去重
        g = payload["workflow"]
        keys = [k for k in g["136"]["inputs"] if k.startswith("ref_images.")]
        self.assertEqual(len(keys), 3)                        # 但槽位仍是 3 个

    def test_rejects_empty_and_overflow(self):
        with self.assertRaises(ParamError):
            self._build([])
        with self.assertRaises(ParamError):
            self._build([self.a] * 10)
