"""部署声明与 deploy 流程的回归测试(不碰真实 RunPod)。"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfyagent.core import deploy as deploy_mod  # noqa: E402


class FakeClient:
    """记录调用的假 RunpodClient。"""

    def __init__(self, templates=None, endpoints=None):
        self._templates = list(templates or [])
        self._endpoints = list(endpoints or [])
        self.calls = []

    def list_templates(self):
        return self._templates

    def list_endpoints(self):
        return self._endpoints

    def create_template(self, name, image, container_disk_gb=30, env=None):
        self.calls.append(("create_template", name, image, env))
        return {"id": "tpl-new"}

    def update_template(self, template_id, **fields):
        self.calls.append(("update_template", template_id, fields))
        return {"id": template_id}

    def create_endpoint(self, name, template_id, gpu_type_ids, **kw):
        self.calls.append(("create_endpoint", name, template_id, kw))
        return {"id": "ep-new"}

    def update_endpoint(self, endpoint_id, **fields):
        self.calls.append(("update_endpoint", endpoint_id, fields))
        return {"id": endpoint_id}

    def assert_gpu_tiers_ok(self, endpoint_id):
        self.calls.append(("assert_gpu_tiers_ok", endpoint_id))


class TestSpec(unittest.TestCase):
    def test_repo_spec_loads_and_merges_defaults(self):
        specs = deploy_mod.load_spec()
        self.assertIn("h3", specs)
        h3 = specs["h3"]
        # defaults 合并进来了
        self.assertEqual(h3["workers_min"], 0)
        self.assertEqual(h3["min_cuda_version"], "13.0")
        # 条目自己的值覆盖 defaults
        self.assertTrue(h3["image"].startswith("ghcr.io/"))

    def test_no_deployment_has_standing_billing(self):
        """workers_min>0 = 持续计费,必须是显式人工决定,不能悄悄进声明。"""
        for name, spec in deploy_mod.load_spec().items():
            self.assertEqual(spec.get("workers_min", 0), 0,
                             f"{name} 的 workers_min>0 会持续计费")


class TestApply(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "image": "ghcr.io/x/y:1", "gpu_type_ids": ["NVIDIA H100 80GB HBM3"],
            "workers_min": 0, "workers_max": 2, "flashboot": True,
            "idle_timeout": 5,
            "min_cuda_version": "13.0", "scaler_type": "QUEUE_DELAY",
            "scaler_value": 4, "container_disk_gb": 80,
            "execution_timeout_ms": 1800000, "env": {"A": "1"},
        }

    def test_creates_when_absent(self):
        c = FakeClient()
        with tempfile.TemporaryDirectory() as d:
            deploy_mod.config.PATH = Path(d) / "endpoints.json"
            r = deploy_mod.apply("demo", self.spec, client=c)
        self.assertEqual(r["endpoint_id"], "ep-new")
        kinds = [x[0] for x in c.calls]
        self.assertEqual(kinds, ["create_template", "create_endpoint",
                                 "assert_gpu_tiers_ok"])

    def test_updates_when_present(self):
        """幂等:同名资源已存在时走 PATCH,不再造第二套。"""
        c = FakeClient(templates=[{"id": "tpl-1", "name": "comfyagent-demo"}],
                       endpoints=[{"id": "ep-1", "name": "comfyagent-demo"}])
        with tempfile.TemporaryDirectory() as d:
            deploy_mod.config.PATH = Path(d) / "endpoints.json"
            r = deploy_mod.apply("demo", self.spec, client=c)
        self.assertEqual(r["endpoint_id"], "ep-1")
        self.assertEqual([x[0] for x in c.calls],
                         ["update_template", "update_endpoint", "assert_gpu_tiers_ok"])

    def test_underscore_name_maps_to_dash_resource(self):
        c = FakeClient()
        with tempfile.TemporaryDirectory() as d:
            deploy_mod.config.PATH = Path(d) / "endpoints.json"
            deploy_mod.apply("h3_ck", self.spec, client=c)
        self.assertEqual(c.calls[0][1], "comfyagent-h3-ck")

    def test_resource_name_override_finds_legacy_resource(self):
        """历史遗留的异名资源:用 resource_name 指过去,走更新而不是重建。"""
        c = FakeClient(templates=[{"id": "tpl-old", "name": "comfyagent-demo-legacy"}],
                       endpoints=[{"id": "ep-old", "name": "comfyagent-demo-legacy"}])
        spec = dict(self.spec, resource_name="comfyagent-demo-legacy")
        with tempfile.TemporaryDirectory() as d:
            deploy_mod.config.PATH = Path(d) / "endpoints.json"
            r = deploy_mod.apply("demo", spec, client=c)
        self.assertEqual(r["endpoint_id"], "ep-old")
        self.assertEqual([x[0] for x in c.calls],
                         ["update_template", "update_endpoint", "assert_gpu_tiers_ok"])

    def test_standing_billing_needs_explicit_flag(self):
        spec = dict(self.spec, workers_min=1)
        with self.assertRaises(deploy_mod.DeployError) as cm:
            deploy_mod.apply("demo", spec, client=FakeClient())
        self.assertIn("持续计费", str(cm.exception))

    def test_dry_run_does_not_touch_client(self):
        c = FakeClient()
        r = deploy_mod.apply("demo", self.spec, client=c, dry_run=True)
        self.assertEqual(r["action"], "dry-run")
        self.assertEqual(c.calls, [])


if __name__ == "__main__":
    unittest.main()


class TestNoDrift(unittest.TestCase):
    """deploy 必须把声明里的每个端点字段都推到现网。

    漏推的字段会造成"改了声明却不生效"的静默漂移(实际发生过:gpuTypeIds、
    idleTimeout 漏推,现网 idle 停在 10 秒而声明是 5)。
    """

    # 声明字段 -> RunPod API 字段
    MUST_PUSH = {
        "gpu_type_ids": "gpuTypeIds",
        "workers_min": "workersMin",
        "workers_max": "workersMax",
        "idle_timeout": "idleTimeout",
        "flashboot": "flashboot",
        "execution_timeout_ms": "executionTimeoutMs",
        "min_cuda_version": "minCudaVersion",
        "scaler_type": "scalerType",
        "scaler_value": "scalerValue",
    }

    def test_update_pushes_every_declared_field(self):
        c = FakeClient(templates=[{"id": "tpl-1", "name": "comfyagent-demo"}],
                       endpoints=[{"id": "ep-1", "name": "comfyagent-demo"}])
        spec = {
            "image": "ghcr.io/x/y:1", "gpu_type_ids": ["NVIDIA H100 80GB HBM3"],
            "workers_min": 0, "workers_max": 2, "flashboot": True,
            "idle_timeout": 5, "min_cuda_version": "13.0",
            "scaler_type": "QUEUE_DELAY", "scaler_value": 4,
            "container_disk_gb": 80, "execution_timeout_ms": 1800000, "env": {},
        }
        with tempfile.TemporaryDirectory() as d:
            deploy_mod.config.PATH = Path(d) / "endpoints.json"
            deploy_mod.apply("demo", spec, client=c)

        pushed = next(x[2] for x in c.calls if x[0] == "update_endpoint")
        for decl, api in self.MUST_PUSH.items():
            with self.subTest(field=decl):
                self.assertIn(api, pushed, f"{decl} 没有推到现网,会造成静默漂移")
                self.assertEqual(pushed[api], spec[decl])
