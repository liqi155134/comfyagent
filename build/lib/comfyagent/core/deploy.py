"""从 deployments.yaml 幂等地创建 / 更新 RunPod template + endpoint。

设计取舍:
- **幂等**:按名字找现存资源,有就 PATCH,没有就 POST。重跑一次不会造出第二套。
- **凭据不进声明**:R2 四件套从 ~/.config/comfyagent/r2_env 读,只在提交给
  RunPod 的请求体里出现,不落 git、不打日志。
- **持续计费红线**:workers_min > 0 需要显式 --allow-billing,默认拒绝执行
  (serverless 按 job 计费是唯一免批形态)。
"""

import os
from pathlib import Path

import yaml

from . import config
from .client import RunpodClient

def _default_spec_path():
    """找部署声明:先看仓库根(开发 / editable 安装),再看包内副本(正式 wheel)。

    正式 wheel 里没有仓库根,声明必须作为 package data 随包走 —— 否则
    `pip install comfyagent` 后 deploy 命令直接报"声明不存在"(2026-08-21 复核实锤)。
    """
    repo_root = Path(__file__).resolve().parent.parent.parent / "deployments.yaml"
    if repo_root.exists():
        return repo_root
    return Path(__file__).resolve().parent / "deployments.yaml"


SPEC_PATH = _default_spec_path()
R2_ENV_PATH = Path.home() / ".config" / "comfyagent" / "r2_env"
_R2_KEYS = ("BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY",
            "BUCKET_ENDPOINT_URL", "BUCKET_NAME")


class DeployError(RuntimeError):
    pass


def load_spec(path=None):
    path = Path(path or SPEC_PATH)
    if not path.exists():
        raise DeployError(f"部署声明不存在: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = doc.get("defaults") or {}
    out = {}
    for name, entry in (doc.get("deployments") or {}).items():
        merged = dict(defaults)
        merged.update(entry or {})
        out[name] = merged
    return out


def _r2_env():
    """从 r2_env 读凭据。文件是 KEY=VALUE 行格式(可带 export 前缀)。"""
    if not R2_ENV_PATH.exists():
        raise DeployError(f"需要 R2 凭据但 {R2_ENV_PATH} 不存在(env_from_r2: true)")
    env = {}
    for line in R2_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in _R2_KEYS if not env.get(k)]
    if missing:
        raise DeployError(f"{R2_ENV_PATH} 缺少: {', '.join(missing)}")
    return {k: env[k] for k in _R2_KEYS}


def _build_env(spec):
    env = dict(spec.get("env") or {})
    if spec.get("env_from_r2"):
        env.update(_r2_env())
    return env


def _find_by_name(items, name):
    for it in items or []:
        if it.get("name") == name:
            return it
    return None


def plan(name, spec):
    """返回人类可读的计划摘要(不含凭据值)。"""
    env_keys = sorted(_build_env(spec).keys())
    return {
        "name": name,
        "resource_name": spec.get("resource_name") or f"comfyagent-{name.replace('_', '-')}",
        "image": spec["image"],
        "gpu_type_ids": spec["gpu_type_ids"],
        "workers": f"{spec['workers_min']}-{spec['workers_max']}",
        "cuda_min": spec["min_cuda_version"],
        "timeout_min": round(spec["execution_timeout_ms"] / 60000, 1),
        "env_keys": env_keys,
    }


def apply(name, spec, client=None, allow_billing=False, dry_run=False):
    """创建或更新一个部署单元,返回 {action, template_id, endpoint_id}。"""
    if spec.get("workers_min", 0) > 0 and not allow_billing:
        raise DeployError(
            f"{name}: workers_min={spec['workers_min']} 会产生持续计费,"
            f"需显式 --allow-billing 确认")
    if dry_run:
        return {"action": "dry-run", **plan(name, spec)}

    client = client or RunpodClient()
    # 资源名默认由逻辑名推导;历史遗留的异名资源用 resource_name 显式指过去,
    # 否则 deploy 会当成"不存在"再造一套(实测踩过:h3 vs comfyagent-h3-sagefix)。
    res_name = spec.get("resource_name") or f"comfyagent-{name.replace('_', '-')}"
    env = _build_env(spec)

    tpl = _find_by_name(client.list_templates(), res_name)
    if tpl:
        client.update_template(tpl["id"], imageName=spec["image"], env=env,
                               containerDiskInGb=spec["container_disk_gb"])
        template_id, tpl_action = tpl["id"], "updated"
    else:
        created = client.create_template(res_name, spec["image"],
                                         container_disk_gb=spec["container_disk_gb"],
                                         env=env)
        template_id, tpl_action = created["id"], "created"

    ep = _find_by_name(client.list_endpoints(), res_name)
    # 声明是事实来源:每个字段都要真的推到现网,否则改了声明却不生效,
    # 现网悄悄漂移(2026-08-21 复核实锤:gpuTypeIds / idleTimeout 曾漏推)。
    ep_fields = {
        "templateId": template_id,
        "gpuTypeIds": spec["gpu_type_ids"],
        "workersMin": spec["workers_min"],
        "workersMax": spec["workers_max"],
        "idleTimeout": spec["idle_timeout"],
        "flashboot": spec["flashboot"],
        "executionTimeoutMs": spec["execution_timeout_ms"],
        "minCudaVersion": spec["min_cuda_version"],
        "scalerType": spec["scaler_type"],
        "scalerValue": spec["scaler_value"],
    }
    if ep:
        client.update_endpoint(ep["id"], **ep_fields)
        endpoint_id, ep_action = ep["id"], "updated"
    else:
        created = client.create_endpoint(
            res_name, template_id, spec["gpu_type_ids"],
            workers_min=spec["workers_min"], workers_max=spec["workers_max"],
            idle_timeout=spec["idle_timeout"], flashboot=spec["flashboot"],
            execution_timeout_ms=spec["execution_timeout_ms"],
            min_cuda_version=spec["min_cuda_version"],
            scaler_type=spec["scaler_type"], scaler_value=spec["scaler_value"])
        endpoint_id, ep_action = created["id"], "created"

    # REST 会静默接受非法 GPU 配置,症状是 worker 永远 initializing。
    # 回读实际展开出的档位做断言,把这类问题挡在部署当场。
    try:
        client.assert_gpu_tiers_ok(endpoint_id)
    except Exception as e:
        raise DeployError(f"{name}: endpoint {endpoint_id} 的 GPU 档位校验没过 — {e}")

    config.set_endpoint(name, endpoint_id, template_id=template_id, image=spec["image"])
    return {"action": f"template {tpl_action}, endpoint {ep_action}",
            "template_id": template_id, "endpoint_id": endpoint_id}
