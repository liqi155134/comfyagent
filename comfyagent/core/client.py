"""RunPod 客户端 —— 部署面(template/endpoint)与调用面(job)。

状态归一化:RunPod 原生状态有 8 个且大小写混杂,直接透给 agent 会让它写出
一堆脆弱的分支判断。这里收敛成 5 个:queued / running / completed /
failed / cancelled,与产物一起构成 agent 唯一需要理解的契约。
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

REST = "https://rest.runpod.io/v1"
RUN = "https://api.runpod.ai/v2"
CONFIG = Path.home() / ".config" / "comfyagent" / "env"

# RunPod 原生 → 归一化。未知状态一律当 running:
# 猜成 completed 会让调用方以为拿到了产物,猜成 failed 会把还在烧钱的任务
# 当成已结束 —— 两者都比"多轮询一次"糟糕。
_STATUS = {
    "IN_QUEUE": "queued",
    "IN_PROGRESS": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "TIMED_OUT": "failed",
    "RETRIED": "running",
}


class RunpodError(RuntimeError):
    pass


def load_api_key():
    """只从文件或环境变量读,绝不接受命令行参数 —— 免得进 shell 历史。"""
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    if CONFIG.exists():
        for line in CONFIG.read_text(encoding="utf-8").splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RunpodError(
        f"没有 RunPod API key。写入 {CONFIG}(格式 RUNPOD_API_KEY=xxx)"
        " 或设置同名环境变量。"
    )


def _request(url, key, method="GET", payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:500]
        # 带上响应体:RunPod 的 4xx 通常在 body 里写明了哪个字段不对,
        # 只报状态码等于把唯一的线索丢掉。
        raise RunpodError(f"{method} {url} → HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RunpodError(f"{method} {url} 连接失败: {e.reason}") from None


class RunpodClient:
    def __init__(self, api_key=None):
        self.key = api_key or load_api_key()

    # ---------- 部署面 ----------
    def list_templates(self):
        return _request(f"{REST}/templates", self.key)

    def create_template(self, name, image, container_disk_gb=30, env=None):
        return _request(
            f"{REST}/templates", self.key, "POST",
            {
                "name": name,
                "imageName": image,
                "isServerless": True,
                "containerDiskInGb": container_disk_gb,
                "env": env or {},
            },
        )

    def update_template(self, template_id, **fields):
        return _request(f"{REST}/templates/{template_id}", self.key, "PATCH", fields)

    def list_endpoints(self):
        return _request(f"{REST}/endpoints", self.key)

    def create_endpoint(self, name, template_id, gpu_type_ids, workers_min=0,
                        workers_max=1, idle_timeout=5, flashboot=True,
                        execution_timeout_ms=None, allowed_cuda=None):
        body = {
            "name": name,
            "templateId": template_id,
            "computeType": "GPU",
            "gpuTypeIds": gpu_type_ids,
            "workersMin": workers_min,
            "workersMax": workers_max,
            "idleTimeout": idle_timeout,
            "flashboot": flashboot,
        }
        if execution_timeout_ms:
            body["executionTimeoutMs"] = execution_timeout_ms
        if allowed_cuda:
            body["allowedCudaVersions"] = allowed_cuda
        return _request(f"{REST}/endpoints", self.key, "POST", body)

    def update_endpoint(self, endpoint_id, **fields):
        return _request(f"{REST}/endpoints/{endpoint_id}", self.key, "PATCH", fields)

    # ---------- 调用面 ----------
    def health(self, endpoint_id):
        return _request(f"{RUN}/{endpoint_id}/health", self.key)

    def submit(self, endpoint_id, payload):
        r = _request(f"{RUN}/{endpoint_id}/run", self.key, "POST", {"input": payload})
        if "id" not in r:
            raise RunpodError(f"提交成功但响应里没有 job id: {r}")
        return r["id"]

    def status(self, endpoint_id, job_id):
        r = _request(f"{RUN}/{endpoint_id}/status/{job_id}", self.key)
        raw = r.get("status", "")
        return {
            "status": _STATUS.get(raw, "running"),
            "raw_status": raw,
            "output": r.get("output"),
            "error": r.get("error"),
            "delay_ms": r.get("delayTime"),
            "execution_ms": r.get("executionTime"),
        }

    def cancel(self, endpoint_id, job_id):
        return _request(f"{RUN}/{endpoint_id}/cancel/{job_id}", self.key, "POST")
