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


# RunPod serverless 每个 endpoint 最多绑 3 个 GPU **档位**。
#
# 两个坑叠在一起,都只有网页 UI 会提示:
#   1. REST API 收下 5 种、12 种型号都返回 200、不给任何警告,但配置实际非法 ——
#      worker 永远停在 initializing,不报错、不标 throttled,表现和「没有 GPU
#      可调度」一模一样。
#   2. 上限数的是**档位**不是型号。平台把具体型号展开成档位(4090→ADA_24、
#      A5000→AMPERE_24…),一个型号可能激活整个档位,所以传 3 个型号照样可能
#      展开成 6 个档位而超限。
#
# 结论:传型号要少而精,并且提交后回读 gpuIds 核对真实档位数。
MAX_GPU_TIERS = 3
MAX_GPU_TYPES = 3


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
    # 不加 UA 时 GraphQL 端点会被 Cloudflare 拦(403 / error 1010)
    req.add_header("User-Agent", "comfyagent/0.1")
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


def _check_gpu_types(gpu_type_ids):
    n = len(gpu_type_ids or [])
    if n == 0:
        raise RunpodError("至少要指定 1 种 GPU 类型")
    if n > MAX_GPU_TYPES:
        raise RunpodError(
            f"指定了 {n} 种 GPU 类型,RunPod 上限是 {MAX_GPU_TYPES} 种。"
            f"API 不会拒绝超限配置,但 worker 会永远卡在 initializing。"
            f"请挑最多 {MAX_GPU_TYPES} 种:{list(gpu_type_ids)}"
        )


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
                        execution_timeout_ms=None, allowed_cuda=None,
                        min_cuda_version=None, scaler_type=None, scaler_value=None):
        _check_gpu_types(gpu_type_ids)
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
        # allowedCudaVersions 是精确匹配且枚举有上限(实测传 "13.1" 直接 400);
        # 要表达"13.0 及以上"必须用 minCudaVersion。
        if allowed_cuda:
            body["allowedCudaVersions"] = allowed_cuda
        if min_cuda_version:
            body["minCudaVersion"] = min_cuda_version
        if scaler_type:
            body["scalerType"] = scaler_type
        if scaler_value is not None:
            body["scalerValue"] = scaler_value
        return _request(f"{REST}/endpoints", self.key, "POST", body)

    def endpoint_gpu_tiers(self, endpoint_id):
        """回读 RunPod 内部把型号展开成了哪些档位。

        REST 只回显你传进去的型号,看不到展开结果;只有 GraphQL 的 gpuIds
        暴露真实档位。超过 3 个档位的 endpoint 是坏的,但平台不会告诉你。
        """
        r = _request("https://api.runpod.io/graphql", self.key, "POST",
                     {"query": "{ myself { endpoints { id gpuIds } } }"})
        for e in r.get("data", {}).get("myself", {}).get("endpoints", []):
            if e["id"] == endpoint_id:
                ids = e.get("gpuIds") or ""
                return [x for x in ids.split(",") if x and not x.startswith("-")]
        return []

    def assert_gpu_tiers_ok(self, endpoint_id):
        """提交任务前的自检:档位超限就大声失败,别等 worker 卡死才发现。"""
        tiers = self.endpoint_gpu_tiers(endpoint_id)
        if len(tiers) > MAX_GPU_TIERS:
            raise RunpodError(
                f"endpoint {endpoint_id} 展开成 {len(tiers)} 个 GPU 档位,"
                f"上限 {MAX_GPU_TIERS}:{tiers}。worker 会永远卡在 initializing。"
                f"减少 gpuTypeIds 里的型号数量。"
            )
        return tiers

    def update_endpoint(self, endpoint_id, **fields):
        if "gpuTypeIds" in fields:
            _check_gpu_types(fields["gpuTypeIds"])
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
