"""模型获取降级链的第二级:从 HF 直拉整个精简 repo 到 ComfyUI models 目录。

前提约定:HF_FETCH_REPO 指向的 repo 目录结构与 ComfyUI models 目录一一对应
(diffusion_models/ text_encoders/ vae/ loras/),snapshot_download 直接
落到 /comfyui/models 即为最终形态。

为什么存在:RunPod model caching(第一级)实测会在"model pending download"
上卡住数小时且用户侧无法干预;而 worker 直拉实测单流 80MB/s,多流并发下
44GB 约 3-5 分钟,可控、可观测(下载日志走 stdout → 日志 API 可见)。
"""

import os
import sys
import time

repo = os.environ.get("HF_FETCH_REPO")
if not repo:
    sys.exit(0)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download  # noqa: E402

t0 = time.time()
print(f"fetch_models: downloading {repo} -> /comfyui/models", flush=True)
snapshot_download(
    repo,
    local_dir="/comfyui/models",
    allow_patterns=["*.safetensors"],
    max_workers=5,
)
print(f"fetch_models: done in {time.time() - t0:.0f}s", flush=True)
