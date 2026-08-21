"""模型获取降级链的第二级:从 HF 直拉整个精简 repo 到 ComfyUI models 目录。

前提约定:HF_FETCH_REPO 指向的 repo 目录结构与 ComfyUI models 目录一一对应
(diffusion_models/ text_encoders/ vae/ loras/),snapshot_download 直接
落到 /comfyui/models 即为最终形态。

HF_FETCH_PATTERNS(可选,分号分隔)按需只拉指定文件,不给就拉全部
*.safetensors。有了它就能**直接用官方多变体 repo**(如 Comfy-Org/MiniMax-H3
同时放着 bf16/pruned/int8 五种精度和 fl2va/ref2va 两套模型,全拉是几百 GB),
按端点各取所需,省掉"自建精简 repo"这一层搬运。

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

patterns = [x.strip() for x in os.environ.get("HF_FETCH_PATTERNS", "").split(";") if x.strip()]
patterns = patterns or ["*.safetensors"]

t0 = time.time()
print(f"fetch_models: downloading {repo} -> /comfyui/models", flush=True)
print(f"fetch_models: patterns = {patterns}", flush=True)
snapshot_download(
    repo,
    local_dir="/comfyui/models",
    allow_patterns=patterns,
    max_workers=5,
)

# 显式清点:pattern 写错时 snapshot_download 会静默拉个空,
# 而症状要到 ComfyUI 报"找不到模型"才暴露,那时已经烧掉一次冷启动。
got = []
for root, _, files in os.walk("/comfyui/models"):
    for f in files:
        if f.endswith(".safetensors"):
            fp = os.path.join(root, f)
            got.append((os.path.relpath(fp, "/comfyui/models"), os.path.getsize(fp)))
if not got:
    print(f"fetch_models: FATAL - patterns {patterns} 一个文件都没匹配到", flush=True)
    sys.exit(1)
total = sum(sz for _, sz in got)
for name, sz in sorted(got):
    print(f"fetch_models:   {sz / 1e9:6.1f}GB  {name}", flush=True)
print(f"fetch_models: done in {time.time() - t0:.0f}s, "
      f"{len(got)} files / {total / 1e9:.1f}GB", flush=True)
