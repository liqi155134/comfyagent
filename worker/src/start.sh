#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# ---------------------------------------------------------------------------
# GPU pre-flight check
# Verify that the GPU is accessible before starting ComfyUI. If PyTorch
# cannot initialize CUDA the worker will never be able to process jobs,
# so we fail fast with an actionable error message.
# 预检逻辑在 /gpu_check.py(独立文件,不内联 —— 内联 python 的注释里一个
# 双引号就切断过 bash 字符串,把整个 endpoint 送进 crash loop)。
# ---------------------------------------------------------------------------
echo "worker-comfyui: Checking GPU availability..."
if ! GPU_CHECK=$(python /gpu_check.py 2>&1); then
    echo "worker-comfyui: GPU is not available or incompatible with this PyTorch build:"
    echo "worker-comfyui: $GPU_CHECK"
    echo "worker-comfyui: 'no kernel image is available' 说明这个 torch 构建缺少该 GPU 架构的 kernel;"
    echo "worker-comfyui: 否则可能是这台机器的 GPU 没初始化好 —— 联系 RunPod 支持。"
    exit 1
fi
echo "worker-comfyui: GPU available — $GPU_CHECK"

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is WARNING.
: "${COMFY_LOG_LEVEL:=WARNING}"

# PID file used by the handler to detect if ComfyUI is still running
COMFY_PID_FILE="/tmp/comfyui.pid"

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen \
        --use-pytorch-cross-attention --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata \
        --use-pytorch-cross-attention --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi
