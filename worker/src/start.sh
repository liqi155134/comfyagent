#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# ---------------------------------------------------------------------------
# GPU pre-flight check
# Verify that the GPU is accessible before starting ComfyUI. If PyTorch
# cannot initialize CUDA the worker will never be able to process jobs,
# so we fail fast with an actionable error message.
# ---------------------------------------------------------------------------
echo "worker-comfyui: Checking GPU availability..."
if ! GPU_CHECK=$(python -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    # 必须真的 launch 一个 kernel。上面那些只碰驱动的调用,在「这个 torch 构建
    # 没有该架构编译 kernel」时(例:旧 torch 撞新卡)照样返回成功 —— 于是 worker
    # 正常起来、ComfyUI 在第一次 GPU 运算时死掉,最后表现为和真实原因毫无关系的
    # "server not reachable"。在这里炸掉,原因是明确的。
    _ = (torch.zeros(8, device='cuda') + 1).sum().item()
    torch.cuda.synchronize()
    print(f'OK: {name} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}, cuda {torch.version.cuda}')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>&1); then
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
