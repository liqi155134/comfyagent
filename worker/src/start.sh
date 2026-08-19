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

# ---------------------------------------------------------------------------
# Model caching(RunPod):endpoint 配了 modelName 时,平台把整个 HF repo
# 下载到缓存卷并注入 MODEL_NAME / MODEL_REVISION 两个 env(实测,2026-08)。
# 转存 repo 的子目录结构与 ComfyUI models 目录一一对应,逐目录 symlink。
# 没配 model caching(如 smoke 镜像)时整块跳过,零影响。
# ---------------------------------------------------------------------------
if [ -n "$MODEL_NAME" ] && [ -n "$MODEL_REVISION" ]; then
    SNAP="/runpod-volume/huggingface-cache/hub/models--${MODEL_NAME//\//--}/snapshots/${MODEL_REVISION}"
    if [ -d "$SNAP" ]; then
        echo "worker-comfyui: Linking cached model files from $SNAP"
        for sub in diffusion_models text_encoders vae loras checkpoints clip_vision; do
            if [ -d "$SNAP/$sub" ]; then
                mkdir -p "/comfyui/models/$sub"
                ln -sfn "$SNAP/$sub"/* "/comfyui/models/$sub/"
                echo "worker-comfyui:   models/$sub <- $(ls "$SNAP/$sub" | tr '\n' ' ')"
            fi
        done
    else
        echo "worker-comfyui: WARNING: MODEL_NAME=$MODEL_NAME set but snapshot dir missing: $SNAP"
    fi
fi

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is WARNING.
: "${COMFY_LOG_LEVEL:=WARNING}"

# SageAttention 按 env 开关(h3 模板置 true;镜像里没装 sage 时不要开)
COMFY_EXTRA_ARGS=""
if [ "$USE_SAGE_ATTENTION" = "true" ]; then
    COMFY_EXTRA_ARGS="--use-sage-attention"
    echo "worker-comfyui: SageAttention enabled"
fi

# PID file used by the handler to detect if ComfyUI is still running
COMFY_PID_FILE="/tmp/comfyui.pid"

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen \
        --verbose "${COMFY_LOG_LEVEL}" --log-stdout ${COMFY_EXTRA_ARGS} &
    echo $! > "$COMFY_PID_FILE"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata \
        --verbose "${COMFY_LOG_LEVEL}" --log-stdout ${COMFY_EXTRA_ARGS} &
    echo $! > "$COMFY_PID_FILE"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi
