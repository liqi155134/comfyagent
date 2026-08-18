# ==============================================================================
# smoke 镜像 —— 验证整条链路用,不是生产镜像。
#
# 目的:用一个体积够大能测出真实拉取速度、又小到构建不会超时的模型,把
# 「构建 → 部署 → 冷启动 → 提交 job → 拿产物」跑通一遍,并测出这些数字:
# 构建耗时、镜像拉取 MB/s、FlashBoot 快照恢复耗时。
#
# RunPod GitHub 构建的硬约束:
#   * `docker build` 必须在 30 分钟内完成
#   * 镜像总量 ≤ 80GB
#   * 构建期不能用 GPU
#
# 分层策略(直接决定重建速度,在 30 分钟窗口下很关键):
#   base       —— 系统 + ComfyUI + torch,重且极少变
#   downloader —— 只下模型,重且几乎不变
#   final      —— 只放自己的代码,轻且天天变
# 代码放在最后一层是有意为之:改 handler.py 只重建 final,模型层命中缓存。
# (上游把代码 ADD 进 base、再让 downloader FROM base,结果改一行代码就得重下全部模型。)
# ==============================================================================

ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# ------------------------------------------------------------------------------
# Stage 1: base —— 运行时环境
# ------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS base

# ComfyUI 与 torch 都 pin 死,理由见 docs/decisions.md:
#   * cu128 而非 cu130 —— cu13 要求宿主驱动 ≥580,会切窄可调度 GPU 池,
#     而 serverless 上「没卡」表现为排队不是报错,极难归因。
#   * torch 装在 ComfyUI requirements.txt 之前 —— requirements 声明的是裸 `torch`,
#     PyPI 默认发 CUDA 13 构建(自 2.11 起依赖 nvidia-*-cu13),在驱动 570/575 的
#     宿主上 CUDA init 直接失败。先装 cu128 占位,后面那趟就不会去动它。
ARG COMFYUI_REF=v0.29.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# 只装 ComfyUI / OpenCV / Pillow 运行期真正需要的系统库。
# Python 不走 apt —— 用 uv 的 python-build-standalone,版本可控。
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      ffmpeg \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# ComfyUI 本体:git clone + pin tag,不用 comfy-cli。
# 每个依赖都摊在 requirements.txt 里,可读可调试,且能 pin 到确切 tag/SHA;
# comfy-cli 挑 torch index 的逻辑是个会变的黑盒。
#
# transformers / huggingface-hub 的上界 pin 必须和 requirements 装在同一个 RUN:
# ComfyUI 声明 `transformers>=4.50.3` 且 huggingface-hub 无上限,新装会拉到
# 5.x / 1.x,其破坏性 API 变更同样会让 ComfyUI 启动即崩。同层降级才不会把
# 不想要的版本留在镜像里白白撑大体积。
RUN git clone --depth 1 --branch "${COMFYUI_REF}" \
      https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui \
    && git rev-parse HEAD > /comfyui/.git_sha_pinned \
    && uv pip install --no-cache \
         --index-url "${TORCH_INDEX_URL}" \
         "torch==${TORCH_VERSION}" torchvision torchaudio \
    && uv pip install --no-cache -r requirements.txt \
    && uv pip install --no-cache "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
    && uv pip uninstall xformers 2>/dev/null || true \
    && rm -rf /comfyui/.git /comfyui/.github /comfyui/tests /root/.cache /tmp/*

# handler 运行期依赖
RUN uv pip install --no-cache runpod requests websocket-client boto3

# 构建期断言:装错了就在这里炸。
# 量化 / attention 后端的失效模式往往是「静默出错」而非崩溃,
# 凡是构建期能钉死的事实就必须在构建期钉死。
RUN python -c "\
import torch, importlib.util; \
assert torch.__version__.startswith('${TORCH_VERSION}'), f'torch 版本不符: {torch.__version__}'; \
assert 'cu128' in torch.__version__, f'CUDA 构建不符: {torch.__version__}'; \
assert importlib.util.find_spec('xformers') is None, 'xformers 溜回来了'; \
import transformers, huggingface_hub; \
assert int(transformers.__version__.split('.')[0]) < 5, f'transformers 越界: {transformers.__version__}'; \
assert int(huggingface_hub.__version__.split('.')[0]) < 1, f'huggingface-hub 越界: {huggingface_hub.__version__}'; \
print('OK torch', torch.__version__, '| transformers', transformers.__version__, '| hf-hub', huggingface_hub.__version__)"

# 构建期冒烟:真的把 ComfyUI 起一次(会导入完整节点图),让「启动就炸」
# 在构建阶段暴露,而不是变成线上 worker 一个误导性的 "server not reachable"。
# 走 CPU,不需要 GPU。
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# ------------------------------------------------------------------------------
# Stage 2: downloader —— 只下模型
#
# 模型烤进镜像,不用 network volume:volume 绑单个数据中心,会切窄可调度 GPU 池,
# 而 serverless 上「没资源」表现为排队而非报错 —— 正是要逃离的失败模式。
# ------------------------------------------------------------------------------
FROM base AS downloader

# SDXL Turbo:6.94GB 单文件,1-4 步出图。体积够大能测出真实镜像拉取速率,
# 又不至于让首次构建逼近 30 分钟上限。
ARG SMOKE_MODEL_URL=https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors

RUN mkdir -p /comfyui/models/checkpoints \
    && wget -q --progress=dot:giga -O \
         /comfyui/models/checkpoints/sd_xl_turbo_1.0_fp16.safetensors \
         "${SMOKE_MODEL_URL}" \
    && ls -lh /comfyui/models/checkpoints/

# ------------------------------------------------------------------------------
# Stage 3: final —— 模型 + 自己的代码
# ------------------------------------------------------------------------------
FROM base AS final

COPY --from=downloader /comfyui/models /comfyui/models

WORKDIR /
COPY worker/handler.py /handler.py
COPY worker/src/start.sh /start.sh
COPY worker/src/gpu_check.py /gpu_check.py
COPY worker/scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /start.sh /usr/local/bin/comfy-node-install

# 启动路径的构建期断言:start.sh 是 CMD,构建期从来不会执行它 ——
# 里面的语法错误(以及它调用的 python 文件的语法错误)会以「线上 worker
# 无限 crash loop、job 永远排队」的形态出现,而不是构建失败。在这里钉死。
RUN bash -n /start.sh \
    && python -m py_compile /gpu_check.py /handler.py \
    && echo "start path syntax OK"

# 末层瘦身:strip C 扩展调试符号 + 清 __pycache__,省几百 MB。
# 镜像每小一点,撞到新宿主机时的冷启动税就少一点。
RUN find /opt/venv /comfyui -depth -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv/lib -type f \( -name '*.so' -o -name '*.so.*' \) \
         -exec strip --strip-unneeded {} + 2>/dev/null || true \
    && rm -rf /root/.cache /tmp/*

CMD ["/start.sh"]
