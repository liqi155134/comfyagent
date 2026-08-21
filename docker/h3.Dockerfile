# ==============================================================================
# MiniMax H3 环境镜像 —— 不含模型(44.4GB 权重走 RunPod model caching)。
#
# 与 smoke 镜像的三个关键差异:
#   1. torch cu130(PyPI 默认构建,依赖 nvidia-*-cu13):ComfyUI 的
#      comfy/quant_ops.py 在 torch CUDA < 13 时会整个禁用 comfy-kitchen 的
#      CUDA 后端 —— H3 的 DiT(int8_convrot)和 TE(nvfp4_awq)的融合量化
#      算子全部回落 eager。cu130 是这两个模型的性能前提,不是可选项。
#      代价:要求宿主机驱动 >= 580,endpoint 侧用 allowedCudaVersions=["13.0"]
#      锁调度(H100 宿主机实测 580.126.09)。
#   2. SageAttention v2.2.0:官方无 Linux wheel,builder 阶段用 CUDA 13 devel
#      镜像从源码交叉编译(TORCH_CUDA_ARCH_LIST 指定 sm90,无 GPU 也能编)。
#   3. 无模型层:镜像 ~7GB,GHCR 单层上限、构建时长都不再是约束;
#      模型由 start.sh 按 MODEL_NAME/MODEL_REVISION 从缓存卷 symlink 进来。
# ==============================================================================

ARG CUDA_IMAGE_TAG=13.0.1
ARG COMFYUI_REF=v0.33.1
ARG TORCH_VERSION=2.13.0
# ⚠️ 不要用 v2.2.0 tag:它带着 PR #218 的 sm90 wrapper bug —— custom op 往
# out tensor 写结果却没声明 mutates_args,torch functionalization 把写入当
# dead store 丢弃,H100 上输出规则网格马赛克且日志全绿(上游 #288/#320,
# 本项目与 Modal 侧各自踩过一次)。pin 到 2025-12-22 之后的 main commit。
ARG SAGE_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
ARG PYTHON_VERSION=3.12
# H100/H200 都是 sm90;要兼容 4090/L40S 再加 "8.9"
ARG SAGE_CUDA_ARCHS=9.0

# ------------------------------------------------------------------------------
# Stage 1: sage-builder —— 编译 SageAttention(需要 nvcc,用 devel 镜像)
# ------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_IMAGE_TAG}-devel-ubuntu24.04 AS sage-builder

ARG TORCH_VERSION
ARG SAGE_REF
ARG PYTHON_VERSION
ARG SAGE_CUDA_ARCHS

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# torch 装 PyPI 默认构建(cu13)。sage 编译要链接 torch 头文件与库。
RUN uv pip install --no-cache "torch==${TORCH_VERSION}" setuptools wheel ninja

# 交叉编译:无 GPU 环境必须显式给 TORCH_CUDA_ARCH_LIST,
# 否则 setup.py 会尝试探测本机 GPU 然后失败。
RUN mkdir /sage && cd /sage \
    && git init -q && git remote add origin https://github.com/thu-ml/SageAttention.git \
    && git fetch -q --depth 1 origin "${SAGE_REF}" \
    && git checkout -q FETCH_HEAD \
    && TORCH_CUDA_ARCH_LIST="${SAGE_CUDA_ARCHS}" \
       MAX_JOBS=2 NVCC_APPEND_FLAGS="--threads 2" \
       python setup.py bdist_wheel \
    && ls -lh /sage/dist/

# ------------------------------------------------------------------------------
# Stage 2: final —— 运行时环境(无模型)
# ------------------------------------------------------------------------------
# wheel 导出专用微型 stage:target=sage-wheel-export 的 local output 只含
# wheel 文件(几十 MB);直接 target=sage-builder 会导出整个 rootfs(十几 GB),
# 曾把独立提取 workflow 拖到 GHA 6h 超时。
FROM scratch AS sage-wheel-export
COPY --from=sage-builder /sage/dist/*.whl /

FROM nvidia/cuda:${CUDA_IMAGE_TAG}-cudnn-runtime-ubuntu24.04 AS final

ARG COMFYUI_REF
ARG TORCH_VERSION
ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# gcc 不是可选项:triton 在推理期 JIT 编译 kernel(H3 量化算子路径)需要
# 宿主 C 编译器,缺了会在第一次采样时报 "Failed to find C compiler"(实测)。
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates \
      build-essential \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      ffmpeg \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# torch 先装(PyPI 默认即 cu13 构建),ComfyUI requirements 里的裸 torch
# 就不会再动它。transformers / huggingface-hub 上界 pin 与 requirements
# 同层安装,理由同 smoke 镜像。
RUN git clone --depth 1 --branch "${COMFYUI_REF}" \
      https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui \
    && git rev-parse HEAD > /comfyui/.git_sha_pinned \
    && uv pip install --no-cache "torch==${TORCH_VERSION}" torchvision torchaudio \
    && uv pip install --no-cache -r requirements.txt \
    && uv pip install --no-cache "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
    && uv pip uninstall xformers 2>/dev/null || true \
    && rm -rf /comfyui/.git /comfyui/.github /comfyui/tests /root/.cache /tmp/*

# handler 运行期依赖 + sage wheel
COPY --from=sage-builder /sage/dist/*.whl /tmp/wheels/
RUN uv pip install --no-cache runpod requests websocket-client boto3 hf_transfer /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# sage int32 指针溢出修复(2026-08-20 定论,见 docs/sage-sm90-issue-draft.md):
# 量化 kernel 的行偏移 offs_n * stride_in 在 int32 域相乘,fused QKV 布局
# (Q seq-stride=21504)下行号 > 2^31/21504 ≈ 99865 即 wrap 负 → 尾帧塌坏 /
# illegal memory access。构建期打补丁(此前是启动时 wget,依赖外网且 FlashBoot
# 快照行为易混淆)。上游合入后删除本步骤即可。
COPY scripts/patch_sage_int64.py /tmp/patch_sage_int64.py
RUN python /tmp/patch_sage_int64.py && rm /tmp/patch_sage_int64.py

# 构建期断言:cu13 / comfy-kitchen CUDA 后端未被禁 / sage 可导入 / int64 补丁在位。
# comfy-kitchen 的禁用逻辑只看 torch.version.cuda(静态值),无 GPU 也能验。
RUN python -c "\
import torch, importlib.util; \
assert torch.__version__.startswith('${TORCH_VERSION}'), f'torch 版本不符: {torch.__version__}'; \
cu = tuple(map(int, str(torch.version.cuda).split('.'))); \
assert cu >= (13,), f'CUDA 构建不是 cu13: {torch.version.cuda} —— comfy-kitchen CUDA 后端会被禁用'; \
assert importlib.util.find_spec('sageattention') is not None, 'sageattention 没装上'; \
import sageattention, pathlib; \
_qp = pathlib.Path(sageattention.__file__).parent / 'triton' / 'quant_per_thread.py'; \
assert _qp.read_text().count('.to(tl.int64)') >= 4, 'sage int64 补丁没生效'; \
assert importlib.util.find_spec('xformers') is None, 'xformers 溜回来了'; \
import comfy_kitchen; \
import importlib.metadata as md; \
import transformers, huggingface_hub; \
assert int(transformers.__version__.split('.')[0]) < 5; \
assert int(huggingface_hub.__version__.split('.')[0]) < 1; \
print('OK torch', torch.__version__, '| cuda', torch.version.cuda, \
      '| comfy_kitchen', md.version('comfy-kitchen'), '| transformers', transformers.__version__)"

# 构建期冒烟:CPU 起一次 ComfyUI 完整节点图(不带 sage,sage 需要 GPU)。
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

WORKDIR /
COPY worker/handler.py /handler.py
COPY worker/src/start.sh /start.sh
COPY worker/src/gpu_check.py /gpu_check.py
COPY worker/src/fetch_models.py /fetch_models.py
COPY worker/scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /start.sh /usr/local/bin/comfy-node-install

# 启动路径构建期断言(理由见 smoke.Dockerfile)
RUN bash -n /start.sh \
    && python -m py_compile /gpu_check.py /fetch_models.py /handler.py \
    && echo "start path syntax OK"

RUN find /opt/venv /comfyui -depth -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /root/.cache /tmp/*

CMD ["/start.sh"]
