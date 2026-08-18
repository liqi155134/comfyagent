# 诊断专用极小镜像(~200MB)。
#
# 目的:分离最后两个变量 —— H100 拉 Docker Hub 小镜像正常、拉 GHCR 14GB 镜像卡死,
# 中间差着「registry 来源」和「镜像体积」两个变量。这个镜像在 GHCR 上但很小,
# 能一刀切开:它起得来 = GHCR 没问题、是体积;起不来 = GHCR 链路有问题。
FROM python:3.12-slim
RUN pip install --no-cache-dir runpod
COPY docker/probe_handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
