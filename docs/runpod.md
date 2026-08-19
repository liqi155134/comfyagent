# RunPod 实战笔记

全部条目基于实测或官方文档核实,标注日期。文档索引:https://docs.runpod.io/llms.txt

## API 面地图(三套并存,极易混淆)

| API 面 | 地址 | 用途 |
|---|---|---|
| Job 调用面 | `api.runpod.ai/v2/{endpointId}/...` | run / runsync / status / health / cancel / **purge-queue**(⚠️ 域名是 `.ai`,用 `.io` 会 404) |
| 资源管理 v1 | `rest.runpod.io/v1` | endpoints / templates CRUD |
| 资源管理 v2 | `api.runpod.io/v2/serverless/...` | worker 列表 + **worker 日志(SSE)** —— v1 / GraphQL / runpodctl 都没有的能力 |
| GraphQL(旧) | `api.runpod.io/graphql` | endpoint gpuIds 回读等;**必须带自定义 User-Agent**,否则 Cloudflare 403 error 1010 |

## Worker 日志 API(2026-08-18 实测可用)

```
GET https://api.runpod.io/v2/serverless/{ep}/workers                     # worker 列表
GET https://api.runpod.io/v2/serverless/{ep}/workers/{workerId}/logs\
    ?source=container&tail=100                                           # SSE 流
```

Bearer 认证同 REST。endpoint 日志保留 90 天,worker 日志仅存活期。
runpodctl v2.9.0 无 logs 命令、GraphQL 无对应字段——只有这套 v2 有。
**job 卡 IN_QUEUE 时第一动作就是抓 worker 日志,不要按表象猜根因**(见下)。

## 踩过的坑(全部实测,2026-08)

1. **GPU 档位上限 3**:数的是档位(ADA_80_PRO 等)不是型号,传型号会被展开成档位再计数。REST **静默接受**非法配置,症状是 worker 永远 initializing。防线:创建后 GraphQL 回读 gpuIds 断言(`client.assert_gpu_tiers_ok`)。
2. **crash loop 在 health API 显示为 idle/ready**:容器快速崩溃重启时,health 的 worker 状态机完全失真。曾因此先后误判为镜像体积 / GHCR 链路 / GPU 容量,实际是 start.sh 一个 bash 引号事故(内联 `python -c "…"` 的注释里含双引号,字符串被切断 → SyntaxError → 预检 exit 1 → 无限重启)。修复:预检抽成独立 `gpu_check.py` + 构建期 `bash -n` / `py_compile` 断言,因为 **CMD 在构建期从不执行,启动路径的语法错误只会在线上爆炸**。
3. **REST 建 template 默认 `startJupyter/startSsh=true` 且 v1 schema 不接受显式关闭**(只回显不可写)。serverless 下实测无害,但 template 一律显式设 `dockerStartCmd`,不赌平台对空启动命令的处理。
4. **`dockerStartCmd` 覆盖 = 免重建的容器内探针**:同一镜像 + 新 template 覆盖启动命令为内联诊断 handler,job 返回值携带容器内任意命令的 stdout/stderr。在日志 API 之外多一条取证通道,也可用于 A/B 隔离"镜像内容"与"启动脚本"。
5. GHCR public 镜像 RunPod 直接可拉,无需 registry 凭据;单层压缩后 10GB 上限,模型层压缩率实测 91.2%(20GB 单文件模型必超限)。
6. RunPod GitHub 构建:30 分钟 docker build 硬上限(smoke 镜像 26m34s 已占 88%)、镜像总量 ≤80GB、构建期无 GPU。

## Model caching(2026-08-19 结论:小模型可用,44GB 级实测卡死)

**大 repo 实测失败**:44.4GB 转存 repo 配上 modelName 后,7 个 worker(两个 DC)
全部停在 system 日志 "image ready, model pending download",3 小时零进展后放弃。
对照实验排除了链路因素:同宿主机 curl 拉 HF 单流 80MB/s(官方 LFS 与 Xet 转存
repo 速度相同)—— 是 RunPod 平台下载服务自身卡死,用户侧无法干预、无日志。

**部署红线:endpoint 配了 modelName 而模型没下完时,worker 被扣住不启动容器**,
镜像内的任何降级逻辑都没机会执行。在 RunPod 修好该服务前,生产 endpoint 不配
modelName,模型分发走 worker 直拉(见 start.sh 降级链)或全量镜像。

以下为机制记录(tiny 模型验证过,API/路径信息仍有效):


- **API 配置字段(三个公开 API 面均无文档,错误消息探测法逼出)**:GraphQL
  `saveEndpoint` mutation 的 **`modelName`** 字段(写入侧存在、查询侧不可读;
  v1/v2 REST 均 422 拒绝同名键)。已实测写入生效。
- 实测(4090,tiny 模型):worker 环境里被自动注入 **`MODEL_NAME`** 与
  **`MODEL_REVISION`**(snapshot commit hash)两个 env;整个 HF repo 下载到
  `/runpod-volume/huggingface-cache/hub/models--{org}--{name}/snapshots/{MODEL_REVISION}/`
  —— 启动脚本可直接用这两个 env 拼出确定性路径,不需要读 `refs/main`。
- **模型下载期间不计费**;调度优先落在已有缓存的宿主机
- 限制:每 endpoint 一个模型(= 一个 HF repo,repo 内多文件都会下);仅 HF 来源
- **对 H3 的意义**:镜像只装 ComfyUI 环境(~7GB),40GB 权重传自建 HF repo 走缓存
  → 同时绕开 GHCR 10GB 层上限与 GitHub 构建 30 分钟上限,改 handler 不再动模型分发
- 代价:ComfyUI 需把缓存路径接进 models 目录(启动时按 `MODEL_NAME`/`MODEL_REVISION` symlink)
- 待实测:40GB 级真实下载耗时、gated model 的 token 传递方式

## 实测数字(H100 = ADA_80_PRO 档位,$4.79/hr,2026-08-18)

| 场景 | delay |
|---|---|
| CPU + Docker Hub 小镜像 | 4.6s |
| H100 + Docker Hub 小镜像(首拉) | 60s |
| H100 + GHCR 200MB(首拉) | 80s |
| H100 + GHCR 14GB(宿主机已有镜像缓存) | 27s |
| H100 + GHCR 14GB(首拉,含容量等待/throttle,两次实测一致) | **~10 min**(615s / 590s) |
| H100 + FlashBoot 热启动(同 endpoint 连续请求) | **0.9s** |

execution(SDXL Turbo 512² 1 步):冷 13.4s(含 6.94GB checkpoint 首载)/ 热 **2.0s**;
热启动端到端(CLI 提交→出图→下载到本地)墙钟 6.6s。

启动命令:**空启动命令(镜像 CMD)与显式 dockerStartCmd 均实测可用**;
此前 worker 全部卡死的唯一根因是 start.sh 内联 python 的引号事故,与启动命令形态无关。
规范仍建议显式 dockerStartCmd(不赌平台对空命令的处理,且它是免重建诊断的入口)。

smoke 镜像:14.08GB / 最大单层 6.33GB / GHA 构建 26m34s。
A/B/C 三组实验(两次 10 分钟冷启动 + 3 张图 + model caching 探测)合计消耗 **$0.14**
—— RunPod 对冷启动/排队时段基本不计费,只计 running 秒数。

## 官方 agent 工具(2026-08 文档,未接入)

- skills plugin:`npx skills add runpod/runpod-plugins-official`(router + runpodctl + Flash + MCP)
- API MCP:`@runpod/mcp-server`(REST 包装,RUNPOD_API_KEY);docs MCP:`https://docs.runpod.io/mcp`(免认证)
- 本项目取舍:comfyagent 作为自动化框架直接调 REST 更可控;docs MCP 适合配给 agent 随手查文档
