# comfyagent 使用指南(面向 AI agent 与人类)

把注册好的 ComfyUI 工作流(RunPod serverless GPU)封装成一条命令 / 一次工具调用。
当前生产工作流:**MiniMax H3 文生视频(带音频)** 与 SDXL Turbo 文生图(冒烟用)。

## 前置条件(一次性)

| 项 | 位置 | 说明 |
|---|---|---|
| RunPod API key | `~/.config/comfyagent/env` | `RUNPOD_API_KEY=...`,权限 600 |
| R2 凭据 | `~/.config/comfyagent/r2_env` | `BUCKET_*` 四件套,deploy 时注入 endpoint |
| 部署声明 | `deployments.yaml`(仓库内) | 端点配置的事实来源,`deploy` 命令读它 |
| 部署单元登记表 | `~/.config/comfyagent/endpoints.json` | 逻辑名 → endpoint_id,由 `deploy` 自动写 |
| 依赖 | `requirements.lock` | `pip install -r requirements.lock && pip install -e .` |
| 运行目录 | `/workspace/comfyagent` | CLI 以 `python3 -m comfyagent.cli` 运行 |

**从零重建生产环境**(换台机器 / 端点被误删时):

```bash
python3 -m comfyagent.cli deploy --dry-run   # 先看计划
python3 -m comfyagent.cli deploy             # 全部创建/更新(幂等,可反复跑)
python3 -m comfyagent.cli deploy h3          # 只处理一个
```

幂等:同名资源存在就 PATCH,不会造出第二套。`workers_min>0`(持续计费)会被
拒绝执行,需显式 `--allow-billing`。

调用前 `set -a; source ~/.config/comfyagent/env; set +a`(或让 shell 已带该 env)。

## CLI

```bash
# 看有哪些工作流、各要什么参数
python3 -m comfyagent.cli list

# 某个工作流的完整参数说明(动态生成)
python3 -m comfyagent.cli run h3_t2v -h

# 生成一条 15 秒 720p 级视频(8 步 LoRA 加速,约 4 分钟 / $0.34)
python3 -m comfyagent.cli --json run h3_t2v \
  --prompt "A red fox running through a misty bamboo forest at dawn. Audio: birdsong." \
  --duration 15 --megapixels 0.9 --enable-lora true \
  --poll 900 --download-dir ./out

# 提交后不等(拿 job_id 稍后查)
python3 -m comfyagent.cli --json run h3_t2v --prompt "..." 

# 查任务 / 下载产物(job 非本机提交也能查,会遍历登记的 endpoint)
python3 -m comfyagent.cli --json query <job_id> --download-dir ./out

# 最近任务(含 seed,复现用)
python3 -m comfyagent.cli jobs
```

约定:`--json` 输出机器可读结构;产物为 R2 直链(7 天有效)时 `query --download-dir`
会自动下载;**seed 每次随机但记录在结果与任务库里**,复现必须显式传回。

## MCP(agent 首选入口)

```bash
claude mcp add comfyagent -- python3 -m comfyagent.mcp_server   # 在 /workspace/comfyagent 下
```

| 工具 | 用途 |
|---|---|
| `list_workflows` | 工作流清单 + 每个参数的类型/默认/范围,**先调这个** |
| `run_workflow(workflow_id, params, wait_seconds=0)` | 提交;`wait_seconds>0` 顺便等待,完成直接带产物 URL |
| `query_job(job_id)` | 状态 + 产物 URL |
| `recent_jobs(limit)` | 回溯任务与 seed |

## h3_i2v(图生视频)要点

与 h3_t2v 参数一致,另加必填 `image`(首帧图片本地路径,png/jpg/webp,自动
base64 上传)。默认走 `h3`(**sage int64 修复版**——曾经的尾帧塌坏根因是
Triton 量化 kernel int32 指针溢出,已修并双重验证,见 docs/h3.md),同规格比
t2v 慢:15s/0.9MP/8 步 LoRA ≈ 258s($0.35)。高忠实度需求(对比素材等)可把
workflow endpoint 临时改 `h3_ck`(comfy-kitchen,首帧 MAD 1.77/255 vs sage
6.80,慢 ~49%)。prompt 的场景描述必须与输入图一致,否则画面突变。

```bash
python3 -m comfyagent.cli --json run h3_i2v \
  --image /path/to/first_frame.png \
  --prompt "..." --duration 15 --megapixels 0.9 --enable-lora true \
  --poll 900 --download-dir ./out
```

## h3_t2v 参数速查

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | string | 必填 | 支持分镜脚本式写法(时间轴/运镜/音频提示),英文更稳,避免要求文字/logo |
| `duration` | number | 5 | 秒,1-15;按 24fps 折算成合法帧数 |
| `aspect_ratio` | enum | `16:9 (Widescreen)` | 8 档,值须与枚举完全一致 |
| `megapixels` | number | 0.4 | 0.4→848×480,0.9→1280×736;越高越慢 |
| `enable_lora` | boolean | false | true = 8 步 Lightning 加速(≈2.1x,质感略柔);false = 20 步原生 |
| `seed` | integer | 随机 | 复现时显式传入 |

## 耗时与成本参考(H100,实测 2026-08)

全部为 `h3` 端点(sage int64 修复版)实测,15s/0.9MP 档尾帧均经帧间差扫尾验证:

| 配置 | execution | 成本 |
|---|---|---|
| t2v 5s / 0.4MP / 8 步 LoRA | ~40s | ~$0.06 |
| t2v 15s / 0.9MP / 8 步 LoRA | ~253s | ~$0.34 |
| t2v 15s / 0.9MP / 20 步原生 | ~532s | ~$0.71 |
| i2v 15s / 0.9MP / 8 步 LoRA | ~258s | ~$0.35 |
| i2v 15s / 0.9MP / 20 步原生 | ~558s | ~$0.75 |
| i2v 15s / 0.9MP / 8 步 LoRA(h3_ck 备选) | ~384s | ~$0.51 |

20 步 vs 8 步 LoRA:约 2.1x 耗时换更实的细节与更稳的运动;i2v 比同规格 t2v
慢约 5%(多 818 个 keyframe conditioning token)。

**成本按 execution 折算,实际账单更高**:RunPod 对启动、执行、以及缩到 IDLE 前的
idle timeout 都计费(实测冒烟 execution 39.6s → 计费 150.0s)。冷启动税(仅新宿主机
首次):拉镜像 ~2-3 分钟 + 拉 44GB 模型 ~96s;FlashBoot 复用时 <1s。排队 15 分钟以上不是 bug 就是 H100 供给紧张,
用 worker 日志 API 看现场(见 docs/runpod.md)。

## 常见错误与自纠

| 报错 | 处理 |
|---|---|
| `参数 X 需要 boolean,收到 str` | CLI 传 `--xx true/false`;MCP 里传 JSON 布尔 |
| `未知参数 [...] 接受的是 [...]` | 按提示改参数名,或先看 `run <id> -h` / `list_workflows` |
| `部署单元 'h3' 还没登记` | endpoints.json 缺条目,查 `comfyagent endpoints` |
| job 长时间 IN_QUEUE | 看 worker 日志定位(平台侧),别盲目重提 |
| COMPLETED 但无产物 | 大产物必须走 R2(endpoint env 配 BUCKET_*);无 R2 时超限被平台静默丢弃 |
