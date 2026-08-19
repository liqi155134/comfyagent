# comfyagent 使用指南(面向 AI agent 与人类)

把注册好的 ComfyUI 工作流(RunPod serverless GPU)封装成一条命令 / 一次工具调用。
当前生产工作流:**MiniMax H3 文生视频(带音频)** 与 SDXL Turbo 文生图(冒烟用)。

## 前置条件(一次性)

| 项 | 位置 | 说明 |
|---|---|---|
| RunPod API key | `~/.config/comfyagent/env` | `RUNPOD_API_KEY=...`,权限 600 |
| 部署单元登记表 | `~/.config/comfyagent/endpoints.json` | 逻辑名 → endpoint_id;重建 endpoint 只改这里 |
| 运行目录 | `/workspace/comfyagent` | CLI 以 `python3 -m comfyagent.cli` 运行 |

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

| 配置 | execution | 成本 |
|---|---|---|
| 5s / 0.4MP / 8 步 LoRA | ~42s | ~$0.06 |
| 15s / 0.9MP / 8 步 LoRA | ~253s | ~$0.34 |
| 15s / 0.9MP / 20 步原生 | ~529s | ~$0.70 |

冷启动税(仅新宿主机首次):拉镜像 ~2-3 分钟(不计费)+ 拉 44GB 模型 ~96s(计费
$0.13);FlashBoot 复用时 <1s。排队 15 分钟以上不是 bug 就是 H100 供给紧张,
用 worker 日志 API 看现场(见 docs/runpod.md)。

## 常见错误与自纠

| 报错 | 处理 |
|---|---|
| `参数 X 需要 boolean,收到 str` | CLI 传 `--xx true/false`;MCP 里传 JSON 布尔 |
| `未知参数 [...] 接受的是 [...]` | 按提示改参数名,或先看 `run <id> -h` / `list_workflows` |
| `部署单元 'h3' 还没登记` | endpoints.json 缺条目,查 `comfyagent endpoints` |
| job 长时间 IN_QUEUE | 看 worker 日志定位(平台侧),别盲目重提 |
| COMPLETED 但无产物 | 大产物必须走 R2(endpoint env 配 BUCKET_*);无 R2 时超限被平台静默丢弃 |
