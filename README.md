# comfyagent

把 ComfyUI 工作流封装成 **AI agent 可直接调用**的接口。

```
Agent (Claude Code / Codex / ...)
    ↓  CLI 或 MCP
comfyagent
    ↓  提交工作流 + 参数
RunPod serverless (ComfyUI + 烤在镜像里的模型)
    ↓
产物(本地文件 / R2 URL)
```

agent 只认 **workflow id + 参数**,不需要知道 ComfyUI 存在,也不拼节点图。

## 设计要点

- **模型烤进镜像,不用 network volume** —— volume 绑单个数据中心,会切窄可调度
  GPU 池,而 serverless 上「没资源」表现为排队而非报错,极难归因。
- **一个镜像可承载多个工作流** —— 共享同一批权重的能力(例:原版 / LoRA 加速版)
  共用一个 endpoint,避免把 40GB 权重和冷启动代价付两遍。
- **失败要大声** —— 兜底逻辑绝不制造假终态;构建期能钉死的事实在构建期钉死。

## 目录

| 路径 | 说明 |
|---|---|
| `worker/` | RunPod serverless worker(handler + 启动脚本) |
| `docker/` | 镜像定义,一个部署目标一份 |
| `workflows/` | 预注册工作流:节点图 JSON + 参数声明 |
| `comfyagent/` | CLI 与 MCP server 的共同核心 |
| `tests/` | 回归测试 |

## 状态

验证阶段 —— 正在用 smoke 镜像(SDXL Turbo)跑通构建 → 部署 → 调用全链路。
