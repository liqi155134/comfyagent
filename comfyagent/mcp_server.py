"""MCP 入口:把注册过的 ComfyUI 工作流暴露成 agent 可直接调用的工具。

与 CLI 完全共享 core(注册表 / 渲染 / RunPod client / 任务库),这里只是
第二个壳。设计取向:

* run_workflow 默认立即返回 job_id(agent 自己决定轮询节奏),
  wait_seconds > 0 时才顺便等待 —— 与 CLI 的 --poll 语义一致。
* 产物返回 R2 URL(7 天有效),不做 base64 —— agent 拿 URL 比拿字节有用。
* 参数错误的报错文本面向 agent 可自纠(缺什么、收到什么、合法值是什么)。

接入(以 Claude Code 为例):
    claude mcp add comfyagent -- python -m comfyagent.mcp_server
"""

import json
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .core import config, store
from .core.client import RunpodClient
from .core.registry import load_workflows

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"

mcp = MCPServer("comfyagent")


def _workflows():
    return load_workflows(WORKFLOW_DIR)


def _find_endpoint(job_id):
    """job_id → endpoint_id。优先任务库;非本机提交的 job 遍历登记表试查。"""
    rec = store.get(job_id)
    if rec:
        return rec["endpoint_id"]
    client = RunpodClient()
    for entry in config.load().values():
        eid = entry.get("endpoint_id")
        if not eid:
            continue
        try:
            s = client.status(eid, job_id)
            if s.get("raw_status"):
                return eid
        except Exception:
            continue
    return None


def _status_dict(client, endpoint_id, job_id):
    s = client.status(endpoint_id, job_id)
    outputs = (s.get("output") or {}).get("images") or []
    return {
        "job_id": job_id,
        "status": s["status"],
        "error": s.get("error"),
        "execution_ms": s.get("execution_ms"),
        "outputs": [
            {"filename": o.get("filename"), "type": o.get("type"),
             "url": o.get("data") if o.get("type") in ("r2_url", "s3_url", "url") else None}
            for o in outputs if isinstance(o, dict)
        ],
    }


@mcp.tool()
def list_workflows() -> str:
    """列出全部可用的视频/图像生成工作流,含每个参数的类型、默认值、取值范围与说明。调用 run_workflow 前先看这个。"""
    result = []
    for wf in _workflows().values():
        result.append({
            "id": wf.id,
            "title": wf.title,
            "description": wf.description,
            "params": {name: {
                "type": p.type, "required": p.required, "default": p.default,
                "enum": p.enum, "min": p.min, "max": p.max,
                "description": p.description,
            } for name, p in wf.params.items()},
        })
    return json.dumps(result, ensure_ascii=False, indent=1)


@mcp.tool()
def run_workflow(workflow_id: str, params: dict, wait_seconds: int = 0) -> str:
    """提交一个生成任务。params 按 list_workflows 里该工作流的声明传
    (如 {"prompt": "...", "duration": 5, "enable_lora": true});seed 省略则随机、
    结果里会记录实际值以便复现。wait_seconds=0 立即返回 job_id(之后用 query_job 查);
    >0 则最多等这么久,完成的话直接带回产物 URL。"""
    wfs = _workflows()
    if workflow_id not in wfs:
        return json.dumps({"error": f"未知工作流 {workflow_id!r},可用: {sorted(wfs)}"},
                          ensure_ascii=False)
    wf = wfs[workflow_id]
    try:
        payload, resolved = wf.build_payload(params or {})
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    endpoint_id = config.resolve(wf.endpoint)
    client = RunpodClient()
    job_id = client.submit(endpoint_id, payload)
    store.record(job_id, wf.id, endpoint_id, resolved)

    result = {"job_id": job_id, "workflow": wf.id, "params": resolved, "status": "queued"}
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        time.sleep(min(5, max(1, deadline - time.time())))
        s = _status_dict(client, endpoint_id, job_id)
        store.update(job_id, s["status"], s["outputs"] or None, s.get("error"))
        if s["status"] in ("completed", "failed", "cancelled", "timed_out"):
            result.update(s)
            return json.dumps(result, ensure_ascii=False, indent=1)
    if wait_seconds:
        result["hint"] = "未在等待窗口内完成,用 query_job 继续查"
    return json.dumps(result, ensure_ascii=False, indent=1)


@mcp.tool()
def query_job(job_id: str) -> str:
    """查询任务状态;完成时带回产物的下载 URL(R2,7 天有效)。"""
    endpoint_id = _find_endpoint(job_id)
    if not endpoint_id:
        return json.dumps({"error": f"找不到 job {job_id}(任务库无记录,登记的 endpoint 也查不到)"},
                          ensure_ascii=False)
    s = _status_dict(RunpodClient(), endpoint_id, job_id)
    if store.get(job_id):
        store.update(job_id, s["status"], s["outputs"] or None, s.get("error"))
    return json.dumps(s, ensure_ascii=False, indent=1)


@mcp.tool()
def recent_jobs(limit: int = 10) -> str:
    """最近提交的任务列表(job_id / 工作流 / 状态 / 参数),用于回溯或找 seed 复现。"""
    return json.dumps(store.recent(limit), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    mcp.run()
