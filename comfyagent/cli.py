"""comfyagent CLI —— 给人用,也给任何能跑 shell 的 agent 用。

设计取向:
  * 参数**按工作流动态生成**,所以 `run <id> -h` 直接列出该工作流真实接受的
    参数、默认值和取值范围。agent 不确定时能自己查,不必回来问人。
  * `--json` 让输出可解析;不带则输出人类可读文本。
  * 提交与取结果分离。视频类工作流跑十几分钟,阻塞等待不现实;
    `--poll N` 只是「顺便等一会儿」,超时就返回 job_id,之后 `query` 捡回来。
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

from .core import config, store
from .core.client import RunpodClient, RunpodError
from .core.registry import ParamError, WorkflowError, load_workflows

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"
_TERMINAL = {"completed", "failed", "cancelled"}


def _emit(data, as_json, text_fn):
    print(json.dumps(data, ensure_ascii=False, indent=2) if as_json else text_fn(data))


def _save_outputs(outputs, directory):
    """把产物落到本地,返回文件路径列表。

    agent 拿到本地路径比拿到 base64 有用得多 —— 它能直接把路径交给下一步,
    不必自己解码再落盘。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for item in outputs or []:
        name = item.get("filename") or "output.png"
        target = directory / name
        n = 1
        while target.exists():
            target = directory / f"{target.stem}_{n}{target.suffix}"
            n += 1
        kind = item.get("type")
        if kind == "base64":
            target.write_bytes(base64.b64decode(item["data"]))
        elif kind in ("s3_url", "url", "r2_url"):
            with urllib.request.urlopen(item["data"], timeout=300) as r:
                target.write_bytes(r.read())
        else:
            continue
        saved.append(str(target))
    return saved


# ---------------------------------------------------------------- list
def cmd_list(args):
    wfs = load_workflows(WORKFLOW_DIR)
    eps = config.load()
    rows = []
    for w in wfs.values():
        entry = eps.get(w.endpoint) or {}
        rows.append({
            "id": w.id,
            "title": w.title,
            "description": w.description,
            "endpoint": w.endpoint,
            "deployed": bool(entry.get("endpoint_id")),
            "params": {
                n: {k: v for k, v in
                    {"type": p.type, "required": p.required, "default": p.default,
                     "enum": p.enum, "min": p.min, "max": p.max,
                     "description": p.description}.items() if v not in (None, False, "")}
                for n, p in w.params.items()
            },
        })

    def text(rows):
        out = []
        for r in rows:
            mark = "✓" if r["deployed"] else "✗ 未部署"
            out.append(f"{r['id']:20} [{mark}]  {r['title']}")
            if r["description"]:
                out.append(f"  {r['description'].splitlines()[0]}")
            req = [n for n, p in r["params"].items() if p.get("required")]
            opt = [n for n in r["params"] if n not in req]
            out.append(f"  必填: {', '.join(req) or '无'}   可选: {', '.join(opt) or '无'}")
        return "\n".join(out) or "(workflows/ 下没有工作流)"

    _emit(rows, args.json, text)


# ---------------------------------------------------------------- run
def cmd_run(args, workflow, given):
    client = RunpodClient()
    endpoint_id = config.resolve(workflow.endpoint)
    graph, resolved = workflow.build(given)

    job_id = client.submit(endpoint_id, {"workflow": graph})
    store.record(job_id, workflow.id, endpoint_id, resolved)

    result = {"job_id": job_id, "workflow": workflow.id, "status": "queued",
              "params": resolved}

    if args.poll:
        deadline = time.time() + args.poll
        while time.time() < deadline:
            time.sleep(3)
            s = client.status(endpoint_id, job_id)
            if s["status"] in _TERMINAL:
                result.update(status=s["status"], error=s.get("error"),
                              execution_ms=s.get("execution_ms"))
                outputs = (s.get("output") or {}).get("images")
                store.update(job_id, s["status"], outputs, s.get("error"))
                if outputs and args.download_dir:
                    result["files"] = _save_outputs(outputs, args.download_dir)
                break
        else:
            result["status"] = "querying"
            result["hint"] = f"仍在运行。稍后用 `comfyagent query {job_id}` 取结果。"

    _emit(result, args.json, lambda r: "\n".join(
        [f"job_id: {r['job_id']}", f"状态:   {r['status']}"]
        + ([f"文件:   {', '.join(r['files'])}"] if r.get("files") else [])
        + ([f"错误:   {r['error']}"] if r.get("error") else [])
        + ([r["hint"]] if r.get("hint") else [])))
    return 0 if result["status"] in ("completed", "queued", "querying") else 1


# ---------------------------------------------------------------- query
def cmd_query(args):
    rec = store.get(args.job_id)
    client = RunpodClient()
    if rec:
        endpoint_id = rec["endpoint_id"]
    else:
        # 非本机提交的 job(如直接调 API):遍历登记表逐个试查
        endpoint_id = None
        for entry in config.load().values():
            eid = entry.get("endpoint_id")
            if not eid:
                continue
            try:
                if client.status(eid, args.job_id).get("raw_status"):
                    endpoint_id = eid
                    break
            except Exception:
                continue
        if not endpoint_id:
            print(f"找不到 job {args.job_id}(任务库无记录,登记的 endpoint 也查不到)",
                  file=sys.stderr)
            return 1
    s = client.status(endpoint_id, args.job_id)
    outputs = (s.get("output") or {}).get("images")
    if rec:
        store.update(args.job_id, s["status"], outputs, s.get("error"))

    result = {"job_id": args.job_id,
              "workflow": rec["workflow_id"] if rec else None,
              "status": s["status"], "error": s.get("error"),
              "execution_ms": s.get("execution_ms"),
              "params": json.loads(rec["params"]) if rec else None}
    if outputs and args.download_dir:
        result["files"] = _save_outputs(outputs, args.download_dir)
    elif outputs:
        result["output_count"] = len(outputs)

    _emit(result, args.json, lambda r: "\n".join(
        [f"状态: {r['status']}"]
        + ([f"文件: {', '.join(r['files'])}"] if r.get("files") else [])
        + ([f"产物: {r['output_count']} 个(加 --download-dir 落盘)"]
           if r.get("output_count") else [])
        + ([f"错误: {r['error']}"] if r.get("error") else [])))
    return 0 if s["status"] in ("completed", "queued", "running") else 1


# ---------------------------------------------------------------- jobs
def cmd_jobs(args):
    rows = store.recent(args.limit, args.status)
    _emit(rows, args.json, lambda rs: "\n".join(
        f"{r['job_id'][:24]:26} {r['status']:10} {r['workflow_id']}" for r in rs)
        or "(还没有任务记录)")


# ---------------------------------------------------------------- endpoints
def cmd_endpoints(args):
    client = RunpodClient()
    live = {e["id"]: e for e in client.list_endpoints()}
    rows = []
    for name, entry in config.load().items():
        eid = entry.get("endpoint_id")
        e = live.get(eid)
        rows.append({"name": name, "endpoint_id": eid,
                     "image": entry.get("image"),
                     "exists": bool(e),
                     "workers_min": (e or {}).get("workersMin"),
                     "workers_max": (e or {}).get("workersMax")})
    _emit(rows, args.json, lambda rs: "\n".join(
        f"{r['name']:12} {r['endpoint_id'] or '-':22} "
        f"{'在线' if r['exists'] else '不存在'}  {r['image'] or ''}" for r in rs)
        or "(还没有登记任何部署单元)")


def build_parser():
    p = argparse.ArgumentParser(
        prog="comfyagent",
        description="把 ComfyUI 工作流封装成可直接调用的接口(RunPod serverless)")
    p.add_argument("--json", action="store_true", help="输出 JSON(便于程序解析)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出可用工作流及其参数")

    r = sub.add_parser("run", help="提交一个工作流")
    r.add_argument("workflow", help="工作流 id(见 comfyagent list)")
    r.add_argument("--poll", type=int, metavar="秒",
                   help="提交后最多等这么久;超时则返回 job_id 稍后再查")
    r.add_argument("--download-dir", metavar="目录", help="把产物下载到这里")

    q = sub.add_parser("query", help="查询任务结果")
    q.add_argument("job_id")
    q.add_argument("--download-dir", metavar="目录")

    j = sub.add_parser("jobs", help="列出最近的任务")
    j.add_argument("--status", choices=["queued", "running", "completed",
                                        "failed", "cancelled"])
    j.add_argument("--limit", type=int, default=20)

    sub.add_parser("endpoints", help="查看部署单元状态")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # run 的参数取决于具体工作流,所以先摸出 workflow id,再把它声明的参数
    # 动态挂上去 —— 这样 `run <id> -h` 才能列出真实可用的参数。
    if "run" in argv:
        try:
            wfs = load_workflows(WORKFLOW_DIR)
        except WorkflowError as e:
            print(f"工作流定义有误: {e}", file=sys.stderr)
            return 2
        idx = argv.index("run")
        wf_id = next((a for a in argv[idx + 1:] if not a.startswith("-")), None)
        if wf_id in wfs:
            wf = wfs[wf_id]
            sub_run = parser._subparsers._group_actions[0].choices["run"]
            for name, spec in wf.params.items():
                kw = {"help": f"{spec.description} (默认 {spec.default})".strip(),
                      "required": spec.required}
                if spec.type == "integer":
                    kw["type"] = int
                elif spec.type == "number":
                    kw["type"] = float
                elif spec.type == "boolean":
                    # agent 会传 --xx true/false 字符串;这里转成真 bool,
                    # coerce 层保持严格(API 直调时不吞类型错误)。
                    kw["type"] = lambda s: s.lower() in ("true", "1", "yes")
                    kw["metavar"] = "true|false"
                if spec.enum:
                    kw["choices"] = spec.enum
                sub_run.add_argument(f"--{name.replace('_', '-')}", **kw)
            args = parser.parse_args(argv)
            given = {n: getattr(args, n.replace("-", "_"))
                     for n in wf.params
                     if getattr(args, n.replace("-", "_"), None) is not None}
            try:
                return cmd_run(args, wf, given)
            except (ParamError, KeyError) as e:
                print(f"参数错误: {e}", file=sys.stderr)
                return 2
            except RunpodError as e:
                print(f"RunPod 调用失败: {e}", file=sys.stderr)
                return 3
        elif wf_id:
            print(f"未知工作流 {wf_id!r}。可用: {sorted(wfs)}", file=sys.stderr)
            return 2

    args = parser.parse_args(argv)
    try:
        return {"list": cmd_list, "query": cmd_query, "jobs": cmd_jobs,
                "endpoints": cmd_endpoints}[args.cmd](args) or 0
    except RunpodError as e:
        print(f"RunPod 调用失败: {e}", file=sys.stderr)
        return 3
    except (WorkflowError, KeyError) as e:
        print(f"配置错误: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
