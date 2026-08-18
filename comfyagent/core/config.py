"""部署单元登记表:工作流声明的 endpoint 名 → 真实的 RunPod 资源。

工作流里写的是逻辑名(endpoint: smoke),不是 RunPod 生成的随机 ID。
好处是重建 endpoint 后只改这张表,工作流定义一行不用动;agent 那侧更是
完全无感 —— 它只认 workflow_id。
"""

import json
from pathlib import Path

PATH = Path.home() / ".config" / "comfyagent" / "endpoints.json"


def load():
    if not PATH.exists():
        return {}
    data = json.loads(PATH.read_text(encoding="utf-8"))
    # 手写短格式 "name": "endpoint_id" 归一化成完整条目,
    # 消费方(set_endpoint / resolve / CLI)一律只面对 dict。
    return {name: entry if isinstance(entry, dict) else {"endpoint_id": entry}
            for name, entry in data.items()}


def save(data):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def set_endpoint(name, endpoint_id, template_id=None, image=None):
    data = load()
    entry = data.get(name, {})
    entry["endpoint_id"] = endpoint_id
    if template_id:
        entry["template_id"] = template_id
    if image:
        entry["image"] = image
    data[name] = entry
    save(data)
    return entry


def resolve(name):
    """返回 endpoint_id;没登记过就抛出可操作的错误,而不是返回 None
    让调用方在后面某个地方莫名其妙地失败。"""
    entry = load().get(name)
    if not entry or not entry.get("endpoint_id"):
        raise KeyError(
            f"部署单元 {name!r} 还没登记。先跑 `comfyagent deploy {name} --image <镜像>`,"
            f" 或手动写入 {PATH}"
        )
    return entry["endpoint_id"]
