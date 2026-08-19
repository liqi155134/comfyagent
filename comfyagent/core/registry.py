"""工作流注册表 —— agent 能看见的全部能力都来自这里。

一个工作流 = 一份 ComfyUI 节点图(.json) + 一份参数声明(.yaml)。
新增能力只需要往 workflows/ 放这两个文件,CLI 和 MCP 两侧同时生效。

参数校验的报错信息是写给 **agent** 读的:它拿到错误后会自己改参数重试,
所以每条消息都要说清「哪个参数、什么问题、允许什么」,不能只说 invalid。
"""

import base64
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .render import referenced_params, render


class WorkflowError(Exception):
    """工作流定义本身有问题(作者的错,不是调用方的错)。"""


class ParamError(ValueError):
    """调用方传的参数有问题。消息面向 agent,必须可据此纠正。"""


@dataclass
class ParamSpec:
    name: str
    type: str = "string"
    required: bool = False
    default: object = None
    description: str = ""
    enum: list | None = None
    min: float | None = None
    max: float | None = None

    _PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
           "image": str}

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

    def coerce(self, value):
        want = self._PY.get(self.type)
        if want is None:
            raise WorkflowError(f"参数 {self.name} 声明了未知类型 {self.type!r}")
        # 布尔要在整数之前判断:Python 里 True 是 int 的实例
        if self.type == "integer" and isinstance(value, bool):
            raise ParamError(f"参数 {self.name} 需要整数,收到布尔值 {value!r}")
        if not isinstance(value, want):
            if self.type in ("integer", "number") and isinstance(value, str):
                try:
                    return int(value) if self.type == "integer" else float(value)
                except ValueError:
                    pass
            raise ParamError(
                f"参数 {self.name} 需要 {self.type},收到 {type(value).__name__} ({value!r})"
            )
        return value

    def check(self, value):
        value = self.coerce(value)
        if self.enum is not None and value not in self.enum:
            raise ParamError(
                f"参数 {self.name} 只接受 {self.enum} 之一,收到 {value!r}"
            )
        if self.min is not None and value < self.min:
            raise ParamError(f"参数 {self.name} 不能小于 {self.min},收到 {value}")
        if self.max is not None and value > self.max:
            raise ParamError(f"参数 {self.name} 不能大于 {self.max},收到 {value}")
        if self.type == "image":
            path = Path(value).expanduser()
            if not path.is_file():
                raise ParamError(f"参数 {self.name} 指向的图片不存在: {value}")
            if path.suffix.lower() not in self._IMAGE_EXTS:
                raise ParamError(
                    f"参数 {self.name} 只接受 {sorted(self._IMAGE_EXTS)} 格式,收到 {path.suffix!r}"
                )
        return value


@dataclass
class Workflow:
    id: str
    title: str
    endpoint: str
    template_path: Path
    description: str = ""
    params: dict = field(default_factory=dict)

    def template(self):
        return json.loads(self.template_path.read_text(encoding="utf-8"))

    def build(self, given):
        """兼容旧签名:返回 (workflow_dict, resolved_params)。提交请用 build_payload。"""
        graph, resolved, _ = self._build_full(given)
        return graph, resolved

    def _build_full(self, given):
        """校验参数、填默认值、渲染出可直接提交的节点图。

        返回 (workflow_dict, resolved_params, images) —— resolved_params 要落进
        任务记录,否则「同 seed 复现」就无从谈起。
        """
        unknown = set(given) - set(self.params)
        if unknown:
            raise ParamError(
                f"未知参数 {sorted(unknown)};{self.id} 接受的是 {sorted(self.params)}"
            )

        resolved = {}
        for name, spec in self.params.items():
            if name in given and given[name] is not None:
                resolved[name] = spec.check(given[name])
            elif spec.required:
                raise ParamError(f"缺少必填参数 {name}:{spec.description or spec.type}")
            elif spec.default is not None:
                resolved[name] = spec.default
            elif name == "seed":
                # seed 省略即随机,但必须记下实际用的值,否则无法复现
                resolved[name] = random.randint(0, 2**32 - 1)
            else:
                resolved[name] = None

        # image 参数:节点图里渲染成内容 hash 文件名(worker 端上传后的引用名,
        # 避开中文/空格路径的编码问题,同图天然同名),字节以 base64 随 payload 上行。
        render_params = dict(resolved)
        images = []
        for name, spec in self.params.items():
            if spec.type == "image" and resolved.get(name):
                path = Path(resolved[name]).expanduser()
                data = path.read_bytes()
                fname = hashlib.sha1(data).hexdigest()[:16] + path.suffix.lower()
                images.append({"name": fname,
                               "image": base64.b64encode(data).decode()})
                render_params[name] = fname

        return render(self.template(), render_params), resolved, images

    def build_payload(self, given):
        """build 的提交侧封装:直接给出可发往 RunPod 的 input payload。"""
        graph, resolved, images = self._build_full(given)
        payload = {"workflow": graph}
        if images:
            payload["images"] = images
        return payload, resolved


def load_workflows(directory):
    """扫描目录加载全部工作流。定义有问题就直接抛错,不静默跳过。"""
    directory = Path(directory)
    out = {}
    for yml in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        for key in ("id", "endpoint", "template"):
            if key not in raw:
                raise WorkflowError(f"{yml.name} 缺少必填字段 {key!r}")

        tpl = directory / raw["template"]
        if not tpl.exists():
            raise WorkflowError(f"{yml.name} 指向的节点图不存在: {raw['template']}")

        params = {
            name: ParamSpec(name=name, **(spec or {}))
            for name, spec in (raw.get("params") or {}).items()
        }
        wf = Workflow(
            id=raw["id"],
            title=raw.get("title", raw["id"]),
            description=(raw.get("description") or "").strip(),
            endpoint=raw["endpoint"],
            template_path=tpl,
            params=params,
        )

        # 声明与节点图必须对得上。少了会在运行时炸在 ComfyUI 里(报错指向节点,
        # 极难追回这里);多了说明声明写了个没人用的参数,agent 会被误导。
        used = referenced_params(wf.template())
        if missing := used - set(params):
            raise WorkflowError(f"{yml.name}: 节点图用到了未声明的参数 {sorted(missing)}")
        if extra := set(params) - used:
            raise WorkflowError(f"{yml.name}: 声明了节点图并未使用的参数 {sorted(extra)}")

        if wf.id in out:
            raise WorkflowError(f"工作流 id 重复: {wf.id}")
        out[wf.id] = wf
    return out
