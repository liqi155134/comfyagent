"""工作流注册表 —— agent 能看见的全部能力都来自这里。

一个工作流 = 一份 ComfyUI 节点图(.json) + 一份参数声明(.yaml)。
新增能力只需要往 workflows/ 放这两个文件,CLI 和 MCP 两侧同时生效。

参数校验的报错信息是写给 **agent** 读的:它拿到错误后会自己改参数重试,
所以每条消息都要说清「哪个参数、什么问题、允许什么」,不能只说 invalid。
"""

import copy
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
    # 仅 images 类型:张数上限,以及节点图里怎么把这些图接进去
    max_items: int | None = None
    slot_node: str | None = None      # 作克隆样板的 LoadImage 节点 id
    target_node: str | None = None    # 消费这些图的节点 id
    target_prefix: str | None = None  # 该节点上的输入键前缀,如 ref_images.ref_image_

    _PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
           "image": str, "images": list}

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
            self._check_image_path(value)
        elif self.type == "images":
            if not value:
                raise ParamError(f"参数 {self.name} 至少要给一张图")
            if self.max_items and len(value) > self.max_items:
                raise ParamError(
                    f"参数 {self.name} 最多 {self.max_items} 张,收到 {len(value)} 张")
            for item in value:
                if not isinstance(item, str):
                    raise ParamError(
                        f"参数 {self.name} 每一项都得是路径字符串,"
                        f"收到 {type(item).__name__}")
                self._check_image_path(item)
        return value

    def _check_image_path(self, value):
        path = Path(value).expanduser()
        if not path.is_file():
            raise ParamError(f"参数 {self.name} 指向的图片不存在: {value}")
        if path.suffix.lower() not in self._IMAGE_EXTS:
            raise ParamError(
                f"参数 {self.name} 只接受 {sorted(self._IMAGE_EXTS)} 格式,收到 {path.suffix!r}"
            )


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
                render_params[name] = self._stage_image(resolved[name], images)

        # images 参数(可变张数):按实际张数克隆样板 LoadImage 节点并接线,
        # 多余的槽位从图里删掉 —— ComfyUI 对悬空引用会直接拒绝整张图。
        graph = self.template()
        for name, spec in self.params.items():
            if spec.type == "images" and resolved.get(name):
                self._expand_image_slots(graph, spec, resolved[name], images)
                render_params.pop(name, None)

        return render(graph, render_params), resolved, images

    def _stage_image(self, path_str, images):
        """读图 → 内容 hash 文件名 + base64 收进上传列表,返回节点图里该用的名字。"""
        path = Path(path_str).expanduser()
        data = path.read_bytes()
        fname = hashlib.sha1(data).hexdigest()[:16] + path.suffix.lower()
        if not any(u["name"] == fname for u in images):   # 同图只上传一次
            images.append({"name": fname,
                           "image": base64.b64encode(data).decode()})
        return fname

    def _expand_image_slots(self, graph, spec, paths, images):
        slot = graph.get(spec.slot_node)
        target = graph.get(spec.target_node)
        if slot is None or target is None:
            raise WorkflowError(
                f"参数 {spec.name} 声明的 slot_node={spec.slot_node!r} / "
                f"target_node={spec.target_node!r} 在模板里不存在")

        for i, p in enumerate(paths):
            node_id = spec.slot_node if i == 0 else f"{spec.slot_node}_ref{i}"
            node = slot if i == 0 else copy.deepcopy(slot)
            node["inputs"]["image"] = self._stage_image(p, images)
            graph[node_id] = node
            target["inputs"][f"{spec.target_prefix}{i}"] = [node_id, 0]

        # 模板里预置但这次没用到的槽位:一并清掉
        for key in [k for k in target["inputs"]
                    if k.startswith(spec.target_prefix)
                    and int(k[len(spec.target_prefix):]) >= len(paths)]:
            del target["inputs"][key]

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
        # images 类型不走占位符渲染(build 时直接改节点图),改为校验它指名的
        # slot_node / target_node 确实存在,否则同样是「运行时炸在 ComfyUI 里」。
        graph_ids = set(wf.template())
        slot_params = set()
        for name, spec in params.items():
            if spec.type != "images":
                continue
            slot_params.add(name)
            for field_name in ("slot_node", "target_node", "target_prefix"):
                if not getattr(spec, field_name):
                    raise WorkflowError(
                        f"{yml.name}: images 参数 {name} 必须声明 {field_name}")
            for field_name in ("slot_node", "target_node"):
                nid = getattr(spec, field_name)
                if nid not in graph_ids:
                    raise WorkflowError(
                        f"{yml.name}: images 参数 {name} 的 {field_name}={nid!r} "
                        f"在节点图里不存在")

        if missing := used - set(params):
            raise WorkflowError(f"{yml.name}: 节点图用到了未声明的参数 {sorted(missing)}")
        if extra := set(params) - used - slot_params:
            raise WorkflowError(f"{yml.name}: 声明了节点图并未使用的参数 {sorted(extra)}")

        if wf.id in out:
            raise WorkflowError(f"工作流 id 重复: {wf.id}")
        out[wf.id] = wf
    return out
