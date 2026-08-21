"""把参数注入 ComfyUI 节点图。

节点图里用 {{name}} 占位。两种替换规则:

  "{{seed}}"          整个值就是一个占位符 → 替换为参数的**原始类型**
                      (int 仍是 int)。ComfyUI 对类型敏感:seed 传成字符串
                      会在节点校验阶段被拒,而那个报错信息指向的是节点而不是
                      这里,很难追回来。
  "a photo of {{x}}"  占位符只是字符串的一部分 → 按字符串插值。
"""

import re

_WHOLE = re.compile(r"^\{\{\s*(\w+)\s*\}\}$")
_INLINE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class MissingParam(KeyError):
    """节点图引用了一个参数,但调用方没提供、声明里也没有默认值。"""


def render(template, params):
    """递归渲染节点图,返回新对象(不修改入参)。"""
    if isinstance(template, dict):
        return {k: render(v, params) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, params) for v in template]
    if not isinstance(template, str):
        return template

    whole = _WHOLE.match(template)
    if whole:
        name = whole.group(1)
        if name not in params:
            raise MissingParam(name)
        return params[name]          # 保持原始类型

    def sub(m):
        name = m.group(1)
        if name not in params:
            raise MissingParam(name)
        return str(params[name])

    return _INLINE.sub(sub, template)


def referenced_params(template):
    """节点图里引用到的全部参数名 —— 用来校验声明和节点图没有对不上。"""
    found = set()
    if isinstance(template, dict):
        for v in template.values():
            found |= referenced_params(v)
    elif isinstance(template, list):
        for v in template:
            found |= referenced_params(v)
    elif isinstance(template, str):
        found |= set(_INLINE.findall(template))
    return found
