"""往 ComfyUI attention.py 的 attention_sage 注入一次性形状打印(诊断用)。

在 worker 启动前执行:python3 inject_sage_shape_log.py && exec /start.sh
打印格式:SAGE_SHAPE {"q_shape":..., "q_stride":..., ...} —— 每种新形状只打一次。
"""
from pathlib import Path

p = Path("/comfyui/comfy/ldm/modules/attention.py")
src = p.read_text()

anchor = '''def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):'''
inject = anchor + '''
    import json as _json
    _sig = (tuple(q.shape), tuple(q.stride()), str(q.dtype), q.is_contiguous(),
            tuple(k.shape), tuple(v.shape), heads, skip_reshape, mask is not None,
            tuple(sorted(kwargs)))
    _seen = getattr(attention_sage, "_seen", set())
    if _sig not in _seen:
        attention_sage._seen = _seen | {_sig}
        print("SAGE_SHAPE " + _json.dumps({
            "q_shape": list(q.shape), "q_stride": list(q.stride()),
            "dtype": str(q.dtype), "contig": q.is_contiguous(),
            "k_shape": list(k.shape), "v_shape": list(v.shape),
            "heads": heads, "skip_reshape": skip_reshape,
            "has_mask": mask is not None, "kwargs": sorted(kwargs)}), flush=True)'''

assert src.count(anchor) == 1, "attention_sage 锚点不唯一"
p.write_text(src.replace(anchor, inject, 1))
print("shape log injected")
