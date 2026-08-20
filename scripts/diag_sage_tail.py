"""sage sm90 尾块塌坏的最小复现探针(serverless diag handler,不加载 H3 模型)。

input: {"lengths": [...], "layouts": ["contig","noncontig"], "heads": 56, "dim": 128}
对每个 (L, layout) 用合成 QKV 跑 sageattn vs SDPA,输出全序列/8 段/末 128 tokens
的相对误差。正常 ~0.01-0.05;尾块塌坏时对应段会到 0.5+。
"""
import torch
import torch.nn.functional as F
import runpod


def _mk(L, heads, dim, layout, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "contig":
        return torch.randn(1, heads, L, dim, generator=g, device="cuda", dtype=torch.bfloat16)
    # 模拟 H3 fused-QKV split 后的非连续视图:(L, H, D) -> (1, H, L, D)
    t = torch.randn(L, heads, dim, generator=g, device="cuda", dtype=torch.bfloat16)
    return t.transpose(0, 1).unsqueeze(0)


def handler(job):
    from sageattention import sageattn
    inp = job.get("input") or {}
    Ls = inp.get("lengths") or [65536, 73728, 80000, 80017, 81920, 81921, 84000, 86016, 88064, 90000]
    layouts = inp.get("layouts") or ["contig", "noncontig"]
    heads = inp.get("heads", 56)
    dim = inp.get("dim", 128)

    def rel(d, r):
        return round((d.norm() / (r.norm() + 1e-8)).item(), 4)

    rows = []
    for L in Ls:
        for layout in layouts:
            try:
                q = _mk(L, heads, dim, layout, 1)
                k = _mk(L, heads, dim, layout, 2)
                v = _mk(L, heads, dim, layout, 3)
                ref = F.scaled_dot_product_attention(q, k, v)
                res = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
                d = (res.float() - ref.float())
                segs = [rel(d[:, :, L * i // 8:L * (i + 1) // 8],
                            ref[:, :, L * i // 8:L * (i + 1) // 8].float()) for i in range(8)]
                rows.append({"L": L, "layout": layout, "rel_all": rel(d, ref.float()),
                             "rel_segments": segs,
                             "rel_last128": rel(d[:, :, -128:], ref[:, :, -128:].float())})
                del q, k, v, ref, res, d
            except Exception as e:
                rows.append({"L": L, "layout": layout, "error": f"{type(e).__name__}: {e}"[:300]})
            torch.cuda.empty_cache()
    return {"gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "rows": rows}


runpod.serverless.start({"handler": handler})
