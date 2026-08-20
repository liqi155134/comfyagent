"""sage sm90 尾块塌坏探针 v2:精确复刻 H3 真实布局。

真实抓取(2026-08-20):视频流 q/k/v 是 fused QKV 大张量的 as_strided 切片
  shape (1, 56, L, 128), stride (7168, 128, 21504, 1), bf16, HND
input: {"cases": [{"L": 101509, "layout": "fused"|"contig"}, ...], "heads":56, "dim":128}
输出各 case 的全序列/8 段/末 128 tokens 相对误差(vs SDPA)。
"""
import torch
import torch.nn.functional as F
import runpod


def _mk(L, heads, dim, layout, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "contig":
        qkv = [torch.randn(1, heads, L, dim, generator=g, device="cuda", dtype=torch.bfloat16)
               for _ in range(3)]
        return qkv
    # fused:一个 (L, 3, heads, dim) 连续张量,q/k/v = 精确复刻真实 stride 的三段视图
    big = torch.randn(L * 3 * heads * dim, generator=g, device="cuda", dtype=torch.bfloat16)
    hd = heads * dim
    return [big.as_strided((1, heads, L, dim), (hd, dim, 3 * hd, 1), storage_offset=i * hd)
            for i in range(3)]


def handler(job):
    from sageattention import sageattn
    inp = job.get("input") or {}
    cases = inp.get("cases") or [{"L": 101509, "layout": "fused"}]
    heads = inp.get("heads", 56)
    dim = inp.get("dim", 128)

    def rel(d, r):
        return round((d.norm() / (r.norm() + 1e-8)).item(), 4)

    rows = []
    for c in cases:
        L, layout = c["L"], c.get("layout", "fused")
        try:
            q, k, v = _mk(L, heads, dim, layout, c.get("seed", 1))
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
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows}


runpod.serverless.start({"handler": handler})
