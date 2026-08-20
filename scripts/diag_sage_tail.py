"""sage sm90 尾块塌坏探针 v3:张量级二分 + 坏 token 范围精确测量。

v2 结论(2026-08-20,H100 实测):fused 布局(真实 H3 的 QKV 切片 stride)在
L=100613/100691/101445/101509 全部复现尾部塌坏(末 128 token 相对误差 0.8-1.1),
contig 布局同长度全净 → 触发条件是非连续布局,不是长度残差类。

v3 新增:
- case 可带 "contig": ["q"] 把指定张量 .contiguous()(二分定位是哪个张量触发)
- 输出坏 token 精确范围(per-token 相对误差 > max(5*中位数, 0.15) 的索引区间)
- 输出 "ver": 3(warm worker 可能缓存旧脚本,调用方必须校验此字段)

input: {"cases": [{"L": 101509, "layout": "fused", "contig": ["q"], "seed": 1}], "heads": 56, "dim": 128}
"""
import torch
import torch.nn.functional as F
import runpod


def _mk(L, heads, dim, layout, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "contig":
        return [torch.randn(1, heads, L, dim, generator=g, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)]
    # fused:一个连续大张量,q/k/v = 精确复刻真实 stride (7168, 128, 21504, 1) 的三段视图
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
            qkv = _mk(L, heads, dim, layout, c.get("seed", 1))
            for name in c.get("contig", []):
                i = {"q": 0, "k": 1, "v": 2}[name]
                qkv[i] = qkv[i].contiguous()
            q, k, v = qkv
            ref = F.scaled_dot_product_attention(q, k, v).float()
            res = sageattn(q, k, v, tensor_layout="HND", is_causal=False).float()
            d = res - ref
            # per-token 相对误差 → 坏区间
            et = d.norm(dim=(0, 1, 3)) / (ref.norm(dim=(0, 1, 3)) + 1e-8)
            med = et.median().item()
            bad = (et > max(5 * med, 0.15)).nonzero().flatten()
            row = {"L": L, "layout": layout, "contig": c.get("contig", []),
                   "rel_all": rel(d, ref), "rel_last128": rel(d[:, :, -128:], ref[:, :, -128:]),
                   "err_median": round(med, 4), "bad_tokens": int(bad.numel())}
            if bad.numel():
                row["bad_first"] = int(bad.min())
                row["bad_last"] = int(bad.max())
                row["bad_rel_L"] = [round(int(bad.min()) / L, 4), round(int(bad.max()) / L, 4)]
            # 尾部 8 个 128-token 块的误差 profile
            row["tail_blocks128"] = [rel(d[:, :, max(L - b * 128, 0):L - (b - 1) * 128],
                                         ref[:, :, max(L - b * 128, 0):L - (b - 1) * 128])
                                     for b in range(8, 0, -1)]
            rows.append(row)
            del qkv, q, k, v, ref, res, d, et
        except Exception as e:
            rows.append({"L": L, "layout": layout, "contig": c.get("contig", []),
                         "error": f"{type(e).__name__}: {e}"[:300]})
        torch.cuda.empty_cache()
    return {"ver": 3, "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "rows": rows}


runpod.serverless.start({"handler": handler})
