"""sage sm90 尾块塌坏探针 v4:直击 Q 量化 Triton kernel + 步骤级 crash 定位。

已知(2026-08-20,H100 实测):
- v2:fused 布局(真实 H3 QKV 切片 stride (7168,128,21504,1))在 4 个 L 全部复现
  尾部塌坏(末 128 token 相对误差 0.8-1.1),contig 同长度全净。
- v3:L=8192 fused 直接 CUDA illegal memory access → 是真越界访存,不只是数值错。
  (顺带解释了生产上 i2v 首跑偶发 illegal memory access。)

v4:每 job 单 case(illegal access 会废掉 CUDA context)。
- mode="quant":单独跑 per_thread_int8_triton,strided Q vs contiguous Q 逐字节比
  int8 输出与 scale,直接看错在哪些 seq 行。
- mode="full":整个 sageattn,每步 synchronize,crash 时报告最后成功的步骤。
- 输出 "ver": 4(warm worker 可能缓存旧脚本,调用方必须校验)。

input: {"case": {"L": 8192, "mode": "quant", "layout": "fused", "contig": [], "seed": 1},
        "heads": 56, "dim": 128}
"""
import torch
import torch.nn.functional as F
import runpod


def _mk(L, heads, dim, layout, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "contig":
        return [torch.randn(1, heads, L, dim, generator=g, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)]
    big = torch.randn(L * 3 * heads * dim, generator=g, device="cuda", dtype=torch.bfloat16)
    hd = heads * dim
    return [big.as_strided((1, heads, L, dim), (hd, dim, 3 * hd, 1), storage_offset=i * hd)
            for i in range(3)]


def _bad_rows(neq, seq_dim):
    """neq: bool 张量;返回沿 seq 维的坏行统计。"""
    dims = [d for d in range(neq.dim()) if d != seq_dim]
    rows = neq.any(dim=dims[-1]) if len(dims) == 1 else neq
    for d in sorted(dims, reverse=True):
        rows = rows.any(dim=d) if rows.dim() > 1 else rows
    idx = rows.nonzero().flatten()
    if not idx.numel():
        return {"count": 0}
    return {"count": int(idx.numel()), "first": int(idx.min()), "last": int(idx.max())}


def _quant_case(c, heads, dim, out):
    from sageattention.core import per_thread_int8_triton
    L = c["L"]
    q, k, v = _mk(L, heads, dim, "fused", c.get("seed", 1))
    km = k.mean(dim=2, keepdim=True)
    kw = dict(tensor_layout="HND", BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)
    r_s = per_thread_int8_triton(q, k, km, **kw)
    torch.cuda.synchronize()
    out["steps_ok"].append("quant_strided")
    r_c = per_thread_int8_triton(q.contiguous(), k.contiguous(), km, **kw)
    torch.cuda.synchronize()
    out["steps_ok"].append("quant_contig")
    for name, a, b in zip(["q_int8", "q_scale", "k_int8", "k_scale"], r_s, r_c):
        neq = a != b
        d = {"mismatch_elems": int(neq.sum())}
        if d["mismatch_elems"]:
            d["rows"] = _bad_rows(neq, 2)  # int8 张量与 scale 的 dim2 都是 seq/块 维
        out[name] = d


def _full_case(c, heads, dim, out):
    from sageattention import sageattn
    L, layout = c["L"], c.get("layout", "fused")
    qkv = _mk(L, heads, dim, layout, c.get("seed", 1))
    for name in c.get("contig", []):
        i = {"q": 0, "k": 1, "v": 2}[name]
        qkv[i] = qkv[i].contiguous()
    q, k, v = qkv
    torch.cuda.synchronize()
    out["steps_ok"].append("mk")
    ref = F.scaled_dot_product_attention(q, k, v).float()
    torch.cuda.synchronize()
    out["steps_ok"].append("sdpa")
    res = sageattn(q, k, v, tensor_layout="HND", is_causal=False).float()
    torch.cuda.synchronize()
    out["steps_ok"].append("sageattn")

    def rel(d, r):
        return round((d.norm() / (r.norm() + 1e-8)).item(), 4)

    d = res - ref
    # 注:Tensor.norm(dim=3元组) 走 matrix_norm 会报错,手写 L2
    et = d.square().sum(dim=(0, 1, 3)).sqrt() / (ref.square().sum(dim=(0, 1, 3)).sqrt() + 1e-8)
    med = et.median().item()
    bad = (et > max(5 * med, 0.15)).nonzero().flatten()
    out.update({"rel_all": rel(d, ref), "rel_last128": rel(d[:, :, -128:], ref[:, :, -128:]),
                "err_median": round(med, 4), "bad_tokens": int(bad.numel())})
    if bad.numel():
        out["bad_first"] = int(bad.min())
        out["bad_last"] = int(bad.max())


def handler(job):
    inp = job.get("input") or {}
    c = inp.get("case") or {"L": 8192, "mode": "quant"}
    heads = inp.get("heads", 56)
    dim = inp.get("dim", 128)
    out = {"ver": 4, "case": c, "steps_ok": [],
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    try:
        if c.get("mode", "full") == "quant":
            _quant_case(c, heads, dim, out)
        else:
            _full_case(c, heads, dim, out)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:400]
    return out


runpod.serverless.start({"handler": handler})
