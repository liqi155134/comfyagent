"""sage 塌坏探针 v5:根因 = Q 量化 Triton kernel 指针算术 int32 溢出,本版做 int64 补丁 A/B。

证据链(2026-08-20,H100 实测,均在本探针历史版本产出):
- fused 布局(真实 H3 QKV 切片 stride (7168,128,21504,1))L>1e5 全部复现尾部塌坏,
  contig 同长度全净;65k-90k 任何布局全净。
- quant 单测:strided Q @ L=8192(偏移 < 2^31)逐字节正确;L=101509/101504 直接
  Triton illegal memory access → 与整除性无关,与 L 绝对大小相关。
- 阈值二分:L=99840(最大行偏移 2,146,937,856 < 2^31)全净;L=99904(行 99865+
  偏移 wrap 负)illegal access。int32 溢出阈值 n = 2^31/21504 ≈ 99864.7 精确命中。
- 独立 bug A:L%128==0 时 V 不做 pad、保持 strided 进 per_channel_fp8(CUDA kernel)
  → crash;contig V 后同 case 干净。

v5:mode="ab" —— 内嵌只改一处的 int64 kernel 副本,strided Q 输入下与
contiguous+原版 kernel 金标准逐字节比对。一致即定论。
每 job 单 case(illegal access 会废 CUDA context);CUDA_LAUNCH_BLOCKING=1 让
crash 栈可信;输出 "ver": 5(warm worker 可能缓存旧脚本,调用方必须校验)。

input: {"case": {"L": 101509, "mode": "ab", "seed": 1}, "heads": 56, "dim": 128}
"""
import os

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # 必须在 import torch 之前

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import runpod


@triton.jit
def _quant_q_int64_kernel(Input, Output, Scale, L,
                          stride_iz, stride_ih, stride_in,
                          stride_oz, stride_oh, stride_on,
                          stride_sz, stride_sh,
                          C: tl.constexpr, BLK: tl.constexpr):
    """quant_query_per_thread_int8_kernel 的副本,唯一改动:行偏移用 int64 计算。

    原版 offs_n[:, None] * stride_in 在 int32 域相乘,stride_in=21504 时
    n > 2^31/21504 ≈ 99865 即 wrap 成负偏移 → 非法访存/读垃圾。
    """
    off_blk = tl.program_id(0) // 8
    off_tld = tl.program_id(0) % 8
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
    offs_k = tl.arange(0, C)
    in_row = offs_n.to(tl.int64)[:, None] * stride_in     # <-- 唯一改动
    out_row = offs_n.to(tl.int64)[:, None] * stride_on    # <-- 唯一改动

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + in_row + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + out_row + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / 127. + 0.0000001
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


def _quant_q(q, kernel, BLKQ=64, WARPQ=16):
    """复刻 per_thread_int8 的 Q 侧封装(sm90 参数),kernel 可替换。"""
    b, h, L, dim = q.shape
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    q_scale = torch.empty((b, h, (L + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8),
                          device=q.device, dtype=torch.float32)
    grid = ((L + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h, b)
    kernel[grid](q, q_int8, q_scale, L,
                 q.stride(0), q.stride(1), q.stride(2),
                 q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
                 q_scale.stride(0), q_scale.stride(1),
                 C=dim, BLK=WARPQ)
    torch.cuda.synchronize()
    return q_int8, q_scale


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


def _ab_case(c, heads, dim, out):
    """int64 补丁 A/B:patched kernel + strided Q vs 原版 kernel + contiguous Q(金标准)。"""
    from sageattention.triton.quant_per_thread import quant_query_per_thread_int8_kernel
    L = c["L"]
    q, _, _ = _mk(L, heads, dim, "fused", c.get("seed", 1))
    gold_int8, gold_scale = _quant_q(q.contiguous(), quant_query_per_thread_int8_kernel)
    out["steps_ok"].append("gold_contig_orig")
    p_int8, p_scale = _quant_q(q, _quant_q_int64_kernel)
    out["steps_ok"].append("patched_strided")
    for name, a, b in [("int8", p_int8, gold_int8), ("scale", p_scale, gold_scale)]:
        neq = a != b
        d = {"mismatch_elems": int(neq.sum())}
        if d["mismatch_elems"]:
            d["rows"] = _bad_rows(neq, 2)
        out[f"patched_vs_gold_{name}"] = d


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
    out = {"ver": 5, "case": c, "steps_ok": [],
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    try:
        mode = c.get("mode", "full")
        if mode == "ab":
            _ab_case(c, heads, dim, out)
        elif mode == "quant":
            _quant_case(c, heads, dim, out)
        else:
            _full_case(c, heads, dim, out)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:400]
    return out


runpod.serverless.start({"handler": handler})
