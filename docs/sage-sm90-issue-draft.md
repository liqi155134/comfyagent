# Title

per_thread quant Triton kernels: int32 pointer-arithmetic overflow corrupts Q (or crashes) for large-stride fused-QKV inputs at seq_len ≈ 100k+

# Body

## Environment

- GPU: NVIDIA H100 80GB HBM3 (sm90), RunPod serverless
- CUDA 13.0 host driver, PyTorch 2.13.0+cu130
- SageAttention built from source at main `d1a57a546c3d` (latest as of 2026-08),
  `TORCH_CUDA_ARCH_LIST=9.0`
- ComfyUI v0.33.1 with `--use-sage-attention`, MiniMax H3 (official int8_convrot
  quantized DiT), bf16 activations

## Symptom (production)

With MiniMax H3 image-to-video (15 s / 1280×736, video-stream seq_len **101509**),
the last ~3-4 output pixel frames (= last latent frame) collapse into gray noise;
occasionally the job dies with `CUDA error: an illegal memory access was encountered`
instead. Q/K/V arrive as **non-contiguous views of a fused QKV tensor**:

```
shape  = (1, 56, L, 128)
stride = (7168, 128, 21504, 1)   # seq-stride = 3 × heads × head_dim
```

## Root cause

`sageattention/triton/quant_per_thread.py :: quant_query_per_thread_int8_kernel`
computes row offsets in int32:

```python
offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld     # int32
...
input_ptrs = Input + ... + offs_n[:, None] * stride_in + offs_k[None, :]
```

With `stride_in = 21504`, the product `offs_n * stride_in` exceeds `INT32_MAX`
for every **valid** row `n > 2^31 / 21504 ≈ 99864`, wrapping negative. The load
mask (`offs_n < L`) does not help — these are in-bounds rows whose *addresses*
are computed wrong. Depending on where `base − ~2 GiB` lands, the kernel either
reads garbage (→ tail of the sequence quantizes to nonsense → last video frames
collapse) or hits an unmapped page (→ illegal memory access).

K is unaffected only by luck: with the default `smooth_k=True`, `k = k - km`
materializes a contiguous copy first, so its seq-stride is 128. Q keeps the
original stride 21504 all the way into the kernel. For contiguous Q the
overflow threshold is `2^31 / 128 ≈ 16.7M` tokens — unreachable, which is why
only the fused-layout path fails.

This is the same class of issue as triton-lang/triton#6346 (large-stride
multiplies compiled to `mul.lo.s32`) and #9247; the standard fix is an explicit
`offs_n.to(tl.int64) * stride_in`.

## Evidence (all on H100, bf16, heads=56, head_dim=128, HND)

Direct calls to `per_thread_int8_triton(q, k, km, BLKQ=64, WARPQ=16, BLKK=128,
WARPK=128)` with fused-layout Q (seq-stride 21504), comparing against the same
call on `q.contiguous()`:

| L | max row offset vs 2^31 | result |
|---|---|---|
| 8192 | far below | byte-identical int8 + scales |
| **99840** | 2,146,937,856 < 2^31 (largest safe 64-multiple) | **byte-identical** |
| **99904** | rows 99865..99903 wrap negative | **Triton illegal memory access** |
| 101504 (multiple of 128) | ~1640 tail rows wrap | illegal memory access |
| 101509 (production i2v length) | ~1644 tail rows wrap | illegal memory access |

The 64-row-granularity bisection lands exactly on `2^31 / 21504 = 99864.7`.

End-to-end `sageattn()` with the same fused layout (when the wild reads happen
to hit mapped memory instead of crashing): sequence-uniform rel-error 0.039 vs
SDPA for the first ~99.8k tokens, then the tail blows up (last-128-token
rel-error 0.8–1.1) — matching the production symptom, where the corrupted tail
tokens map to the last video frames. Same lengths with fully contiguous Q/K/V:
uniform 0.039, no tail anomaly. Lengths 65k–90k (below threshold): clean in
every layout.

Probe script (all experiments reproducible):
https://github.com/liqi155134/comfyagent/blob/main/scripts/diag_sage_tail.py

## Minimal repro

```python
import torch
from sageattention.core import per_thread_int8_triton  # triton path used by sm90 per_thread

L, H, D = 101509, 56, 128
big = torch.randn(L * 3 * H * D, device="cuda", dtype=torch.bfloat16)
q, k, v = (big.as_strided((1, H, L, D), (H * D, D, 3 * H * D, 1), storage_offset=i * H * D)
           for i in range(3))
km = k.mean(dim=2, keepdim=True)
per_thread_int8_triton(q, k, km, tensor_layout="HND", BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)
torch.cuda.synchronize()   # -> RuntimeError: Triton Error [CUDA]: an illegal memory access
```

`L = 99840` succeeds (byte-identical to the contiguous path); `L = 99904` is the
first failing 64-multiple.

## Verified fix

A copy of `quant_query_per_thread_int8_kernel` with a single change —

```python
in_row  = offs_n.to(tl.int64)[:, None] * stride_in
out_row = offs_n.to(tl.int64)[:, None] * stride_on
```

— run on the **strided** Q at L=101509 produces int8 output and scales
**byte-identical** to the original kernel on `q.contiguous()` (0 mismatched
elements in both), where the unpatched kernel on the same strided input dies
with an illegal memory access. (H100, torch 2.13.0+cu130, CUDA_LAUNCH_BLOCKING=1.)

## Impact & suggested fix

- Not sm90-specific: every arch path that uses `per_thread_int8_triton` (and the
  int4 / per-warp Triton variants with the same `offs_n * stride_in` pattern) is
  affected for any large-stride input with `seq_len > 2^31 / seq_stride`.
  Fused-QKV layouts (stride = 3·H·D) hit this at ~100k tokens — exactly the
  regime of current long video DiTs.
- Fix: promote row offsets to int64 in the quant kernels
  (`offs_n.to(tl.int64) * stride_in`, likewise for the output side when the
  output can be strided), or `.contiguous()`-fallback on large-stride inputs at
  the dispatch level.

Happy to send a PR for the int64 promotion if useful.
