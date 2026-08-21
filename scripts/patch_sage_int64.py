"""把已安装 sageattention 的 per-thread 量化 Triton kernel 行偏移提升到 int64。

根因(2026-08-20 定论,见 docs/sage-sm90-issue-draft.md):offs_n * stride_in
在 int32 域相乘,fused QKV 布局(Q seq-stride=21504)下行号 > 2^31/21504 ≈ 99865
的有效行偏移 wrap 负 → 尾帧塌坏 / illegal memory access。

用法:
  python patch_sage_int64.py              # patch 已安装的 sageattention 包
  python patch_sage_int64.py <文件路径>    # patch 指定源码文件(编译 wheel 前用)

**必须在 bdist_wheel 之前 patch 源码**,否则导出的 wheel 仍是脆弱版 —— 单独
安装那个 wheel 的人会重新踩坑(2026-08-21 复核发现 Release wheel 正是如此)。
幂等:已 patch 则直接成功退出。
"""
import os
import re
import sys

if len(sys.argv) > 1:
    p = sys.argv[1]
else:
    import sageattention
    p = os.path.join(os.path.dirname(sageattention.__file__), "triton", "quant_per_thread.py")

if not os.path.isfile(p):
    print(f"patch_sage_int64: FATAL - 目标文件不存在: {p}", flush=True)
    sys.exit(1)
src = open(p).read()
if ".to(tl.int64)" in src:
    print(f"patch_sage_int64: already patched: {p}", flush=True)
    sys.exit(0)

pat = re.compile(r"offs_n(\d?)\[:, None\] \* stride_(in|on)")
patched, n = pat.subn(r"offs_n\1.to(tl.int64)[:, None] * stride_\2", src)
if n < 4:
    print(f"patch_sage_int64: FATAL - only {n} sites replaced, source layout changed?", flush=True)
    sys.exit(1)
open(p, "w").write(patched)
print(f"patch_sage_int64: patched {n} sites in {p}", flush=True)
