"""把已安装 sageattention 的 per-thread 量化 Triton kernel 行偏移提升到 int64。

根因(2026-08-20 定论,见 docs/sage-sm90-issue-draft.md):offs_n * stride_in
在 int32 域相乘,fused QKV 布局(Q seq-stride=21504)下行号 > 2^31/21504 ≈ 99865
的有效行偏移 wrap 负 → 尾帧塌坏 / illegal memory access。

用法:容器启动时、ComfyUI 起来之前执行(dockerStartCmd 里 && 链接,失败即容器
不起,保证"job 能跑 = 补丁已生效")。幂等:已 patch 则直接成功退出。
"""
import os
import re
import sys

import sageattention

p = os.path.join(os.path.dirname(sageattention.__file__), "triton", "quant_per_thread.py")
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
