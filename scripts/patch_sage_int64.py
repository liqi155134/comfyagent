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

# 脆弱写法:行偏移在 int32 域相乘。判据是"还剩几处",不是"有没有出现过 int64" ——
# 后者在部分打过补丁的源码上会误判为已修完,漏掉剩下的(2026-08-21 复核指出)。
VULNERABLE = re.compile(r"offs_n(\d?)\[:, None\] \* stride_(in|on)")
before = len(VULNERABLE.findall(src))
already = src.count(".to(tl.int64)")

if before == 0:
    if already == 0:
        print(f"patch_sage_int64: FATAL - 一处脆弱写法都没找到,源码结构变了? {p}", flush=True)
        sys.exit(1)
    print(f"patch_sage_int64: already patched ({already} 处 int64,0 处脆弱): {p}", flush=True)
    sys.exit(0)

patched, n = VULNERABLE.subn(r"offs_n\1.to(tl.int64)[:, None] * stride_\2", src)
if n < 4:
    print(f"patch_sage_int64: FATAL - 只替换了 {n} 处,源码结构变了?", flush=True)
    sys.exit(1)

# 落盘后回读断言:必须一处脆弱写法都不剩
open(p, "w").write(patched)
after = len(VULNERABLE.findall(open(p).read()))
if after != 0:
    print(f"patch_sage_int64: FATAL - 补完仍剩 {after} 处脆弱写法", flush=True)
    sys.exit(1)
print(f"patch_sage_int64: patched {n} sites, 0 vulnerable left in {p}", flush=True)
