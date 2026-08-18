"""GPU 预检:worker 启动前确认 CUDA 真正可用。

以独立文件存在而不是 start.sh 里的内联 python -c —— 内联字符串里注释带个
双引号就会把 bash 的引号切断,产生一个残缺程序。这个 bug 曾让整个 endpoint
无限 crash loop,而 RunPod 的 health API 把快速崩溃循环显示成 idle/ready,
表象是「job 永远排队」,与真实原因隔了三层。独立文件 + 构建期 py_compile
断言让这类错误活不过 CI。
"""

import sys

import torch

try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    # 必须真的 launch 一个 kernel。只碰驱动的调用在「这个 torch 构建没有该
    # 架构编译 kernel」时(例:旧 torch 撞新卡)照样返回成功 —— 于是 worker
    # 正常起来、ComfyUI 在第一次 GPU 运算时死掉,最后表现为一个与真实原因
    # 毫无关系的启动失败。在这里炸掉,原因是明确的。
    _ = (torch.zeros(8, device="cuda") + 1).sum().item()
    torch.cuda.synchronize()
    print(
        f"OK: {name} (sm_{cap[0]}{cap[1]}), "
        f"torch {torch.__version__}, cuda {torch.version.cuda}"
    )
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
