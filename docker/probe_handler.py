"""最小 serverless handler —— 只回报自己活着,以及所在机器的基本信息。"""
import os
import runpod


def handler(job):
    return {
        "ok": True,
        "echo": job.get("input"),
        "hostname": os.uname().nodename,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
    }


runpod.serverless.start({"handler": handler})
