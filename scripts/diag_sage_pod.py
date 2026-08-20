"""Pod 版探针:直接跑 case 矩阵,结果 JSON 上传 R2(绕开 serverless 调度)。"""
import json
import os

import boto3
import torch
import torch.nn.functional as F


def _mk(L, heads, dim, layout, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "contig":
        return [torch.randn(1, heads, L, dim, generator=g, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)]
    big = torch.randn(L * 3 * heads * dim, generator=g, device="cuda", dtype=torch.bfloat16)
    hd = heads * dim
    return [big.as_strided((1, heads, L, dim), (hd, dim, 3 * hd, 1), storage_offset=i * hd)
            for i in range(3)]


def main():
    from sageattention import sageattn
    heads, dim = 56, 128
    cases = []
    for L in (15409, 16227, 100691, 101509):
        cases += [{"L": L, "layout": "fused"}, {"L": L, "layout": "contig"}]
    cases += [{"L": 943, "layout": "contig"}, {"L": 423, "layout": "contig"}]

    def rel(d, r):
        return round((d.norm() / (r.norm() + 1e-8)).item(), 4)

    rows = []
    for c in cases:
        L, layout = c["L"], c["layout"]
        try:
            q, k, v = _mk(L, heads, dim, layout, 1)
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
        print("done", c, flush=True)

    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows}
    s3 = boto3.client("s3", endpoint_url=os.environ["BUCKET_ENDPOINT_URL"],
                      aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"])
    s3.put_object(Bucket=os.environ["BUCKET_NAME"], Key="diag/sage_tail_result.json",
                  Body=json.dumps(result, ensure_ascii=False).encode())
    print("UPLOADED", flush=True)


if __name__ == "__main__":
    main()
