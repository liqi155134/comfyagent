# Title

sm90: tail of sequence corrupted (last video frames become gray noise) with MiniMax H3 image-to-video — works on text-to-video with nearly identical length

# Body

## Environment

- GPU: NVIDIA H100 80GB HBM3 (sm90), RunPod serverless
- CUDA 13.0 host driver, PyTorch 2.13.0+cu130
- SageAttention built from source at main `d1a57a546c3d` (latest as of 2026-08),
  `TORCH_CUDA_ARCH_LIST=9.0`
- ComfyUI v0.33.1 with `--use-sage-attention`, MiniMax H3 (official int8_convrot
  quantized DiT), bf16 activations

## Symptom

With MiniMax H3 **image-to-video** (first-frame conditioned), the last ~3-4 output
pixel frames (= last 1 latent frame, temporal VAE 4x) collapse into gray low-frequency
noise. The preceding ~470 frames are perfect. 100% reproducible across seeds, hosts,
and step counts (20-step base and 8-step LoRA both affected).

**Text-to-video through the same pipeline is clean**, including a 4-way same-seed
matrix (sage on/off × 20-step/8-step) where sage-on output is nearly pixel-identical
to the fp16 SDPA reference.

## Isolation

| variable | result |
|---|---|
| same seed, `--use-sage-attention` off (pytorch SDPA) | clean |
| same shapes through comfy-kitchen int8 attention (`--use-ck-attention`) | clean |
| different worker/host | still corrupted |
| t2v vs i2v | only i2v corrupted |

So this is specific to the sageattention sm90 path, triggered by the i2v workload.

## Actual attention call shapes (captured in-process)

All calls are `sageattn(q, k, v, tensor_layout="HND", is_causal=False)`, bf16.
Q/K/V are **non-contiguous strided views of a fused QKV tensor**
(hidden 21504 = 3×56×128):

| stream | shape | stride |
|---|---|---|
| video stream, t2v 15s/1280×736 | (1, 56, **100691**, 128) | (7168, 128, **21504**, 1) |
| video stream, i2v 15s/1280×736 (corrupted) | (1, 56, **101509**, 128) | (7168, 128, 21504, 1) |
| text/cond stream, i2v | (1, 56, 943, 128) | (7168, 128, 7168, 1) |

The only difference between the clean t2v case and the corrupted i2v case on the
video stream is +818 tokens (keyframe conditioning tokens appended by the i2v path).
The corruption appears at the **tail of the sequence** (which maps to the last
video frames).

## What did NOT reproduce it

A synthetic probe (random bf16 data, same head count/dim, both contiguous and
transposed layouts, L from 65k to 90k including odd remainders) shows uniform
~0.039 relative error vs SDPA — no tail blowup. So plain random-data self-attention
at these sizes is fine; the trigger seems to need the exact fused-QKV strided layout
at L≈101k and/or real data distribution.

Probe script: https://github.com/liqi155134/comfyagent/blob/main/scripts/diag_sage_tail.py

## Suspicion

Something in the sm90 kernel's handling of the final KV/Q tiles at large L with the
strided (stride_n=3×head_stride) layout — possibly out-of-range reads on the tail
block, or per-block quantization scales going degenerate. Related reports:
#320 (sm90 backend broken, fixed on main), Comfy-Org/ComfyUI#15263 (H3 + sage noise
above ~160k tokens on sm_120).

Happy to run further probes on H100 if you can suggest what to instrument.
