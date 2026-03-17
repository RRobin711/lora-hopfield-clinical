# fp16 Mixed Precision — ROCm 6.2 / RX 6700S Investigation
**Date:** March 16, 2026
**Decision:** fp32 only. fp16 rejected.

## Setup
- AMD Radeon RX 6700S (gfx1032, masquerades as gfx1030)
- ROCm 6.2.0, PyTorch 2.5.1+rocm6.2
- hipBLASLt unsupported → falls back to hipBLAS

## Results (20 training steps, Hopfield GPT-2 on DREADDIT)
| Config | Steps/s | Samples/s | Peak VRAM | Inf gradient steps |
|---|---|---|---|---|
| fp32, bs=16 | 4.65 | 74.5 | 741 MB | 0/20 |
| fp16, bs=16 | 6.09 | 97.4 | 687 MB | 6/20 |
| fp16, bs=32 | 2.95 | 94.3 | 898 MB | 6/20 |

## Finding
fp16 is 31% faster in raw throughput but 30% of steps produce Inf gradients.
GradScaler collapsed from 65536 → 256 in 26 steps — chronic instability, not warmup.

Effective throughput: fp16 x 0.70 = 68.2 samples/sec vs fp32 74.5 samples/sec.
fp32 is 9% faster in terms of actual gradient updates delivered.

VRAM savings negligible (54 MB). fp32 is the correct choice for this hardware.

## Root cause
gfx1032 → gfx1030 masquerade disables hipBLASLt. The hipBLAS fallback path has
less robust fp16 overflow handling than NVIDIA Tensor Cores. This is a known
limitation of RDNA 2 under ROCm.
