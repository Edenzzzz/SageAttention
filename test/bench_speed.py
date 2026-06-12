#!/usr/bin/env python3
"""Throughput benchmark for sageattn3_blackwell."""

import sys
import torch

sys.path.insert(0, ".")
from sageattn3 import sageattn3_blackwell


def bench(B, H, L, D, is_causal, warmup=20, iters=100):
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)

    for _ in range(warmup):
        sageattn3_blackwell(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        sageattn3_blackwell(q, k, v, is_causal=is_causal)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    flops = 4 * B * H * L * L * D / 1e12
    tflops = flops / (ms / 1e3)
    return ms, tflops


def main():
    B, H = 2, 32
    print(f"{'B':>2} {'H':>2} {'L':>5} {'D':>4} {'causal':>6}  {'ms':>8}  {'TFLOPS':>7}")
    print("-" * 50)
    for is_causal in [False, True]:
        for L in [512, 1024, 2048, 4096]:
            for D in [64, 128]:
                ms, tflops = bench(B, H, L, D, is_causal)
                print(
                    f"{B:>2} {H:>2} {L:>5} {D:>4} {str(is_causal):>6}"
                    f"  {ms:>8.3f}  {tflops:>7.2f}"
                )


if __name__ == "__main__":
    main()
