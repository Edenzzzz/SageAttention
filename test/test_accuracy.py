#!/usr/bin/env python3
"""Cosine similarity of sageattn3_blackwell vs F32 SDPA reference."""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from sageattn3 import sageattn3_blackwell


def reference_sdpa(q, k, v, is_causal):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=is_causal
        )
    return out.to(q.dtype)


def test(B, H, L, D, is_causal, dtype=torch.bfloat16):
    torch.manual_seed(42)
    q = torch.randn(B, H, L, D, device="cuda", dtype=dtype) * 0.3
    k = torch.randn(B, H, L, D, device="cuda", dtype=dtype) * 0.3
    v = torch.randn(B, H, L, D, device="cuda", dtype=dtype) * 0.3
    ref = reference_sdpa(q, k, v, is_causal=is_causal)
    out = sageattn3_blackwell(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
    return F.cosine_similarity(
        out.float().reshape(-1), ref.float().reshape(-1), dim=0
    ).item()


def main():
    threshold = 0.97
    failures = []

    print(f"{'B':>2} {'H':>2} {'L':>5} {'D':>4} {'causal':>6}  {'cos':>8}")
    print("-" * 40)
    for is_causal in [False, True]:
        for L in [128, 256, 512, 1024, 2048, 4096]:
            for D in [64, 128]:
                cos = test(2, 4, L, D, is_causal)
                ok = cos >= threshold
                tag = "OK" if ok else "FAIL"
                print(
                    f"{2:>2} {4:>2} {L:>5} {D:>4} {str(is_causal):>6}"
                    f"  {cos:>8.5f}  {tag}"
                )
                if not ok:
                    failures.append((L, D, is_causal, cos))

    print()
    if failures:
        print(f"FAILED {len(failures)} cases (threshold={threshold}):")
        for L, D, causal, cos in failures:
            print(f"  L={L} D={D} causal={causal}: cos={cos:.5f}")
        sys.exit(1)
    else:
        print("All cases passed.")


if __name__ == "__main__":
    main()
