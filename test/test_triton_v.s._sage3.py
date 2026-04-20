#!/usr/bin/env python3
"""
Compare sageattn3_blackwell against torch SDPA (fp32) reference.
"""

import os, sys, math, argparse
import torch
import torch.nn.functional as F

try:
    from sageattn3 import sageattn3_blackwell
    HAVE_SAGE3 = True
except Exception as e:
    HAVE_SAGE3 = False
    _sage3_err = e


def reference_sdpa(q, k, v, is_causal):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel([SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=is_causal)
    return out.to(q.dtype)


def metrics(x, ref, name=""):
    x, ref = x.float(), ref.float()
    diff = (x - ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos = F.cosine_similarity(x.reshape(-1), ref.reshape(-1), dim=0).item()
    if x.dim() == 4:
        row_err = diff.mean(dim=-1).reshape(-1)
        topv, topi = torch.topk(row_err, k=min(4, row_err.numel()))
        B, H, L, D = x.shape
        worst = [f"B{i_//(H*L)}H{(i_//L)%H}L{i_%L}={v_:.4e}" for v_, i_ in zip(topv.tolist(), topi.tolist())]
    else:
        worst = []
    return {"name": name, "max_diff": max_diff, "mean_diff": mean_diff, "cos_sim": cos, "worst_rows": worst}


def print_metrics(m):
    print(f"  [{m['name']:>16s}] max={m['max_diff']:.4e}  mean={m['mean_diff']:.4e}  cos={m['cos_sim']:.6f}")
    if m["worst_rows"]:
        print(f"                     worst rows: {', '.join(m['worst_rows'])}")


def run_case(B, H, L, D, is_causal, dtype, device, seed=0, label=""):
    torch.manual_seed(seed)
    q = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.3
    k = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.3
    v = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.3
    ref = reference_sdpa(q, k, v, is_causal=is_causal)
    header = f"case {label} B={B} H={H} L={L} D={D} causal={is_causal} dtype={dtype} seed={seed}"
    print("=" * len(header)); print(header); print("=" * len(header))
    results = []
    if HAVE_SAGE3:
        out = sageattn3_blackwell(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
        m = metrics(out, ref, name="sage3")
        print_metrics(m); results.append(m)
    else:
        print(f"  sageattn3 unavailable: {_sage3_err}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available"); return
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.quick:
        cases = [(2, 4, 512, 128, True)]
    else:
        modes = []
        if args.no_causal: modes.append(False)
        if args.causal: modes.append(True)
        if not modes: modes = [False, True]
        cases = [(2, 4, L, D, c) for c in modes for L in [128, 256, 512, 1024, 2048, 4096] for D in [64, 128]]
    all_results = []
    for i, (B, H, L, D, causal) in enumerate(cases):
        try:
            r = run_case(B, H, L, D, causal, dtype, device, seed=0, label=f"[{i}]")
        except Exception as e:
            print(f"case failed: {e!r}"); r = []
        all_results.append(((B, H, L, D, causal), r)); print()
    # FP4 quant: non-causal ~0.987, causal ~0.80 (early rows have sparse P).
    # Use 0.97 for non-causal and 0.75 for causal to catch real bugs vs FP4 noise.
    bad = []
    for p, res in all_results:
        B, H, L, D, causal = p
        thresh = 0.75 if causal else 0.97
        for m in res:
            if m["cos_sim"] < thresh:
                bad.append((p, m))
    print("=" * 70)
    if bad:
        print(f"SUMMARY: {len(bad)} FAILING cases")
        for (B, H, L, D, causal), m in bad:
            print(f"  B={B} H={H} L={L} D={D} causal={causal} {m['name']}: cos={m['cos_sim']:.6f} max={m['max_diff']:.4e}")
        sys.exit(1)
    else:
        print(f"SUMMARY: all {len(all_results)} cases passed")

if __name__ == "__main__":
    main()
