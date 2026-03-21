"""Benchmark all skaters policies on M4 data.

Usage:
    uv run python fetch_m4.py    # first, download the data
    uv run python benchmark.py   # then, run the benchmark
"""

import json
import math
import os
import sys
import time

from skaters.api import holt, hosking, laplace, samuelson, wald, dantzig, skater
from skaters.dist import Dist


POLICIES = {
    "holt": holt,
    "hosking": hosking,
    "laplace": laplace,
    "samuelson": samuelson,
    "wald": wald,
    "dantzig": dantzig,
}


def run_policy(factory, series, burn_frac=0.3):
    """Run a policy on a series, return mean logpdf after burn-in."""
    burn = max(int(len(series) * burn_frac), 20)
    f = factory(k=1)
    state = None
    prev_mean = prev_std = None
    logpdfs = []
    for i, y in enumerate(series):
        dists, state = f(y, state)
        if i > burn and prev_mean is not None and prev_std and prev_std > 0:
            lp = Dist.gaussian(prev_mean, prev_std).logpdf(y)
            if math.isfinite(lp):
                logpdfs.append(lp)
        prev_mean = dists[0].mean
        prev_std = dists[0].std
    return sum(logpdfs) / len(logpdfs) if logpdfs else float("-inf")


def main():
    datapath = "data/m4_sample.json"
    if not os.path.exists(datapath):
        print("Run fetch_m4.py first to download the data.")
        return

    with open(datapath) as f:
        all_series = json.load(f)

    series_names = sorted(all_series.keys())
    policy_names = list(POLICIES.keys())
    n_series = len(series_names)
    n_policies = len(policy_names)

    print(f"Benchmarking {n_policies} policies on {n_series} series...\n")

    # scores[policy][series] = mean logpdf
    scores = {p: {} for p in policy_names}
    t0 = time.time()

    for si, sname in enumerate(series_names):
        elapsed = time.time() - t0
        rate = (si + 1) / elapsed if elapsed > 0 else 0
        eta = (n_series - si - 1) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r  {si+1}/{n_series}: {sname:<30} "
            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
        )
        sys.stdout.flush()

        series = all_series[sname]
        for pname in policy_names:
            scores[pname][sname] = run_policy(POLICIES[pname], series)

    print(f"\n\nDone in {time.time()-t0:.1f}s\n")

    # --- Rank table ---
    rank_sums = {p: 0 for p in policy_names}
    rank_counts = {p: 0 for p in policy_names}
    valid_series = []

    for sname in series_names:
        vals = [scores[p][sname] for p in policy_names]
        if not all(math.isfinite(v) for v in vals):
            continue
        valid_series.append(sname)
        ranked = sorted(policy_names, key=lambda p: scores[p][sname], reverse=True)
        for r, p in enumerate(ranked):
            rank_sums[p] += r + 1
            rank_counts[p] += 1

    n_valid = len(valid_series)
    print(f"Valid series (all policies finite): {n_valid}/{n_series}\n")

    # --- Mean rank ---
    print("MEAN RANK (lower = better)")
    print("-" * 40)
    for p in sorted(policy_names, key=lambda p: rank_sums[p] / max(rank_counts[p], 1)):
        mr = rank_sums[p] / max(rank_counts[p], 1)
        print(f"  {p:<15} {mr:.2f}")

    # --- Win counts ---
    print(f"\nWIN COUNTS (best logpdf)")
    print("-" * 40)
    wins = {p: 0 for p in policy_names}
    for sname in valid_series:
        best_p = max(policy_names, key=lambda p: scores[p][sname])
        wins[best_p] += 1
    for p in sorted(policy_names, key=lambda p: wins[p], reverse=True):
        print(f"  {p:<15} {wins[p]:4d} / {n_valid}")

    # --- By frequency ---
    for freq in ["yearly", "quarterly", "monthly", "daily"]:
        freq_series = [s for s in valid_series if s.startswith(freq)]
        if not freq_series:
            continue
        print(f"\n{freq.upper()} ({len(freq_series)} series)")
        print("-" * 40)
        freq_ranks = {p: 0 for p in policy_names}
        for sname in freq_series:
            ranked = sorted(policy_names, key=lambda p: scores[p][sname], reverse=True)
            for r, p in enumerate(ranked):
                freq_ranks[p] += r + 1
        for p in sorted(policy_names, key=lambda p: freq_ranks[p]):
            mr = freq_ranks[p] / len(freq_series)
            print(f"  {p:<15} {mr:.2f}")

    # Save results
    results = {
        "n_series": n_series,
        "n_valid": n_valid,
        "scores": scores,
        "rank_sums": rank_sums,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(results, f)
    print(f"\nResults saved to results/benchmark.json")


if __name__ == "__main__":
    main()
