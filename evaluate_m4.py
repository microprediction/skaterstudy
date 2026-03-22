"""Proper M4 evaluation: sMAPE and MASE on held-out test data.

Trains on the full training series, then forecasts h steps ahead
and compares against the actual test values.

M4 benchmark baselines (from the competition):
- Naive2 (seasonal naive): the official baseline
- SES, Holt, Damped, Theta, Comb (statistical methods)

Usage:
    uv run python fetch_m4.py       # download training data
    uv run python fetch_m4_test.py  # download test data
    uv run python evaluate_m4.py   # evaluate
"""

import json
import math
import os
import sys
import time

from skaters.api import holt, hosking, laplace, samuelson, wald, dantzig

POLICIES = {
    "holt": holt,
    "hosking": hosking,
    "laplace": laplace,
    "samuelson": samuelson,
    "wald": wald,
    "dantzig": dantzig,
}

HORIZONS = {"yearly": 6, "quarterly": 8, "monthly": 18, "daily": 14}

# M4 competition sMAPE results for reference (from the M4 paper)
# These are overall sMAPE across all series for each frequency
M4_BASELINES = {
    "yearly": {"naive2": 16.342, "ses": 13.054, "holt": 13.601, "damped": 12.892, "theta": 12.786, "comb": 12.559},
    "quarterly": {"naive2": 12.284, "ses": 10.291, "holt": 10.175, "damped": 9.733, "theta": 9.542, "comb": 9.430},
    "monthly": {"naive2": 14.208, "ses": 13.264, "holt": 13.049, "damped": 12.621, "theta": 12.231, "comb": 12.128},
    "daily": {"naive2": 3.045, "ses": 3.043, "holt": 3.097, "damped": 3.026, "theta": 3.023, "comb": 2.980},
}


def smape(actual, predicted):
    """Symmetric Mean Absolute Percentage Error."""
    n = len(actual)
    total = 0
    for a, p in zip(actual, predicted):
        denom = abs(a) + abs(p)
        if denom > 0:
            total += 2 * abs(a - p) / denom
    return 100 * total / n


def mase(actual, predicted, train_series):
    """Mean Absolute Scaled Error (scaled by in-sample naive MAE)."""
    # In-sample naive forecast error (one-step seasonal naive)
    n_train = len(train_series)
    if n_train < 2:
        return float("inf")
    naive_errors = [abs(train_series[i] - train_series[i-1]) for i in range(1, n_train)]
    scale = sum(naive_errors) / len(naive_errors)
    if scale < 1e-10:
        return float("inf")
    forecast_errors = [abs(a - p) for a, p in zip(actual, predicted)]
    return sum(forecast_errors) / len(forecast_errors) / scale


def run_and_forecast(factory, train_series, h):
    """Train on full series, extract h-step-ahead forecasts."""
    f = factory(k=h)
    state = None
    for y in train_series:
        dists, state = f(y, state)
    # dists now contains h-step-ahead distributional forecasts
    # Use median (quantile 0.5) since sMAPE/MASE use absolute error
    return [d.quantile(0.5) for d in dists]


def main():
    with open("data/m4_sample.json") as f:
        train_data = json.load(f)
    with open("data/m4_test.json") as f:
        test_data = json.load(f)

    policy_names = list(POLICIES.keys())
    series_names = sorted([s for s in train_data if s in test_data])
    n = len(series_names)

    print(f"Evaluating {len(policy_names)} policies on {n} series (out-of-sample)...\n")

    # Collect sMAPE per policy per frequency
    smapes = {p: {freq: [] for freq in HORIZONS} for p in policy_names}
    mases = {p: {freq: [] for freq in HORIZONS} for p in policy_names}

    t0 = time.time()
    for si, sname in enumerate(series_names):
        freq = sname.split("_")[0]
        h = HORIZONS[freq]
        train = train_data[sname]
        actual = test_data[sname]

        elapsed = time.time() - t0
        rate = (si + 1) / elapsed if elapsed > 0 else 0
        eta = (n - si) / rate if rate > 0 else 0
        pct = 100 * (si + 1) / n
        sys.stdout.write(
            f"\r  [{pct:5.1f}%] {si+1}/{n} {sname:<30} "
            f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left"
        )
        sys.stdout.flush()

        for pname in policy_names:
            predicted = run_and_forecast(POLICIES[pname], train, h)
            if len(predicted) == len(actual):
                s = smape(actual, predicted)
                m = mase(actual, predicted, train)
                if math.isfinite(s):
                    smapes[pname][freq].append(s)
                if math.isfinite(m):
                    mases[pname][freq].append(m)

    print(f"\n\nDone in {time.time()-t0:.1f}s\n")

    # --- Results by frequency ---
    for freq in HORIZONS:
        n_freq = len(smapes[policy_names[0]][freq])
        if n_freq == 0:
            continue
        print(f"\n{'='*60}")
        print(f"  {freq.upper()} (h={HORIZONS[freq]}, {n_freq} series)")
        print(f"{'='*60}")
        print(f"  {'Method':<15} {'sMAPE':>8} {'MASE':>8}")
        print(f"  {'-'*15} {'-'*8} {'-'*8}")

        # Our policies
        results = []
        for p in policy_names:
            s = sum(smapes[p][freq]) / len(smapes[p][freq]) if smapes[p][freq] else float("inf")
            m = sum(mases[p][freq]) / len(mases[p][freq]) if mases[p][freq] else float("inf")
            results.append((s, p, m))
        results.sort()
        for s, p, m in results:
            print(f"  {p:<15} {s:8.3f} {m:8.3f}")

        # M4 baselines
        if freq in M4_BASELINES:
            print(f"  {'-'*15} {'-'*8}")
            print(f"  {'M4 baselines:':<15}")
            for method, val in sorted(M4_BASELINES[freq].items(), key=lambda x: x[1]):
                print(f"  {method:<15} {val:8.3f}")

    # --- Overall ---
    print(f"\n{'='*60}")
    print(f"  OVERALL ({n} series)")
    print(f"{'='*60}")
    print(f"  {'Method':<15} {'sMAPE':>8} {'MASE':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8}")
    results = []
    for p in policy_names:
        all_s = [s for freq in HORIZONS for s in smapes[p][freq]]
        all_m = [m for freq in HORIZONS for m in mases[p][freq]]
        s = sum(all_s) / len(all_s) if all_s else float("inf")
        m = sum(all_m) / len(all_m) if all_m else float("inf")
        results.append((s, p, m))
    results.sort()
    for s, p, m in results:
        print(f"  {p:<15} {s:8.3f} {m:8.3f}")


if __name__ == "__main__":
    main()
