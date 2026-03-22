"""Daily M4 race: dantzig vs baselines with live leaderboard.

Runs dantzig and simple baselines (naive, SES, drift) on all daily
M4 series, printing an updating leaderboard after each series.

Usage:
    ~/.local/bin/uv run python daily_race.py
"""

import json
import math
import os
import sys
import time


def naive_forecast(train, h):
    """Naive: repeat last value."""
    return [train[-1]] * h


def ses_forecast(train, h, alpha=0.1):
    """Simple exponential smoothing."""
    level = train[0]
    for y in train[1:]:
        level = alpha * y + (1 - alpha) * level
    return [level] * h


def drift_forecast(train, h):
    """Random walk with drift: last value + h * average increment."""
    n = len(train)
    if n < 2:
        return [train[-1]] * h
    avg_inc = (train[-1] - train[0]) / (n - 1)
    return [train[-1] + (i + 1) * avg_inc for i in range(h)]


def theta_forecast(train, h, alpha=0.1):
    """Classical Theta: SES + half OLS slope."""
    n = len(train)
    # SES
    level = train[0]
    for y in train[1:]:
        level = alpha * y + (1 - alpha) * level
    # OLS slope
    sum_t = sum(range(1, n + 1))
    sum_t2 = sum(t * t for t in range(1, n + 1))
    sum_y = sum(train)
    sum_ty = sum((t + 1) * y for t, y in enumerate(train))
    denom = n * sum_t2 - sum_t ** 2
    slope = (n * sum_ty - sum_t * sum_y) / denom if abs(denom) > 1e-12 else 0
    return [level + (i + 1) * slope / 2 for i in range(h)]


def smape(actual, predicted):
    total = 0
    for a, p in zip(actual, predicted):
        denom = abs(a) + abs(p)
        if denom > 0:
            total += 2 * abs(a - p) / denom
    return 100 * total / len(actual)


def mase(actual, predicted, train_series):
    """Mean Absolute Scaled Error."""
    n_train = len(train_series)
    if n_train < 2:
        return float("inf")
    naive_errors = [abs(train_series[i] - train_series[i-1]) for i in range(1, n_train)]
    scale = sum(naive_errors) / len(naive_errors)
    if scale < 1e-10:
        return float("inf")
    forecast_errors = [abs(a - p) for a, p in zip(actual, predicted)]
    return sum(forecast_errors) / len(forecast_errors) / scale


def print_leaderboard(cum_smape, cum_mase, n_done):
    """Print a compact leaderboard sorted by sMAPE."""
    methods = sorted(cum_smape.keys(), key=lambda m: cum_smape[m] / n_done)
    print(f"\n  {'Rank':<5} {'Method':<20} {'sMAPE':>8} {'MASE':>8}  {'Series':>6}")
    print(f"  {'-'*5} {'-'*20} {'-'*8} {'-'*8}  {'-'*6}")
    for rank, m in enumerate(methods, 1):
        avg_s = cum_smape[m] / n_done
        avg_m = cum_mase[m] / n_done if cum_mase[m] < float("inf") else float("inf")
        bar = '#' * max(1, int(avg_s * 3))
        print(f"  {rank:<5} {m:<20} {avg_s:8.3f} {avg_m:8.3f}  {n_done:>6}  {bar}")
    print()


def main():
    from skaters.api import dantzig, holt, hosking, laplace, samuelson, wald

    with open("data/m4_sample.json") as f:
        train_data = json.load(f)
    with open("data/m4_test.json") as f:
        test_data = json.load(f)

    h = 14  # daily horizon

    # Get daily series only
    daily = sorted([s for s in train_data if s.startswith("daily_") and s in test_data])
    n = len(daily)
    print(f"Daily M4 race: {n} series, h={h}\n")

    # Methods: baselines + skaters policies
    skater_policies = {
        "dantzig": dantzig,
        "holt": holt,
        "hosking": hosking,
        "laplace": laplace,
        "samuelson": samuelson,
        "wald": wald,
    }

    baseline_methods = {
        "naive": naive_forecast,
        "ses(0.1)": lambda tr, h: ses_forecast(tr, h, alpha=0.1),
        "ses(0.3)": lambda tr, h: ses_forecast(tr, h, alpha=0.3),
        "drift": drift_forecast,
        "theta(0.1)": lambda tr, h: theta_forecast(tr, h, alpha=0.1),
        "theta(0.3)": lambda tr, h: theta_forecast(tr, h, alpha=0.3),
    }

    # Cumulative scores
    all_methods = list(baseline_methods.keys()) + list(skater_policies.keys())
    cum_smape = {m: 0.0 for m in all_methods}
    cum_mase = {m: 0.0 for m in all_methods}
    n_done = 0

    t0 = time.time()
    for si, sname in enumerate(daily):
        train = train_data[sname]
        actual = test_data[sname]
        n_obs = len(train)

        # Baselines (instant)
        for mname, mfunc in baseline_methods.items():
            pred = mfunc(train, h)
            if len(pred) == len(actual):
                cum_smape[mname] += smape(actual, pred)
                m = mase(actual, pred, train)
                if math.isfinite(m):
                    cum_mase[mname] += m

        # Skaters policies
        for pname, factory in skater_policies.items():
            f = factory(k=h)
            state = None
            for y in train:
                dists, state = f(y, state)
            pred = [d.mean for d in dists]
            if len(pred) == len(actual):
                cum_smape[pname] += smape(actual, pred)
                m = mase(actual, pred, train)
                if math.isfinite(m):
                    cum_mase[pname] += m

        n_done += 1
        elapsed = time.time() - t0

        # Print leaderboard every 5 series or at the end
        if n_done % 5 == 0 or n_done == n:
            sys.stdout.write(f"\033[2J\033[H")  # clear screen
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n - n_done) / rate if rate > 0 else 0
            print(f"Daily M4 Race — {n_done}/{n} series ({elapsed:.0f}s, ~{eta:.0f}s left)")
            print(f"Last: {sname} ({n_obs} obs)")
            print_leaderboard(cum_smape, cum_mase, n_done)

    # Final summary
    print("=" * 50)
    print(f"FINAL RESULTS ({n_done} daily series, h={h})")
    print("=" * 50)
    print_leaderboard(cum_smape, cum_mase, n_done)


if __name__ == "__main__":
    main()
