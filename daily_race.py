"""Daily M4 race: skaters policies vs baselines with live progress.

Usage:
    ~/.local/bin/uv run python daily_race.py
"""

import json
import math
import sys
import time


def naive_forecast(train, h):
    return [train[-1]] * h


def ses_forecast(train, h, alpha=0.1):
    level = train[0]
    for y in train[1:]:
        level = alpha * y + (1 - alpha) * level
    return [level] * h


def drift_forecast(train, h):
    n = len(train)
    if n < 2:
        return [train[-1]] * h
    avg_inc = (train[-1] - train[0]) / (n - 1)
    return [train[-1] + (i + 1) * avg_inc for i in range(h)]


def theta_forecast(train, h, alpha=0.1):
    n = len(train)
    level = train[0]
    for y in train[1:]:
        level = alpha * y + (1 - alpha) * level
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
    n_train = len(train_series)
    if n_train < 2:
        return float("inf")
    naive_errors = [abs(train_series[i] - train_series[i - 1]) for i in range(1, n_train)]
    scale = sum(naive_errors) / len(naive_errors)
    if scale < 1e-10:
        return float("inf")
    forecast_errors = [abs(a - p) for a, p in zip(actual, predicted)]
    return sum(forecast_errors) / len(forecast_errors) / scale


def main():
    from skaters.api import dantzig, holt, hosking, laplace, samuelson, wald

    with open("data/m4_sample.json") as f:
        train_data = json.load(f)
    with open("data/m4_test.json") as f:
        test_data = json.load(f)

    h = 14

    daily = sorted([s for s in train_data if s.startswith("daily_") and s in test_data])
    n = len(daily)

    skater_policies = [
        ("dantzig", dantzig),
        ("holt", holt),
        ("hosking", hosking),
        ("laplace", laplace),
        ("samuelson", samuelson),
        ("wald", wald),
    ]

    baseline_methods = [
        ("naive", naive_forecast),
        ("ses(0.1)", lambda tr, h: ses_forecast(tr, h, alpha=0.1)),
        ("ses(0.3)", lambda tr, h: ses_forecast(tr, h, alpha=0.3)),
        ("drift", drift_forecast),
        ("theta(0.1)", lambda tr, h: theta_forecast(tr, h, alpha=0.1)),
        ("theta(0.3)", lambda tr, h: theta_forecast(tr, h, alpha=0.3)),
    ]

    all_names = [m[0] for m in baseline_methods] + [p[0] for p in skater_policies]
    cum_smape = {m: 0.0 for m in all_names}
    cum_mase = {m: 0.0 for m in all_names}
    n_done = 0

    print(f"Daily M4 Race: {n} series, h={h}")
    print(f"Methods: {', '.join(all_names)}")
    print("=" * 80)

    t0 = time.time()
    for si, sname in enumerate(daily):
        train = train_data[sname]
        actual = test_data[sname]
        n_obs = len(train)

        print(f"\n[{si+1}/{n}] {sname} ({n_obs} obs)")

        # Baselines (instant)
        for mname, mfunc in baseline_methods:
            pred = mfunc(train, h)
            if len(pred) == len(actual):
                s = smape(actual, pred)
                m = mase(actual, pred, train)
                cum_smape[mname] += s
                if math.isfinite(m):
                    cum_mase[mname] += m
                print(f"  {mname:<15} sMAPE={s:6.2f}")
                sys.stdout.flush()

        # Skaters policies (slow — show each as it completes)
        for pname, factory in skater_policies:
            t1 = time.time()
            f = factory(k=h)
            state = None
            for y in train:
                dists, state = f(y, state)
            pred = [d.quantile(0.5) for d in dists]
            dt = time.time() - t1
            if len(pred) == len(actual):
                s = smape(actual, pred)
                m = mase(actual, pred, train)
                cum_smape[pname] += s
                if math.isfinite(m):
                    cum_mase[pname] += m
                print(f"  {pname:<15} sMAPE={s:6.2f}  ({dt:.1f}s)")
                sys.stdout.flush()

        n_done += 1

        # Leaderboard after each series
        ranked = sorted(all_names, key=lambda m: cum_smape[m] / n_done)
        top = " | ".join(f"{i+1}.{m} {cum_smape[m]/n_done:.2f}" for i, m in enumerate(ranked[:5]))
        print(f"  >> {top}")
        sys.stdout.flush()

    # Final
    print("\n" + "=" * 80)
    print(f"FINAL ({n_done} series)")
    print("=" * 80)
    ranked = sorted(all_names, key=lambda m: cum_smape[m] / n_done)
    print(f"  {'Rank':<5} {'Method':<15} {'sMAPE':>8} {'MASE':>8}")
    print(f"  {'-'*5} {'-'*15} {'-'*8} {'-'*8}")
    for rank, m in enumerate(ranked, 1):
        avg_s = cum_smape[m] / n_done
        avg_m = cum_mase[m] / n_done
        print(f"  {rank:<5} {m:<15} {avg_s:8.3f} {avg_m:8.3f}")


if __name__ == "__main__":
    main()
