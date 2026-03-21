"""Download M4 competition data and extract a representative sample.

Downloads the M4 training data (daily, monthly, quarterly, yearly),
selects a diverse sample, and saves as a lightweight JSON file
that can be embedded in the skaters package later.

Usage:
    uv run python fetch_m4.py
"""

import csv
import io
import json
import os
import random
import requests

M4_URLS = {
    "yearly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Yearly-train.csv",
    "quarterly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Quarterly-train.csv",
    "monthly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv",
    "daily": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Daily-train.csv",
}

SAMPLE_PER_FREQ = 25  # 25 series per frequency = 100 total


def fetch_csv(url: str) -> list[list[str]]:
    """Download and parse a CSV from URL."""
    print(f"  Fetching {url.split('/')[-1]}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    reader = csv.reader(io.StringIO(resp.text))
    return list(reader)


def parse_series(rows: list[list[str]], n_sample: int, seed: int = 42) -> dict[str, list[float]]:
    """Parse M4 CSV rows into {id: [values]} dict, sampling n_sample series."""
    # First row is header (V1, V2, ...), skip it
    data_rows = rows[1:]
    random.seed(seed)
    sampled = random.sample(data_rows, min(n_sample, len(data_rows)))

    series = {}
    for row in sampled:
        series_id = row[0]
        values = []
        for v in row[1:]:
            v = v.strip()
            if v == "" or v == "NA":
                break
            values.append(float(v))
        if len(values) >= 20:  # skip very short series
            series[series_id] = values

    return series


def main():
    os.makedirs("data", exist_ok=True)

    all_series = {}
    for freq, url in M4_URLS.items():
        rows = fetch_csv(url)
        sampled = parse_series(rows, n_sample=SAMPLE_PER_FREQ)
        print(f"  {freq}: sampled {len(sampled)} series")
        for sid, vals in sampled.items():
            all_series[f"{freq}_{sid}"] = vals

    # Save as JSON
    outpath = "data/m4_sample.json"
    with open(outpath, "w") as f:
        json.dump(all_series, f)
    print(f"\nSaved {len(all_series)} series to {outpath}")
    print(f"File size: {os.path.getsize(outpath) / 1024:.0f} KB")

    # Print summary
    lengths = [len(v) for v in all_series.values()]
    print(f"Lengths: min={min(lengths)}, max={max(lengths)}, median={sorted(lengths)[len(lengths)//2]}")


if __name__ == "__main__":
    main()
