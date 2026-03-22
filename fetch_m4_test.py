"""Download M4 test data for out-of-sample evaluation.

The M4 competition uses:
- sMAPE: symmetric Mean Absolute Percentage Error
- MASE: Mean Absolute Scaled Error (scaled by in-sample naive MAE)

Forecast horizons by frequency:
- Yearly: 6 steps ahead
- Quarterly: 8 steps ahead
- Monthly: 18 steps ahead
- Daily: 14 steps ahead

Usage:
    uv run python fetch_m4_test.py
"""

import csv
import io
import json
import os
import requests

M4_TEST_URLS = {
    "yearly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Yearly-test.csv",
    "quarterly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Quarterly-test.csv",
    "monthly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv",
    "daily": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Daily-test.csv",
}

HORIZONS = {
    "yearly": 6,
    "quarterly": 8,
    "monthly": 18,
    "daily": 14,
}


def fetch_csv(url):
    print(f"  Fetching {url.split('/')[-1]}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return list(csv.reader(io.StringIO(resp.text)))


def main():
    # Load the training sample to know which series IDs we have
    with open("data/m4_sample.json") as f:
        train = json.load(f)

    test_data = {}
    for freq, url in M4_TEST_URLS.items():
        rows = fetch_csv(url)
        h = HORIZONS[freq]
        for row in rows[1:]:
            sid = row[0]
            key = f"{freq}_{sid}"
            if key in train:
                values = [float(v.strip()) for v in row[1:h+1] if v.strip()]
                if len(values) == h:
                    test_data[key] = values

    outpath = "data/m4_test.json"
    with open(outpath, "w") as f:
        json.dump(test_data, f)
    print(f"\nSaved {len(test_data)} test series to {outpath}")

    # Also save horizons
    with open("data/m4_horizons.json", "w") as f:
        json.dump(HORIZONS, f)


if __name__ == "__main__":
    main()
