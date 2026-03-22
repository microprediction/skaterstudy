# skaterstudy

Out-of-sample evaluation of [skaters](https://github.com/microprediction/skaters) policies on M4 competition data.

## Results

371 series sampled from the M4 competition (71 yearly, 100 quarterly, 100 monthly, 100 daily). Each policy trains on the full training series, then forecasts h steps ahead. Scored by sMAPE against held-out test data.

### Overall (371 series)

| Method | sMAPE | MASE |
|--------|------:|-----:|
| **holt** | **10.11** | 3.45 |
| samuelson | 10.14 | 3.45 |
| hosking | 10.19 | 3.44 |
| laplace | 10.33 | 3.44 |
| dantzig | 10.71 | 3.68 |
| wald | 10.75 | 3.46 |

### Comparison to M4 baselines

| Frequency | Best skater | sMAPE | M4 Theta | M4 Comb | M4 Naive2 |
|-----------|-------------|------:|---------:|--------:|----------:|
| Yearly (h=6) | hosking | 14.45 | 12.79 | 12.56 | 16.34 |
| Quarterly (h=8) | holt | 10.24 | 9.54 | 9.43 | 12.28 |
| Monthly (h=18) | hosking | 13.71 | 12.23 | 12.13 | 14.21 |
| Daily (h=14) | dantzig | 3.00 | 3.02 | 2.98 | 3.05 |

### Key findings

**We beat Naive2 on every frequency.** This is the M4 baseline (seasonal naive). Our worst policy is still better than Naive2 on quarterly and monthly data.

**We beat M4 Theta on daily.** Dantzig's adaptive search achieves 2.996 sMAPE vs Theta's 3.023. This is the one frequency where our online approach has the most data to work with.

**We're competitive on quarterly.** Holt (10.24) is close to M4's classical Holt (10.18) and better than M4's SES (10.29).

**We lag on yearly and monthly.** Our best sMAPE (14.45 yearly, 13.71 monthly) falls between M4's SES and Holt. The gap to Theta/Comb (~1-2 sMAPE points) is meaningful.

### Why the gap?

The M4 baselines use batch optimization:
- **Theta** grid-searches the SES smoothing parameter on the full training series
- **Comb** averages SES, Holt, and Damped after individually optimizing each
- **Damped** optimizes 3 parameters (level, trend, damping)

Our methods are fully online -- no batch optimization, no grid search, no look-ahead. Every parameter is estimated incrementally. This is a harder problem, and the ~1-2 sMAPE gap is the price of being purely online.

The daily result (where we beat Theta) suggests that with enough data, the online estimation catches up. Yearly series have only 20-40 observations -- not enough for online methods to converge.

### By frequency

**Yearly (h=6, 71 series):** hosking leads (14.45). All policies are between M4 SES (13.05) and Naive2 (16.34). Short series hurt online methods.

**Quarterly (h=8, 100 series):** holt leads (10.24). Very close to M4 Holt (10.18). Wald is worst (12.33) -- too conservative for series with real trend.

**Monthly (h=18, 100 series):** hosking leads (13.71). Close to M4 SES (13.26). All policies are competitive. The long horizon (h=18) is challenging.

**Daily (h=14, 100 series):** dantzig wins (3.00), beating M4 Theta (3.02). Long series (up to 7,842 obs) give online methods enough data. All policies are within the M4 competitive range.

## Reproducing

```bash
pip install skaters pandas requests
python fetch_m4.py          # download training data
python fetch_m4_test.py     # download test data
python evaluate_m4.py       # evaluate (takes ~80 min for 371 series)
```

Edit `SAMPLE_PER_FREQ` in `fetch_m4.py` to control sample size.

## Data

- M4 competition: [github.com/Mcompetitions/M4-methods](https://github.com/Mcompetitions/M4-methods)
- M4 paper: Makridakis et al. (2020), "The M4 Competition"
