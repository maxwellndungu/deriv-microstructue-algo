"""
Lag-correction audit.

The previous audit/edge_search both assumed:
    decide at tick i  →  settle on digit at tick i+1   (LAG 1)

Per Deriv official ToS §2.2.3.1:
    "For Digital Options... the entry spot is defined as the *next tick* after
    our servers process the contract."

So for a 1-tick DIGITDIFF (or OVER/UNDER) bought at tick i:
    entry spot = tick i+1
    exit spot  = tick i+2   (1 tick *after* entry spot)
    settle on digit at tick i+2   (LAG 2 from decision)

The barrier (for DIGITDIFF) is fixed at buy time. The natural strategy that
references "the current digit" must commit barrier = digit(i) at decision —
you cannot reference digit(i+1) because you do not see it before buying.

This script re-runs the same suite with the CORRECT lag, and side-by-sides
lag-1 vs lag-2 results.
"""
from __future__ import annotations
import gzip
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

P = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "ticks_v100_1s.json.gz"
if P.suffix == ".gz":
    ticks = json.loads(gzip.decompress(P.read_bytes()))
else:
    ticks = json.loads(P.read_text())

epochs = np.array([t[0] for t in ticks], dtype=np.int64)
prices = np.array([float(t[1]) for t in ticks], dtype=np.float64)


def last_digit(x):
    return int(f"{float(x):.2f}"[-1])


digits = np.array([last_digit(p) for p in prices], dtype=np.int64)
n = len(digits)
print(f"n_ticks = {n}    span = {(epochs[-1]-epochs[0])/3600:.2f} h")


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def wilson_lower(p, n, z):
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half) / denom


PAYOUT = {
    "Over0":   0.096,
    "Under9":  0.096,
    "Differs": 0.096,
    "Over4":   0.953,
    "Under5":  0.953,
    "Even":    0.953,
    "Odd":     0.953,
    "Matches": 7.929,
    "Over7":   3.717,
}
BE = {k: 1 / (1 + v) for k, v in PAYOUT.items()}


# ---------------------------------------------------------------------------
# Define the four scenarios.
#   decision-tick index: i
#   strategy ref digit (barrier for DIGITDIFF): digit(i)
#
#   LAG1: exit_d[i] = digit(i+1), entry_ref[i] = digit(i)
#   LAG2: exit_d[i] = digit(i+2), entry_ref[i] = digit(i)
# ---------------------------------------------------------------------------
def slice_for_lag(lag):
    """Return (decision_digit, exit_digit, decision_indices_into_full_arr)."""
    assert lag >= 1
    end = n - lag
    decision_digit = digits[:end]                       # digit at decision tick
    exit_digit     = digits[lag:lag + end]               # digit at exit tick
    return decision_digit, exit_digit


# ---------------------------------------------------------------------------
# 1. Marginal win-rates  (static contracts, no signal)
# ---------------------------------------------------------------------------
print("\n--- 1. Static-bet win rates, lag-1 vs lag-2 ---")
print(f"{'contract':>10s} | {'BE':>6s} | {'lag1 p_win':>10s} {'lag1 EV/$':>10s} | "
      f"{'lag2 p_win':>10s} {'lag2 EV/$':>10s}")
for lag in (1, 2):
    pass  # build per-lag below
for c in ("Over0", "Under9", "Over4", "Under5", "Even", "Odd",
          "Over7", "Matches", "Differs"):
    line = f"{c:>10s} | {BE[c]:.4f} |"
    for lag in (1, 2):
        d_dec, d_exit = slice_for_lag(lag)
        if c == "Over0":   mask = d_exit > 0
        elif c == "Under9": mask = d_exit < 9
        elif c == "Over4":  mask = d_exit > 4
        elif c == "Under5": mask = d_exit < 5
        elif c == "Even":   mask = d_exit % 2 == 0
        elif c == "Odd":    mask = d_exit % 2 == 1
        elif c == "Over7":  mask = d_exit > 7
        elif c == "Matches": mask = d_exit == d_dec
        elif c == "Differs": mask = d_exit != d_dec
        p = mask.mean()
        ev = p * PAYOUT[c] - (1 - p)
        line += f"  {p:.4f}   {ev:+.4f}    |" if lag == 1 else f"  {p:.4f}   {ev:+.4f}"
    print(line)


# ---------------------------------------------------------------------------
# 2. Conditional Differs by entry digit, lag-1 vs lag-2
# ---------------------------------------------------------------------------
print("\n--- 2. P(Differs win | barrier = digit(decision)), per digit ---")
print(f"{'digit':>5s} | {'lag1 n':>7s} {'lag1 p':>7s} {'lag1 CI95_lo':>13s} | "
      f"{'lag2 n':>7s} {'lag2 p':>7s} {'lag2 CI95_lo':>13s}   BE={BE['Differs']:.4f}")
for d in range(10):
    cells = {}
    for lag in (1, 2):
        d_dec, d_exit = slice_for_lag(lag)
        idx = d_dec == d
        if idx.sum() < 100:
            cells[lag] = None
            continue
        wins = (d_exit[idx] != d).sum()
        tot = int(idx.sum())
        p = wins / tot
        lo95 = wilson_lower(p, tot, 1.96)
        cells[lag] = (tot, p, lo95)
    c1 = cells[1]; c2 = cells[2]
    print(f"  d={d}  | {c1[0]:>7d} {c1[1]:.4f}    {c1[2]:.4f}      | "
          f"{c2[0]:>7d} {c2[1]:.4f}    {c2[2]:.4f}")


# ---------------------------------------------------------------------------
# 3. Conditional Over0 by (d_{-2}, d_{-1}) for lag-1 vs lag-2
# ---------------------------------------------------------------------------
print("\n--- 3. Conditional Over0 win-rate by (d_{-1}, d_{0}), lag-1 vs lag-2 ---")
print("(Decision uses last two observed digits; bet d_exit > 0.  BE=0.9124)")
records = []
for a in range(10):
    for b in range(10):
        rec = {"ab": (a, b)}
        for lag in (1, 2):
            # Decision time t means we have observed digits up to and including t.
            # Conditioning on (digit_{t-1}, digit_t) = (a, b).
            # Exit is digit_{t+lag}.
            # Valid t ranges:  1 <= t,   t+lag < n
            t_idx = np.arange(1, n - lag)
            mask = (digits[t_idx - 1] == a) & (digits[t_idx] == b)
            nn = int(mask.sum())
            if nn < 200:
                rec[lag] = None
                continue
            d_exit_t = digits[t_idx + lag][mask]
            p = (d_exit_t > 0).mean()
            lo = wilson_lower(p, nn, 3.291)   # 99.9 % Wilson
            rec[lag] = (p, lo, nn)
        records.append(rec)

# Best lag-2 cells
have_both = [r for r in records if r[2] is not None]
have_both.sort(key=lambda r: -r[2][0])
print(f"\nLAG-2 top 8 conditioning cells (best win-rate):")
print(f"{'a,b':>5s} | {'lag1 p':>7s} {'lag1 CI99.9_lo':>14s} | {'lag2 p':>7s} {'lag2 CI99.9_lo':>14s}  n")
for r in have_both[:8]:
    a, b = r["ab"]
    l1 = r.get(1)
    l2 = r.get(2)
    if l1 is None or l2 is None:
        continue
    print(f"  {a},{b} | {l1[0]:.4f}    {l1[1]:.4f}       | {l2[0]:.4f}    {l2[1]:.4f}     {l2[2]}")
print(f"\nLAG-2 bottom 8 conditioning cells (worst win-rate):")
for r in have_both[-8:]:
    a, b = r["ab"]
    l1 = r.get(1); l2 = r.get(2)
    print(f"  {a},{b} | {l1[0]:.4f}    {l1[1]:.4f}       | {l2[0]:.4f}    {l2[1]:.4f}     {l2[2]}")

# Count cells where lag-2 lower bound > BE (would be flagged edges)
flagged_l2 = [r for r in have_both if r[2][1] > BE["Over0"]]
flagged_l1 = [r for r in have_both if r[1] is not None and r[1][1] > BE["Over0"]]
print(f"\nCells with 99.9% Wilson lower > 0.9124  (real edge):")
print(f"   LAG-1 count: {len(flagged_l1)}")
print(f"   LAG-2 count: {len(flagged_l2)}")


# ---------------------------------------------------------------------------
# 4. Lag-1 vs lag-2 autocorrelation of digit, parity and OverHalf
# ---------------------------------------------------------------------------
def ac(x, k):
    x = np.asarray(x, dtype=float) - np.mean(x)
    return float((x[:-k] * x[k:]).sum() / (x * x).sum())


print("\n--- 4. Autocorrelations at lags 1..10 (significance band = ±{:.4f}) ---"
      .format(1.96 / math.sqrt(n)))
print("lag | rho_digit  | rho_parity | rho_overhalf")
even = (digits % 2 == 0).astype(float)
over = (digits >= 5).astype(float)
for k in range(1, 11):
    print(f"{k:3d} | {ac(digits,k):+.5f}   | {ac(even,k):+.5f}   | {ac(over,k):+.5f}")


# ---------------------------------------------------------------------------
# 5. Joint chi-square: independence of (d_t, d_{t+2}) — i.e. Markov-lag-2
# ---------------------------------------------------------------------------
print("\n--- 5. Chi-square independence of (d_t, d_{t+lag}) ---")
for lag in (1, 2):
    M = np.zeros((10, 10), dtype=np.int64)
    for i in range(n - lag):
        M[digits[i], digits[i + lag]] += 1
    row = M.sum(axis=1, keepdims=True)
    col = M.sum(axis=0, keepdims=True)
    tot = M.sum()
    exp = row @ col / tot
    chi = float(((M - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
    p = 1 - stats.chi2.cdf(chi, 81)
    print(f"   lag={lag}: chi2={chi:.2f}  df=81  p={p:.4f}")


# ---------------------------------------------------------------------------
# 6. Full bonferroni-aware edge search (lag-2 corrected)
# ---------------------------------------------------------------------------
print("\n--- 6. Bonferroni-aware edge search using LAG-2 settlement ---")

# decision = i,  exit = i + 2
# all features computed up to and including decision tick i.
i_max = n - 2
d_dec = digits[:i_max]                 # decision-tick digit (= digit at i)
d_prev = digits[:-1][:i_max]           # placeholder, see below
d_exit = digits[2:2 + i_max]            # exit-tick digit  (= digit at i+2)

# Features at decision time i — only quantities visible at time i.
delta_cents = np.round(np.diff(prices) * 100).astype(np.int64)      # length n-1
# delta_cents[k] = price[k+1] - price[k]; last delta visible at time i is delta_cents[i-1].

# realized vol = sum |delta| over last 10 deltas ending at time i-1
W = 10
abs_delta = np.abs(delta_cents)
realised = np.zeros(n, dtype=np.int64)
for i in range(W, n):
    realised[i] = abs_delta[i - W:i].sum()
realised_dec = realised[:i_max]
bucket_vol = np.where(realised_dec < 30, 0,
              np.where(realised_dec < 80, 1,
              np.where(realised_dec < 200, 2, 3)))

# last delta bucket at decision i:  delta_cents[i-1]
last_delta = np.zeros(n, dtype=np.int64)
last_delta[1:] = abs_delta
last_delta_dec = last_delta[:i_max]
last_delta_bucket = np.where(last_delta_dec <= 1, 0,
                     np.where(last_delta_dec <= 5, 1,
                     np.where(last_delta_dec <= 15, 2, 3)))

hour_dec = np.array([datetime.fromtimestamp(int(epochs[i]), tz=timezone.utc).hour
                      for i in range(i_max)])
price_mod10  = (np.floor(prices[:i_max]).astype(np.int64)) % 10
price_mod100 = (np.floor(prices[:i_max]).astype(np.int64)) % 100

# rolling-10-tick digit mode (mode of last 10 digits ending at i)
def rolling_mode(arr, window=10):
    out = np.zeros(len(arr), dtype=np.int64)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        sub = arr[lo:i + 1]
        out[i] = np.bincount(sub, minlength=10).argmax()
    return out
rmode = rolling_mode(d_dec, window=10)


def win_mask(name):
    if name == "Over0":   return d_exit > 0
    if name == "Under9":  return d_exit < 9
    if name == "Differs": return d_exit != d_dec
    if name == "Over4":   return d_exit > 4
    if name == "Under5":  return d_exit < 5
    if name == "Even":    return d_exit % 2 == 0
    if name == "Odd":     return d_exit % 2 == 1
    if name == "Matches": return d_exit == d_dec
    if name == "Over7":   return d_exit > 7
    raise ValueError(name)


targets = ["Over0", "Under9", "Differs", "Over4", "Under5", "Even", "Odd"]
Z999 = 3.291


def scan_feature(label, feature):
    out = []
    for v in np.unique(feature):
        mask = feature == v
        nn = int(mask.sum())
        if nn < 500:
            continue
        for c in targets:
            wins = int(win_mask(c)[mask].sum())
            p = wins / nn
            lo = wilson_lower(p, nn, Z999)
            if lo > BE[c]:
                out.append((label, int(v), c, p, lo, nn, BE[c]))
    return out


flagged = []
flagged += scan_feature("d_decision", d_dec)
flagged += scan_feature("vol_bucket", bucket_vol)
flagged += scan_feature("hour", hour_dec)
flagged += scan_feature("price_mod10", price_mod10)
flagged += scan_feature("price_mod100", price_mod100)
flagged += scan_feature("last_delta_bucket", last_delta_bucket)
flagged += scan_feature("rmode", rmode)

# pair feature: previous two digits
prev2 = np.concatenate([[-1], d_dec[:-1]]) * 10 + d_dec   # (d_{t-1}, d_t)
prev2 = prev2[1:i_max]
# need to align win_mask too
flagged_pair = []
for v in np.unique(prev2):
    if v < 0:
        continue
    mask = np.concatenate([[False], prev2 == v])  # align with d_dec
    nn = int(mask.sum())
    if nn < 500:
        continue
    for c in targets:
        wins = int(win_mask(c)[mask].sum())
        p = wins / nn
        lo = wilson_lower(p, nn, Z999)
        if lo > BE[c]:
            flagged_pair.append(("pair", int(v), c, p, lo, nn, BE[c]))
flagged += flagged_pair

# joint hour x d_decision
joint = hour_dec * 10 + d_dec
flagged += scan_feature("hour*d_dec", joint)

print(f"\nFlagged conditioning cells (Wilson 99.9% lower > BE) with LAG-2 settlement: {len(flagged)}")
for f in flagged[:30]:
    label, v, c, p, lo, nn, be = f
    print(f"  {label:>20s}={v:<6d} {c:>8s}  p={p:.4f}  lo99.9={lo:.4f}  BE={be:.4f}  n={nn}")

print("\n=== DONE ===")
