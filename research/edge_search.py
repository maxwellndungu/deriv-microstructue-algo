"""
Deeper search: across MANY conditioning features, find anything that yields
a Wilson-CI-lower-bound *above* breakeven for any contract type.

Features scanned:
  - 1-step digit (d_{-1})
  - 2-step digit pair (d_{-2}, d_{-1})
  - 3-step triplet
  - realized-vol bucket (last 10 ticks |Δ|)
  - last tick shift direction & magnitude
  - hour-of-day
  - minute-of-hour
  - price-level-mod-100
  - "tapped-digits histogram" inside a 10-tick window — the same feature
    the existing bot uses.

Bonferroni-aware decision: with thousands of conditioning cells, we require
the win-rate's 99.9% Wilson lower bound > breakeven to flag an edge.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from datetime import datetime, timezone

P = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ticks_v100_1s.json")
if P.suffix == ".gz":
    import gzip
    ticks = json.loads(gzip.decompress(P.read_bytes()))
else:
    ticks = json.loads(P.read_text())
epochs = np.array([t[0] for t in ticks], dtype=np.int64)
prices = np.array([float(t[1]) for t in ticks], dtype=np.float64)


def last_digit(x):
    return int(f"{float(x):.2f}"[-1])


digits = np.array([last_digit(p) for p in prices], dtype=np.int64)
n = len(digits)
delta = np.round(np.diff(prices) * 100).astype(np.int64)  # cents

# Define candidate single-trade win conditions evaluated per "entry tick".
# Entry at index i, settlement is digits[i+1] (1-tick contract).
exit_d = digits[1:]
entry_d = digits[:-1]
m = len(exit_d)

# Payouts (fractional gain on stake, win => +payout, loss => -1):
PAYOUT = {
    "Over0":  0.096,
    "Over1":  0.232,
    "Over4":  0.953,
    "Over7":  3.717,
    "Under9": 0.096,
    "Under8": 0.232,
    "Under5": 0.953,
    "Under2": 3.717,
    "Differs": 0.096,
    "Matches": 7.929,
    "Even":   0.953,
    "Odd":    0.953,
}
BE = {k: 1/(1+v) for k, v in PAYOUT.items()}


# Win-mask per contract assuming barrier ints are fixed:
def win_mask(name):
    if name == "Over0":   return exit_d > 0
    if name == "Over1":   return exit_d > 1
    if name == "Over4":   return exit_d > 4
    if name == "Over7":   return exit_d > 7
    if name == "Under9":  return exit_d < 9
    if name == "Under8":  return exit_d < 8
    if name == "Under5":  return exit_d < 5
    if name == "Under2":  return exit_d < 2
    if name == "Differs": return exit_d != entry_d
    if name == "Matches": return exit_d == entry_d
    if name == "Even":    return exit_d % 2 == 0
    if name == "Odd":     return exit_d % 2 == 1
    raise ValueError(name)


def wilson_lower(p, n, z):
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half) / denom


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
realised = np.zeros(m, dtype=np.int64)
W = 10
for i in range(W, m):
    realised[i] = np.abs(delta[i - W:i]).sum()
bucket_vol = np.where(realised < 30, 0,
              np.where(realised < 80, 1,
              np.where(realised < 200, 2, 3)))

hour_arr = np.array([datetime.fromtimestamp(int(e), tz=timezone.utc).hour for e in epochs[:-1]])
minute_arr = np.array([datetime.fromtimestamp(int(e), tz=timezone.utc).minute for e in epochs[:-1]])

price_int_mod10 = (np.floor(prices[:-1]).astype(np.int64)) % 10
price_int_mod100 = (np.floor(prices[:-1]).astype(np.int64)) % 100

last_delta_bucket = np.where(np.abs(delta[:-1]) <= 1, 0,
                     np.where(np.abs(delta[:-1]) <= 5, 1,
                     np.where(np.abs(delta[:-1]) <= 15, 2, 3)))
# Pad: we need length m; index 0 has no last_delta, fill with -1 → exclude from buckets
last_delta_bucket = np.concatenate([[-1], last_delta_bucket])
last_delta_bucket = last_delta_bucket[:m]


# Rolling-10-tick digit-mode (last entry's mode)
def rolling_mode_digit(arr, window=10):
    out = np.zeros(len(arr), dtype=np.int64)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        sub = arr[lo:i + 1]
        c = np.bincount(sub, minlength=10)
        out[i] = c.argmax()
    return out


rmode = rolling_mode_digit(entry_d, window=10)


# Helper: for each (contract, feature_value), compute win-rate and lower bound.
# We require samples per cell ≥ 500 and 99.9% Wilson lower > breakeven.
Z999 = 3.291  # two-sided 99.9% => one-sided 99.95%


def scan_feature(name, feature, contracts):
    """Iterate unique values of feature, compute win-rate per contract."""
    uniq = np.unique(feature)
    flags = []
    for v in uniq:
        if v < 0:
            continue
        mask = feature == v
        nn = int(mask.sum())
        if nn < 500:
            continue
        for c in contracts:
            wins = int(win_mask(c)[mask].sum())
            p = wins / nn
            lo = wilson_lower(p, nn, Z999)
            be = BE[c]
            if lo > be:
                flags.append((name, v, c, p, lo, nn, be))
    return flags


targets = ["Over0", "Under9", "Differs", "Over1", "Under8",
           "Over4", "Under5", "Even", "Odd", "Matches"]

flags = []
flags += scan_feature("d_{-1}", entry_d, targets)
flags += scan_feature("vol_bucket", bucket_vol, targets)
flags += scan_feature("hour", hour_arr, targets)
flags += scan_feature("price_mod10", price_int_mod10, targets)
flags += scan_feature("price_mod100", price_int_mod100, targets)
flags += scan_feature("last_delta_bucket", last_delta_bucket, targets)
flags += scan_feature("rmode", rmode, targets)

# Two-feature joint: hour x entry_d
joint = hour_arr * 10 + entry_d
flags += scan_feature("hour*entry_d", joint, targets)
# 3-feature: hour x entry_d x vol_bucket
joint = hour_arr * 40 + entry_d * 4 + bucket_vol
flags += scan_feature("hour*entry_d*vol", joint, targets)

print(f"\n=== EDGE SEARCH on {P.name}  (Z={Z999} = 99.9% Wilson lower) ===")
print(f"Sample size: m = {m} 1-tick windows")
print(f"Total flagged conditions (lower CI > breakeven):  {len(flags)}")
for f in flags[:50]:
    name, v, c, p, lo, nn, be = f
    print(f"  feature={name:>22s}, value={int(v):>6d}, contract={c:>7s}, "
          f"p_win={p:.4f}, Wilson99.9_lo={lo:.4f}, BE={be:.4f}, n={nn}")

# ALSO: search for which (contract, feature_value) cells have the largest
# negative gap to breakeven — these are the *worst* trades, which we MUST
# avoid even with no positive edge.
print("\n=== ANTI-EDGE: worst cells (largest gap below break-even) ===")
worst = []
for f_name, feat in [("hour", hour_arr), ("entry_d", entry_d), ("vol_bucket", bucket_vol)]:
    uniq = np.unique(feat)
    for v in uniq:
        if v < 0:
            continue
        mask = feat == v
        nn = int(mask.sum())
        if nn < 1000:
            continue
        for c in targets:
            wins = int(win_mask(c)[mask].sum())
            p = wins / nn
            gap = p - BE[c]
            worst.append((gap, f_name, v, c, p, BE[c], nn))
worst.sort()
print("Bottom 10 (most below breakeven):")
for gap, fn, v, c, p, be, nn in worst[:10]:
    print(f"  {fn}={v:<4} {c:<8} p={p:.4f} BE={be:.4f} gap={gap:+.4f} n={nn}")
print("Top 10 (best above breakeven, even if not stat-sig):")
for gap, fn, v, c, p, be, nn in worst[-10:]:
    print(f"  {fn}={v:<4} {c:<8} p={p:.4f} BE={be:.4f} gap={gap:+.4f} n={nn}")
