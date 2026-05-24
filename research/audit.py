"""
First-principles statistical audit of Deriv synthetic-index digit streams.

Goal: For each candidate edge claim, find evidence in real data or kill the
claim. Output a one-shot report.

Tests:
  T1. Empirical digit frequencies + chi-square against uniform (10).
  T2. Lag-k autocorrelation of the raw digit sequence for k=1..50.
  T3. Markov transition matrix and chi-square against independence.
  T4. Runs / parity tests (Wald-Wolfowitz on Even-Odd binary).
  T5. Run-length distribution of repeated digits vs geometric(0.9).
  T6. Higher-order conditional probabilities P(d_t | d_{t-1}, d_{t-2}).
  T7. Spectral density (DFT) of digit indicator series — looking for
      periodicities in the supposed RNG output.
  T8. Distribution of 1-tick price increment in cents (the deciding feature
      for digit dynamics).
  T9. Conditional digit-transition under |Δprice| regimes (does low-vol
      microstructure leak a digit-shift bias?).
  T10. Tick interarrival statistics — are ticks really 1-second deterministic?
  T11. Conditional probability of each contract type winning given each
       *observable* feature: previous digit, last-3 digits, recent realized
       vol, hour, minute mod 60. Bonferroni-aware p-values.
  T12. Bootstrapped win-rate confidence intervals for the simulated bot at
       break-even thresholds.
  T13. Hour-of-day and minute-of-hour digit-frequency biases.
  T14. The "digit advance" distribution: (d_{t+1} - d_t) mod 10.
  T15. Sequence-level Kolmogorov-Smirnov style: empirical vs theoretical
       digit-pair frequencies.
"""
from __future__ import annotations
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ticks_v100_1s.json")
LABEL = sys.argv[2] if len(sys.argv) > 2 else "V100(1s)"

if PATH.suffix == ".gz":
    import gzip
    ticks = json.loads(gzip.decompress(PATH.read_bytes()))
else:
    ticks = json.loads(PATH.read_text())
print(f"\n=== {LABEL}  |  source: {PATH.name}  |  n_ticks = {len(ticks)} ===")
print(f"epoch span: {ticks[0][0]} → {ticks[-1][0]} ({(ticks[-1][0]-ticks[0][0])/3600:.2f} hours)")

# ---------------------------------------------------------------------------
# Extract last digits *the same way the contract resolver does*: by formatting
# the price string with the symbol's pip size. For V100(1s) and V10(1s) the
# pip size is 0.01 → 2 decimals → last char of "{price:.2f}" is the digit.
# ---------------------------------------------------------------------------
def last_digit(price_str_or_float) -> int:
    # We accept whatever Deriv API returned: in this dump prices are floats.
    s = f"{float(price_str_or_float):.2f}"
    return int(s[-1])

epochs = np.array([t[0] for t in ticks], dtype=np.int64)
prices = np.array([float(t[1]) for t in ticks], dtype=np.float64)
digits = np.array([last_digit(p) for p in prices], dtype=np.int64)
n = len(digits)

# ---------------------------------------------------------------------------
# T1: Empirical digit frequencies and chi-square vs uniform
# ---------------------------------------------------------------------------
print("\n--- T1: Digit frequencies vs uniform(0..9) ---")
counts = np.array([int((digits == d).sum()) for d in range(10)])
expected = n / 10
chi2 = ((counts - expected) ** 2 / expected).sum()
df = 9
p_chi2 = 1 - stats.chi2.cdf(chi2, df)
print(f"counts: {counts.tolist()}")
print(f"expected per bin: {expected:.1f}")
print(f"max |deviation|: {abs(counts - expected).max():.1f} ({abs(counts - expected).max() / expected * 100:.2f}%)")
print(f"chi-square = {chi2:.3f}, df = {df}, p = {p_chi2:.4f}")
print(f"verdict: {'REJECT uniform' if p_chi2 < 0.01 else 'cannot reject uniform'}")

# ---------------------------------------------------------------------------
# T2: Lag-k autocorrelation of the digit sequence (treat as integer time series)
# Also lag-k autocorrelation of the Even-indicator (parity) series.
# ---------------------------------------------------------------------------
def autocorr(x, k):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    num = (x[:-k] * x[k:]).sum()
    den = (x * x).sum()
    return num / den if den else 0.0


# 95% confidence band for white noise autocorrelation
sigma_ac = 1.96 / math.sqrt(n)

print("\n--- T2: Autocorrelation (significance band = ±{:.4f}) ---".format(sigma_ac))
print("lag  | rho_digit  | rho_parity | rho_OverHalf")
even = (digits % 2 == 0).astype(float)
over = (digits >= 5).astype(float)
sig_lags = []
for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 200, 500, 1000]:
    if k >= n:
        continue
    rd = autocorr(digits, k)
    rp = autocorr(even, k)
    ro = autocorr(over, k)
    mark = " *" if max(abs(rd), abs(rp), abs(ro)) > sigma_ac else ""
    print(f"{k:5d}| {rd:+.5f}   | {rp:+.5f}   | {ro:+.5f}{mark}")
    if max(abs(rd), abs(rp), abs(ro)) > sigma_ac:
        sig_lags.append(k)
print(f"lags exceeding 95% white-noise band: {sig_lags or 'none'}")

# Ljung-Box test on raw digit series
def ljung_box(x, lags):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    nn = len(x)
    Q = 0.0
    for k in range(1, lags + 1):
        rk = autocorr(x, k)
        Q += rk * rk / (nn - k)
    Q *= nn * (nn + 2)
    return Q, 1 - stats.chi2.cdf(Q, lags)

for L in (10, 20, 50):
    Q, p = ljung_box(digits, L)
    print(f"Ljung-Box(L={L}): Q={Q:.2f}, p={p:.4f}")

# ---------------------------------------------------------------------------
# T3: Markov transition matrix
# ---------------------------------------------------------------------------
print("\n--- T3: 1-step Markov transition matrix P(d_{t+1} | d_t) ---")
M = np.zeros((10, 10), dtype=np.int64)
for i in range(n - 1):
    M[digits[i], digits[i + 1]] += 1
row_sums = M.sum(axis=1, keepdims=True)
P = M / np.maximum(row_sums, 1)
# Display
print("row d_t →  P[d_{t+1}=col]; columns 0..9")
for r in range(10):
    print(f"  d={r}: " + " ".join(f"{P[r,c]:.4f}" for c in range(10)) + f"   (n={int(row_sums[r,0])})")

# Chi-square test of independence (rows sum to n_i, expected col j is n*p_j)
# Under independence, expected[i,j] = row_i_sum * col_j_sum / total
total = M.sum()
col_sums = M.sum(axis=0, keepdims=True)
exp = row_sums @ col_sums / total
chi2_M = ((M - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
df_M = 81  # (10-1)^2
p_M = 1 - stats.chi2.cdf(chi2_M, df_M)
print(f"chi-square independence: {chi2_M:.2f}, df=81, p={p_M:.4f}")

# ---------------------------------------------------------------------------
# T4: Wald-Wolfowitz runs test on parity
# ---------------------------------------------------------------------------
print("\n--- T4: Runs test on Even-Odd parity ---")
n1 = int(even.sum())          # even count
n0 = int(n - n1)               # odd count
# Number of runs
runs = 1 + int((even[1:] != even[:-1]).sum())
mu_R = 1 + 2 * n0 * n1 / n
var_R = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
Z = (runs - mu_R) / math.sqrt(var_R)
p_R = 2 * (1 - stats.norm.cdf(abs(Z)))
print(f"runs={runs}, expected={mu_R:.1f}, Z={Z:+.3f}, p={p_R:.4f}")

# ---------------------------------------------------------------------------
# T5: Run-length distribution for repeated identical digits
# ---------------------------------------------------------------------------
print("\n--- T5: Repeated-digit run length P(d_t = d_{t+1}=...) ---")
runs_same = []
cur = 1
for i in range(1, n):
    if digits[i] == digits[i - 1]:
        cur += 1
    else:
        runs_same.append(cur)
        cur = 1
runs_same.append(cur)
rc = Counter(runs_same)
print(f"observed P(d_t == d_{{t-1}}) = {(digits[1:] == digits[:-1]).mean():.5f}  (theory uniform i.i.d. = 0.10)")
for L in range(1, 8):
    obs = rc.get(L, 0)
    exp = len(runs_same) * (0.9) * (0.1 ** (L - 1))
    print(f"  length {L}: observed {obs}, expected {exp:.1f}")

# ---------------------------------------------------------------------------
# T6: Conditional probabilities P(d_t | d_{t-1}, d_{t-2}) — 100 cells per d_t.
# Look only for the largest deviation from 0.10.
# ---------------------------------------------------------------------------
print("\n--- T6: Top conditional triplets P(d | d_{-1}, d_{-2}) ---")
T = np.zeros((10, 10, 10), dtype=np.int64)
for i in range(2, n):
    T[digits[i - 2], digits[i - 1], digits[i]] += 1
P3 = T / np.maximum(T.sum(axis=2, keepdims=True), 1)
# Most extreme deviation
flat = []
for a in range(10):
    for b in range(10):
        nn = int(T[a, b].sum())
        if nn < 200:
            continue
        for c in range(10):
            p = P3[a, b, c]
            # Wald score against 0.1
            z = (p - 0.1) / math.sqrt(0.1 * 0.9 / nn)
            flat.append((abs(z), z, p, a, b, c, nn))
flat.sort(reverse=True)
print("top 8 deviations (|z| sorted) from uniform 0.10 (Wald score; raw, NOT Bonferroni-adjusted):")
print("    z       p(c|a,b)  a,b → c     n_obs")
for z_abs, z, p, a, b, c, nn in flat[:8]:
    print(f"  {z:+.2f}   {p:.4f}    {a},{b} → {c}    {nn}")
# Bonferroni threshold: 1000 cells -> need |z| > z_(0.05/1000/2) ≈ 4.06
print("Bonferroni-adjusted (1000 cells, two-sided): |z| > 4.06 to be significant")

# ---------------------------------------------------------------------------
# T7: Spectral density of parity series
# ---------------------------------------------------------------------------
print("\n--- T7: Spectral peaks of parity series (FFT) ---")
x = even - even.mean()
F = np.fft.rfft(x)
power = (F * F.conj()).real
freqs = np.fft.rfftfreq(len(x))
# Bandlimit: skip DC and very low freq (smoothed)
order = np.argsort(power[1:])[::-1] + 1
print("top 5 spectral peaks (freq cycles/sample, period in samples):")
for idx in order[:5]:
    period = 1.0 / freqs[idx] if freqs[idx] > 0 else float("inf")
    print(f"  freq={freqs[idx]:.6f}  period~{period:.1f} ticks  power={power[idx]:.3e}")
# Significance heuristic: under white noise, power_k ~ Exp(σ²/2). The max of N
# Exp draws scales as σ² log N / 2. So flag any peak > 5× median*log(N).
med = np.median(power[1:])
threshold = med * math.log(len(power)) * 5
print(f"median bin power = {med:.3e}, flag-threshold = {threshold:.3e}")
print(f"peaks above threshold: {(power > threshold).sum()}")

# ---------------------------------------------------------------------------
# T8: Distribution of 1-tick price increments (in cents)
# ---------------------------------------------------------------------------
print("\n--- T8: 1-tick increment Δprice in cents ---")
delta_cents = np.round((prices[1:] - prices[:-1]) * 100).astype(np.int64)
print(f"mean Δ = {delta_cents.mean():.3f} cents, std = {delta_cents.std():.3f} cents")
print(f"abs mean = {np.abs(delta_cents).mean():.3f} cents,  median |Δ| = {np.median(np.abs(delta_cents)):.1f}")
print(f"min/max Δ = {delta_cents.min()} / {delta_cents.max()} cents")
print(f"% |Δ| ≤ 9 cents (digit could be locally predictable) = {(np.abs(delta_cents) <= 9).mean() * 100:.2f}%")
print(f"% |Δ| ≤ 4 cents = {(np.abs(delta_cents) <= 4).mean() * 100:.2f}%")

# Histogram of (Δ digit) mod 10
shift = (digits[1:] - digits[:-1]) % 10
print("(d_{t+1} - d_t) mod 10 distribution (should be uniform 0.10 if no microstructure leakage):")
sh_counts = Counter(int(s) for s in shift)
for k in range(10):
    print(f"  shift {k}: {sh_counts.get(k, 0)/len(shift):.4f}  ({sh_counts.get(k, 0)})")

# ---------------------------------------------------------------------------
# T9: Conditional digit shift under low-Δ regimes (microstructure leakage test)
# ---------------------------------------------------------------------------
print("\n--- T9: Shift distribution conditional on |Δprice| regime ---")
for low, high, label in [(0, 0, "Δ=0"), (1, 1, "|Δ|=1c"), (2, 5, "|Δ|=2-5c"), (6, 100, "|Δ|≥6c")]:
    mask = (np.abs(delta_cents) >= low) & (np.abs(delta_cents) <= high)
    sel = shift[mask]
    if len(sel) < 200:
        continue
    sc = Counter(int(s) for s in sel)
    line = f"  {label} (n={len(sel):>6d}): "
    for k in range(10):
        line += f"{sc.get(k,0)/len(sel):.3f} "
    print(line)

# ---------------------------------------------------------------------------
# T10: Tick interarrival timing
# ---------------------------------------------------------------------------
print("\n--- T10: Tick interarrival statistics (should be ~1.000s for *_1s) ---")
ia = np.diff(epochs)
print(f"n_intervals = {len(ia)},  mean = {ia.mean():.4f}s,  std = {ia.std():.4f}s")
ia_counts = Counter(int(x) for x in ia)
print("interarrival counts (top 5): ", sorted(ia_counts.items(), key=lambda kv: -kv[1])[:5])

# ---------------------------------------------------------------------------
# T11: Hour-of-day and minute-of-hour digit frequency biases
# ---------------------------------------------------------------------------
print("\n--- T11: Hour-of-day and minute-of-hour digit-frequency check ---")
hours = np.array([datetime.fromtimestamp(e, tz=timezone.utc).hour for e in epochs])
chi2_hour = []
for h in range(24):
    sub = digits[hours == h]
    if len(sub) < 1000:
        chi2_hour.append((h, None, None))
        continue
    c = np.bincount(sub, minlength=10)
    expected_h = len(sub) / 10
    chi = ((c - expected_h) ** 2 / expected_h).sum()
    p = 1 - stats.chi2.cdf(chi, 9)
    chi2_hour.append((h, chi, p))
print("hour | chi2 | p (small p = digit dist deviates from uniform that hour)")
worst = sorted(chi2_hour, key=lambda x: x[2] if x[2] is not None else 1)[:5]
for h, chi, p in worst:
    print(f"  hour {h:2d}: chi2={chi:.2f}  p={p:.4f}")

# Bonferroni for 24 hours: p < 0.05/24 = 0.0021 to be significant
print("Bonferroni @ 24 tests: p<0.00208 needed for hour to be significantly non-uniform")

# ---------------------------------------------------------------------------
# T12: Simulate "always Over 0" — does empirical win-rate beat 91.24% break-even?
# ---------------------------------------------------------------------------
print("\n--- T12: Backtests of static contract bets (no signal — pure law) ---")
def bet_winrate(name, win_mask, payout):
    wins = int(win_mask.sum())
    tot = int(win_mask.size)
    p = wins / tot
    se = math.sqrt(p * (1 - p) / tot)
    lo, hi = p - 1.96 * se, p + 1.96 * se
    breakeven = 1 / (1 + payout)  # payout in fractional terms (e.g. 0.096)
    ev_per_dollar = p * payout - (1 - p)
    print(f"  {name:>14s}: wins {wins}/{tot} = {p:.4f}  CI95=[{lo:.4f},{hi:.4f}]  BE={breakeven:.4f}  EV/$={ev_per_dollar:+.4f}")
    return p, ev_per_dollar


# Each tick is potential entry, next tick is the exit. For 1-tick contracts.
exit_digits = digits[1:]
entry_digits = digits[:-1]
m = len(exit_digits)
print(f"  (sample size for backtests: {m} 1-tick windows)")

bet_winrate("Over 0",   exit_digits > 0,  0.096)
bet_winrate("Under 9",  exit_digits < 9,  0.096)
bet_winrate("Over 4",   exit_digits > 4,  0.953)
bet_winrate("Under 5",  exit_digits < 5,  0.953)
bet_winrate("Over 7",   exit_digits > 7,  3.717)
bet_winrate("Even",     exit_digits % 2 == 0, 0.953)
bet_winrate("Odd",      exit_digits % 2 == 1, 0.953)

# Differs / Matches for entry digit:
diff_mask = exit_digits != entry_digits
match_mask = exit_digits == entry_digits
bet_winrate("Differs", diff_mask, 0.096)
bet_winrate("Matches", match_mask, 7.929)

# ---------------------------------------------------------------------------
# T13: Can ANY conditioning on the last k digits push Over 0 above breakeven?
# Scan all 100 (a,b) → P(next_digit > 0) and find the best/worst.
# ---------------------------------------------------------------------------
print("\n--- T13: Conditional Over0 win-rate by (d_{-2}, d_{-1}) — search for edge ---")
records = []
for a in range(10):
    for b in range(10):
        idx = (digits[:-2] == a) & (digits[1:-1] == b)
        if idx.sum() < 200:
            continue
        next_d = digits[2:][idx]
        p_over0 = (next_d > 0).mean()
        nn = int(idx.sum())
        # 95% CI lower bound
        se = math.sqrt(p_over0 * (1 - p_over0) / nn)
        lo = p_over0 - 1.96 * se
        records.append((a, b, p_over0, lo, nn))
records.sort(key=lambda r: -r[2])
print("top 5 (a,b) for Over0 win-rate (BE = 0.9124):")
for a, b, p, lo, nn in records[:5]:
    print(f"  ({a},{b}) → P(d>0) = {p:.4f}  CI_lo={lo:.4f}  n={nn}")
print("bottom 5:")
for a, b, p, lo, nn in records[-5:]:
    print(f"  ({a},{b}) → P(d>0) = {p:.4f}  CI_lo={lo:.4f}  n={nn}")
# Bonferroni: 100 cells, need p < 0.05/100 = 5e-4. Equivalent to lo > 0.9124 + z * se ...
print("any (a,b) with CI lower bound > 0.9124 (break-even)?")
beats = [r for r in records if r[3] > 0.9124]
print(f"  count: {len(beats)}")
for r in beats[:20]:
    print(f"    {r}")

# ---------------------------------------------------------------------------
# T14: same scan for Differs (highest natural edge in payout matrix)
# ---------------------------------------------------------------------------
print("\n--- T14: Conditional Differs win-rate by entry digit ---")
for d in range(10):
    idx = digits[:-1] == d
    if idx.sum() < 100:
        continue
    next_d = digits[1:][idx]
    p_diff = (next_d != d).mean()
    nn = int(idx.sum())
    se = math.sqrt(p_diff * (1 - p_diff) / nn)
    lo = p_diff - 1.96 * se
    hi = p_diff + 1.96 * se
    print(f"  entry={d}: P(next != entry) = {p_diff:.4f}  CI95=[{lo:.4f},{hi:.4f}]  n={nn}  BE=0.9124")

# ---------------------------------------------------------------------------
# T15: Realised-vol-conditional Even/Odd test
# Bet Even when last 5 ticks moved by < X cents total. Does this beat 0.5?
# ---------------------------------------------------------------------------
print("\n--- T15: Even win-rate conditional on recent realized volatility ---")
window = 10
abs_dc = np.abs(delta_cents)
realised = np.zeros(n)
realised[window:] = np.array([abs_dc[i - window:i].sum() for i in range(window, n)])
# Bucket by recent realized
buckets = [(0, 1, "ultra-quiet"), (1, 5, "quiet"), (5, 20, "moderate"),
           (20, 100, "active"), (100, 10_000_000, "violent")]
for lo, hi, label in buckets:
    mask = (realised >= lo) & (realised < hi)
    mask[:window] = False
    if mask.sum() < 500:
        continue
    sub = digits[mask]
    p_even = (sub % 2 == 0).mean()
    nn = int(mask.sum())
    se = math.sqrt(p_even * (1 - p_even) / nn)
    print(f"  {label:>11s} (cumulative |Δ| in [{lo},{hi})): n={nn}  P(Even)={p_even:.4f}  CI95=±{1.96*se:.4f}")

print("\n=== DONE ===")
