"""
Multi-lag robustness check.

Test marginal & conditional win-rates at decision-to-exit lags 1, 2, 3, 5, 10
(corresponds to "wrong" 1-tick paper-trade, true 1-tick contract, true 2-tick
contract, true 4-tick contract, true 9-tick contract).

Question: at any reasonable contract duration, does any conditioning beat
break-even?
"""
from __future__ import annotations
import gzip, json, math, sys
from pathlib import Path
import numpy as np
from scipy import stats

P = Path(__file__).resolve().parent / "ticks_v100_1s.json.gz"
ticks = json.loads(gzip.decompress(P.read_bytes()))
prices = np.array([float(t[1]) for t in ticks], dtype=np.float64)
digits = np.array([int(f"{p:.2f}"[-1]) for p in prices], dtype=np.int64)
n = len(digits)

def wlow(p, n, z):
    if n == 0: return 0.0
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (c-h)/d

BE_OVER0 = 1/1.096      # 0.9124
BE_DIFF  = 1/1.096
BE_OVER4 = 1/1.953
BE_OVER7 = 1/4.717
BE_MATCH = 1/8.929

print(f"{'lag':>4s} | {'Over0 p':>9s} {'Differs p':>11s} {'Over4 p':>9s} {'Over7 p':>9s} {'Matches p':>11s}")
print("-" * 70)
for lag in (1, 2, 3, 5, 10):
    de = digits[lag:]
    dd = digits[:-lag] if lag > 0 else digits
    p_over0 = (de > 0).mean()
    p_diff  = (de != dd).mean()
    p_over4 = (de > 4).mean()
    p_over7 = (de > 7).mean()
    p_match = (de == dd).mean()
    print(f"{lag:>4d} | {p_over0:.5f}   {p_diff:.5f}      {p_over4:.5f}   {p_over7:.5f}   {p_match:.5f}")
print(f"  BE | {BE_OVER0:.5f}   {BE_DIFF:.5f}      {BE_OVER4:.5f}   {BE_OVER7:.5f}   {BE_MATCH:.5f}")
print()

# Chi-square independence of (d_t, d_{t+lag}) for each lag
print("Chi-square independence at each lag (df=81):")
for lag in (1, 2, 3, 5, 10, 20, 50):
    M = np.zeros((10, 10), dtype=np.int64)
    for i in range(n - lag):
        M[digits[i], digits[i + lag]] += 1
    row = M.sum(axis=1, keepdims=True)
    col = M.sum(axis=0, keepdims=True)
    tot = M.sum()
    exp = row @ col / tot
    chi = float(((M - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
    pval = 1 - stats.chi2.cdf(chi, 81)
    print(f"   lag={lag:3d}: chi2={chi:7.2f}  p={pval:.4f}   "
          f"{'no dependence' if pval > 0.05 else 'DEPENDENT'}")

# Best conditioning cell on (d_{-1}, d_{0}) → d_exit, across all lags
print("\nBest (a,b)-conditioning cell win-rate per lag (looking for ANY edge):")
print(f"{'lag':>4s} | {'best cell':>11s} {'best p':>9s} {'CI99.9 lo':>11s} {'exceeds BE?':>11s}")
for lag in (1, 2, 3, 5, 10):
    best = None
    for a in range(10):
        for b in range(10):
            t_idx = np.arange(1, n - lag)
            mask = (digits[t_idx - 1] == a) & (digits[t_idx] == b)
            nn = int(mask.sum())
            if nn < 200:
                continue
            d_exit = digits[t_idx + lag][mask]
            p = (d_exit > 0).mean()
            lo = wlow(p, nn, 3.291)
            if best is None or p > best[2]:
                best = ((a, b), nn, p, lo)
    (a, b), nn, p, lo = best
    print(f"{lag:>4d} | {a},{b}        {p:.4f}    {lo:.4f}      {'YES' if lo > BE_OVER0 else 'no':>4s}")

# Are there any specific patterns at lag=2 that would not appear at lag=1?
# Build the full 10x10 lag-2 transition matrix and show row-max above 0.10:
print("\n10x10 transition matrix P(d_{t+2} | d_t), showing deviation from uniform 0.10:")
M2 = np.zeros((10, 10), dtype=np.int64)
for i in range(n - 2):
    M2[digits[i], digits[i + 2]] += 1
row_sums = M2.sum(axis=1, keepdims=True)
P2 = M2 / np.maximum(row_sums, 1)
abs_dev_max = np.abs(P2 - 0.10).max()
ix = np.unravel_index(np.argmax(np.abs(P2 - 0.10)), P2.shape)
print(f"Max |deviation| from 0.10 anywhere in 10x10 table: {abs_dev_max:.4f}  at cell ({ix[0]},{ix[1]})")
print(f"That cell: P(d_{{t+2}}={ix[1]} | d_t={ix[0]}) = {P2[ix]:.4f}  "
      f"(n_t = {int(row_sums[ix[0],0])})")
# z-score
nrow = int(row_sums[ix[0], 0])
z = (P2[ix] - 0.10) / math.sqrt(0.10 * 0.90 / nrow)
print(f"Raw Wald z = {z:+.2f}.  Bonferroni for 100 cells (two-sided p<0.05/100): need |z|>3.29")
print(f"=> {'SIGNIFICANT' if abs(z) > 3.29 else 'not significant'} after multiple-test correction.")

# Most-extreme single cell across THREE conditioning depths (1, 2, 3 prior digits) at LAG 2:
print("\nMost-extreme cells in P(d_{t+2} | last-k digits) for k=1,2,3:")
for k in (1, 2, 3):
    if k == 1:
        idx = lambda t, vals: digits[t] == vals[0]
    elif k == 2:
        idx = lambda t, vals: (digits[t - 1] == vals[0]) & (digits[t] == vals[1])
    elif k == 3:
        idx = lambda t, vals: (digits[t - 2] == vals[0]) & (digits[t - 1] == vals[1]) & (digits[t] == vals[2])
    extremes = []
    t_arr = np.arange(k - 1, n - 2) if k > 1 else np.arange(0, n - 2)
    for vals in np.ndindex(*([10] * k)):
        mask = idx(t_arr, vals)
        nn = int(mask.sum())
        if nn < 500:
            continue
        for digit_to in range(10):
            wins = int((digits[t_arr + 2][mask] == digit_to).sum())
            p = wins / nn
            z = (p - 0.10) / math.sqrt(0.10 * 0.90 / nn)
            extremes.append((abs(z), z, p, vals, digit_to, nn))
    extremes.sort(reverse=True)
    print(f"  k={k}: top 3 extremes (Wald z vs 0.10):")
    for absz, z, p, vals, dt, nn in extremes[:3]:
        print(f"     P(d_{{t+2}}={dt} | last {k} = {vals}) = {p:.4f}   z={z:+.2f}   n={nn}")
    # Bonferroni
    total_cells = (10 ** k) * 10
    crit = stats.norm.ppf(1 - 0.05 / (2 * total_cells))
    print(f"     Bonferroni for {total_cells} cells: |z|>{crit:.2f} needed for sig.")

print("\n=== DONE ===")
