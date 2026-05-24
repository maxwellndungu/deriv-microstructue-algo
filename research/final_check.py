"""
Final sweep before designing the bot:
  (A) Magnet check: is price unusually attracted to round numbers (e.g., X.X0)?
  (B) Drift check: zero-mean assumption for log-returns?
  (C) Final brute-force: scan ALL pairs (feature × value × contract) and report
      the empirical *Sharpe* of each cell (= edge / stderr). Top of the list
      tells us the strongest naive candidate before correction.
  (D) "Reservoir" test: bet the SAME strategy with random restarts to gauge
      how often a 60-trade window randomly produces 56+ wins on Over0/Under9.
"""
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats

P = Path("ticks_v100_1s.json")
if not P.exists():
    P = Path("ticks_v100_1s.json.gz")
if P.suffix == ".gz":
    import gzip
    ticks = json.loads(gzip.decompress(P.read_bytes()))
else:
    ticks = json.loads(P.read_text())
prices = np.array([float(t[1]) for t in ticks])
digits = np.array([int(f"{p:.2f}"[-1]) for p in prices])
exit_d = digits[1:]
entry_d = digits[:-1]
m = len(exit_d)

# (A) Magnet test: histogram of *integer* mod 10 and integer mod 100
print("=== (A) ROUND-NUMBER MAGNET CHECK ===")
last2 = np.round((prices - np.floor(prices)) * 100).astype(int)  # cents of price
hist = np.bincount(last2, minlength=100)
expected = len(prices) / 100
chi2 = ((hist - expected) ** 2 / expected).sum()
p = 1 - stats.chi2.cdf(chi2, 99)
print(f"  chi-square last-2-cents uniform: {chi2:.2f}  p={p:.4f}")
# Look at the 5 most over/under-represented
deviation = (hist - expected) / np.sqrt(expected)
order = np.argsort(deviation)
print("  most under-represented cent values:", order[:5].tolist())
print("  most over-represented cent values:", order[-5:].tolist())

# Special cells: cents=0, 50 (round/half-round)
print(f"  freq cents==0: {hist[0]/len(prices):.4f}  (expected 0.0100)")
print(f"  freq cents==50: {hist[50]/len(prices):.4f}  (expected 0.0100)")
print(f"  freq cents ending in 0: {hist[::10].sum()/len(prices):.4f}  (expected 0.1000)")

# (B) Drift check
print("\n=== (B) DRIFT CHECK ===")
log_ret = np.diff(np.log(prices))
mu = log_ret.mean()
se_mu = log_ret.std() / np.sqrt(len(log_ret))
print(f"  mean log-return per tick: {mu:.6e}  stderr={se_mu:.6e}  Z={mu/se_mu:+.2f}")
print(f"  per-day drift (86400 ticks): {mu*86400:.6f}")

# (C) Brute-force "Sharpe" leaderboard
print("\n=== (C) STRONGEST NAIVE CELLS (sorted by empirical Sharpe per cell) ===")
features = {
    "d_{-1}": entry_d,
    "d_{-2}d_{-1}": (np.roll(entry_d, 1) * 10 + entry_d)[1:],
    "d_{-3}d_{-2}d_{-1}": (np.roll(entry_d, 2) * 100 + np.roll(entry_d, 1) * 10 + entry_d)[2:],
}

PAYOUT = {"Over0": 0.096, "Under9": 0.096, "Differs": 0.096,
          "Over1": 0.232, "Under8": 0.232,
          "Over4": 0.953, "Under5": 0.953, "Even": 0.953, "Odd": 0.953,
          "Over7": 3.717, "Under2": 3.717, "Matches": 7.929}
BE = {k: 1/(1+v) for k, v in PAYOUT.items()}


def win(c, idx):
    e = exit_d[idx]
    en = entry_d[idx]
    if c == "Over0":  return e > 0
    if c == "Under9": return e < 9
    if c == "Over1":  return e > 1
    if c == "Under8": return e < 8
    if c == "Over4":  return e > 4
    if c == "Under5": return e < 5
    if c == "Over7":  return e > 7
    if c == "Under2": return e < 2
    if c == "Even":   return e % 2 == 0
    if c == "Odd":    return e % 2 == 1
    if c == "Differs":return e != en
    if c == "Matches":return e == en


rows = []
for fname, fvec in features.items():
    offset = m - len(fvec)
    for v in np.unique(fvec):
        idx_in_f = np.where(fvec == v)[0]
        idx = idx_in_f + offset
        idx = idx[idx < m]
        nn = len(idx)
        if nn < 200:
            continue
        for c in PAYOUT:
            w = win(c, idx)
            p_hat = w.mean()
            edge = p_hat - BE[c]
            se = math.sqrt(p_hat * (1 - p_hat) / nn) if nn > 0 else 0
            if se == 0:
                continue
            sharpe = edge / se
            rows.append((sharpe, fname, int(v), c, p_hat, BE[c], nn))

rows.sort(reverse=True)
print("Top 10 by Sharpe (raw — DO NOT believe naive ranking — read note below):")
for s, fn, v, c, p, be, nn in rows[:10]:
    print(f"  sharpe={s:+.2f}  feat={fn}=v={v:<5} ctr={c:<8} p̂={p:.4f}  BE={be:.4f}  n={nn}")
print("Bottom 10:")
for s, fn, v, c, p, be, nn in rows[-10:]:
    print(f"  sharpe={s:+.2f}  feat={fn}=v={v:<5} ctr={c:<8} p̂={p:.4f}  BE={be:.4f}  n={nn}")

# How many cells did we test?
print(f"\n  Total cells tested: {len(rows)}")
print(f"  Expected number of cells with Sharpe > 2 (one-sided p<0.025) under H0: {len(rows)*0.025:.1f}")
positive_sig = sum(1 for r in rows if r[0] > 2)
negative_sig = sum(1 for r in rows if r[0] < -2)
print(f"  Observed cells with Sharpe > +2: {positive_sig}")
print(f"  Observed cells with Sharpe < -2: {negative_sig}")
print(f"  (under H0=no edge, expect ~{len(rows)*0.025:.0f} on each side)")

# (D) Probability of observing 56-3-1 in 60 trades at break-even probability
print("\n=== (D) IS THE 56/59 WIN RATE FROM 60 TRADES EVIDENCE OF EDGE? ===")
print("  Assume true win rate equals break-even 0.9124 (no edge).")
print("  Probability of 56 or more wins out of 59 trades:")
p_no_edge = 1 - stats.binom.cdf(55, 59, 0.9124)
print(f"  P(W >= 56) = {p_no_edge:.4f}  (i.e. ~1 in {1/p_no_edge:.1f})")
print(f"  At nominal Bernoulli p=0.90: P(W >= 56) = {1 - stats.binom.cdf(55, 59, 0.90):.4f}")
print("  → 56/59 is NOT statistically significant evidence of edge above break-even.")
