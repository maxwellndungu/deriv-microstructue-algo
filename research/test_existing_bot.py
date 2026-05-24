"""
Direct test of the existing bot's 60-trade dataset.

Goal: ask whether the bot's reported 56-3-1 record represents a real edge
beyond what we'd expect from random sampling at the *nominal* contract
probability — and whether 'convergence_strength' contains real information.
"""
import json
import math
from collections import Counter
from scipy import stats

with open("/home/deriv-microstructue-algo/trades.json") as f:
    raw = json.load(f)
trades = [t for t in raw["trades"] if t["outcome"] in ("win", "loss")]
print(f"N settled trades = {len(trades)}")

# Per-contract: empirical win rate vs nominal vs break-even
def by(key):
    g = {}
    for t in trades:
        k = key(t)
        g.setdefault(k, []).append(t)
    return g

nominal = {
    ("DIGITOVER", "0"): 0.90,
    ("DIGITUNDER", "9"): 0.90,
    ("DIGITDIFF", None): 0.90,
    ("DIGITMATCH", "4"): 0.10,
}
breakeven = {
    ("DIGITOVER", "0"): 0.9124,
    ("DIGITUNDER", "9"): 0.9124,
    ("DIGITDIFF", None): 0.9124,
    ("DIGITMATCH", "4"): 0.1120,
}

groups = by(lambda t: (t["contract_type"], t["barrier"]))
print("\n=== Per-contract test against nominal Bernoulli(p=0.90 or 0.10) ===")
for k, ts in groups.items():
    wins = sum(1 for t in ts if t["outcome"] == "win")
    n = len(ts)
    p_nom = nominal.get(k, 0.9)
    be = breakeven.get(k, 0.9124)
    # exact binomial vs nominal
    if wins / n > p_nom:
        pval = stats.binom.sf(wins - 1, n, p_nom)
    else:
        pval = stats.binom.cdf(wins, n, p_nom)
    # vs break-even (one-sided right)
    pval_be = stats.binom.sf(wins - 1, n, be)
    print(f"  {k}: wins {wins}/{n} = {wins/n:.4f}  "
          f"nominal {p_nom:.4f}  P(≥observed | nom)={pval:.4f}  "
          f"BE {be:.4f}  P(≥observed | BE)={pval_be:.4f}")

# Convergence-strength: does higher signal predict wins?
print("\n=== 'convergence_strength' predictive power ===")
cs_win = [t["convergence_strength"] for t in trades if t["outcome"] == "win"]
cs_loss = [t["convergence_strength"] for t in trades if t["outcome"] == "loss"]
print(f"  winners: n={len(cs_win)}  mean={sum(cs_win)/len(cs_win):.4f}  "
      f"std={(sum((x-sum(cs_win)/len(cs_win))**2 for x in cs_win)/len(cs_win))**0.5:.4f}")
print(f"  losers:  n={len(cs_loss)}  mean={sum(cs_loss)/len(cs_loss):.4f}  "
      f"std={(sum((x-sum(cs_loss)/len(cs_loss))**2 for x in cs_loss)/len(cs_loss))**0.5:.4f}")
if len(cs_win) >= 2 and len(cs_loss) >= 2:
    u, p = stats.mannwhitneyu(cs_win, cs_loss, alternative="greater")
    print(f"  Mann-Whitney U (winners > losers): U={u}, p={p:.4f}")

# What is the maximum N we'd need at 90% baseline to detect a 1% edge with 95% power?
from math import sqrt
def n_for_detect(p0, delta, alpha=0.05, power=0.80):
    p1 = p0 + delta
    z_a = 1.645  # one-sided 0.05
    z_b = 0.842  # 0.80 power
    num = (z_a * sqrt(p0*(1-p0)) + z_b * sqrt(p1*(1-p1)))**2
    den = delta * delta
    return num / den

print("\n=== Sample-size needed to *detect* an edge if it exists ===")
print("  H0: p = 0.9124 (break-even Over0/Diff). H1: p = 0.9124 + delta. α=0.05 (one-sided), power=0.80.")
for d in (0.005, 0.01, 0.02, 0.05, 0.1):
    n_needed = n_for_detect(0.9124, d)
    print(f"  detect +{d*100:>4.1f}% edge → need n = {n_needed:.0f} settled trades")
print("\n  60 trades total is too small to detect anything below +5% above break-even.")
