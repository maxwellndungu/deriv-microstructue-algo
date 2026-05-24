# Edge Analysis — Deriv Volatility 100 (1s) Index

> A first-principles audit of every claim that could underlie a trading edge,
> done with 24h × 86,383 contiguous live ticks pulled directly from Deriv's
> public `ticks_history` endpoint, plus a full statistical sweep of the
> existing bot's trade record. Then a strategy that survives that audit.

---

## TL;DR for the impatient

1. **There is no detectable predictive edge in V100(1s) digit sequences.**
   24h of live ticks pass every test of i.i.d. uniformity I can throw at
   them (chi-square, autocorrelation lags 1–1000, Ljung–Box, Markov,
   triplet-conditional, runs, spectral, hour-of-day, magnet, drift, plus a
   1,320-cell brute-force grid).
2. **The existing bot's 56/59 win rate is not evidence of edge.** Under the
   null hypothesis that the true win rate equals exact break-even (0.9124),
   the probability of seeing ≥56 wins in 59 trades is **0.229** — i.e. it
   happens about 1 in 4 times by chance. The "convergence_strength" feature
   has **no predictive power** (Mann–Whitney U p = 0.45).
3. **The expected value of every contract is structurally negative**,
   ranging from −1.30 % (Differs / Over 0 / Under 9) to −11.22 % (Matches).
   No conditioning we tested moves any cell's lower Wilson bound above
   break-even.
4. **The only mathematically honest strategy is:** default to *do not trade*,
   continuously run a sequential probability ratio test against the live
   feed, restrict any live trades to the lowest-edge contracts (Differs /
   Over 0 / Under 9), use fractional Kelly (which **prescribes zero stake**
   when EV ≤ 0), and have a hard daily-loss kill-switch. This is the bot
   I built.

The rest of this document walks the reasoning so you can argue with it.

---

## 1. The payout matrix is the only thing we can trust

| Contract  | Payout | Win prob (uniform) | EV per $1 | House edge |
|-----------|--------|-------------------:|----------:|-----------:|
| Over 0    | 109.6 % | 0.90 | −$0.0136 | 1.36 % |
| Under 9   | 109.6 % | 0.90 | −$0.0136 | 1.36 % |
| Differs   | 109.6 % | 0.90 | −$0.0136 | 1.36 % |
| Over 1    | 123.2 % | 0.80 | −$0.0144 | 1.44 % |
| Under 8   | 123.2 % | 0.80 | −$0.0144 | 1.44 % |
| Over 4    | 195.3 % | 0.50 | −$0.0235 | 2.35 % |
| Under 5   | 195.3 % | 0.50 | −$0.0235 | 2.35 % |
| Even / Odd| 195.3 % | 0.50 | −$0.0235 | 2.35 % |
| Over 7    | 471.7 % | 0.20 | −$0.0566 | 5.66 % |
| Under 2   | 471.7 % | 0.20 | −$0.0566 | 5.66 % |
| Over 8    | 892.9 % | 0.10 | −$0.1071 | 10.71 % |
| Matches   | 892.9 % | 0.10 | −$0.1071 | 10.71 % |

**Key implication.** If you have *any* small predictive edge ε in win rate,
the cheapest contracts to exploit it on are **Differs / Over 0 / Under 9**:
they have only 1.36 % house edge, so you need to lift the win rate from
0.90 to just above 0.9124 to be profitable. Every other contract demands a
larger predictive edge.

This single observation kills the existing bot's design: it spreads bets
across MATCH (10.71 % edge) and DIFFER (1.36 %) and Over/Under at many
barriers. Even *if* a tiny predictive signal existed, the bot would still
lose on Match and Over 4 because the required edge is much larger there.

---

## 2. Can the digit sequence itself be predicted? — 15 tests, all null.

### T1. Empirical digit frequencies
Counts on 86,383 V100(1s) ticks: `[8725 8538 8524 8762 8659 8517 8805 8675 8573 8605]`
Expected 8638.3, max deviation 1.93 %. χ² = 11.07, df = 9, **p = 0.27**.
Cannot reject uniform.

### T2. Autocorrelation
Tested lags 1–1000 on three series: raw digit, Even-indicator, Over-half-indicator.
The 95 % white-noise band is ±0.0067. Only lag 1000 narrowly crosses the band
on the parity series — exactly the kind of "near-significant" lag you expect to
see at random when scanning 17 lags.

Ljung–Box at L = 10/20/50: **p = 0.62, 0.90, 0.53**. No autocorrelation.

### T3. Markov independence
The empirical 10×10 transition matrix's chi-square test against independence:
χ² = 81.12, df = 81, **p = 0.48**. The sequence is Markov-0
(memoryless) at any conditioning depth we can credibly estimate.

### T4. Wald–Wolfowitz runs test on parity
Observed 43,207 runs, expected 43,192.3, **Z = +0.10, p = 0.92**.
Parity is as random as a fair coin.

### T5. Repeat-digit run-length distribution
Observed P(d_t = d_{t−1}) = 0.09943 vs theory 0.10. Run-length histogram
matches geometric(0.9) within a fraction of one count. Indistinguishable
from i.i.d.

### T6. Triplet conditional probabilities P(d | d₋₁, d₋₂)
Best raw |z| was 3.30, but with 1,000 cells the Bonferroni threshold is
|z| > 4.06. **No triplet is statistically significant.**

### T7. Spectral peaks (FFT) of parity series
Peak power ≈ 2.95 × 10⁵, but a generous false-alarm threshold of
5×median×log(N) ≈ 8.0 × 10⁵. **No periodicity is statistically credible.**
Modern CSPRNGs have no detectable periodicity at any scale.

### T8. 1-tick price-increment distribution
Mean Δprice = +0.062 cents (consistent with zero drift). **σ = 16.6 cents**.
Range [−80, +74]. Median |Δ| = 11 cents.

Because σ_Δ (16.6 c) > 10 c (the modulus that defines the last digit), the
digit shift mod 10 is essentially **uniform** — the empirical
(d_{t+1} − d_t) mod 10 distribution sits within ±0.001 of 0.1000 in every
bin.

### T9. Conditional shift under low-Δ regimes
When |Δ| ≤ 5 cents, the digit shift is mechanically constrained because
|Δ| < modulus. This is the *only* structural feature we found. But the
condition that |Δ| stays small is determined by the *next* tick's
volatility, which we cannot pre-observe. It does not produce a tradable
predictive edge.

### T10. Tick interarrival
Mean = 1.0000 s, std = 0.0000 s. Every tick is exactly 1 second apart.
**No microstructure-timing arbitrage exists.**

### T11. Hour-of-day digit-frequency bias
The most "deviating" hour is hour 12 with χ² = 20.2, p = 0.017. Bonferroni
across 24 hours requires p < 0.00208. **No hour passes correction.**

### T12. Static-contract law of large numbers
At n = 86,382 1-tick windows:

```
Over 0   wins 77,657/86,382 = 0.8990   CI95 [0.8970, 0.9010]   BE 0.9124   EV/$ −0.0147
Under 9  wins 77,778/86,382 = 0.9004   CI95 [0.8984, 0.9024]   BE 0.9124   EV/$ −0.0132
Differs  wins 77,793/86,382 = 0.9006   CI95 [0.8986, 0.9026]   BE 0.9124   EV/$ −0.0130
Even     wins 43,286/86,382 = 0.5011   CI95 [0.4978, 0.5044]   BE 0.5120   EV/$ −0.0214
Matches  wins  8,589/86,382 = 0.0994   CI95 [0.0974, 0.1014]   BE 0.1120   EV/$ −0.1122
```

The empirical rate matches the nominal contract probability with extreme
tightness. The exact house edge is observable in the data.

### T13. Best conditional (d₋₂, d₋₁) cell for Over 0
Top: (7, 4) → P(d > 0) = 0.9236, n = 838. **Wilson 95 % lower bound = 0.9056** — below break-even 0.9124. No (a,b) cell out of 100 has a lower bound above break-even.

### T14. Conditional Differs by entry digit
All 10 entry-digit cells have 95 % CI lower bounds **below** 0.9124. None
beats break-even.

### T15. Realized-volatility-conditional Even
P(Even) = 0.50 across every realized-vol bucket. No clustering effect.

### Final brute-force sweep
1,320 cells (features × values × contracts) scanned. **Under H0 (no edge)
we expect ~33 cells with Sharpe > +2 and ~33 with Sharpe < −2.** Observed:

```
  Sharpe > +2 :   3 cells
  Sharpe < −2 : 252 cells
```

This asymmetry is the **house edge made visible**. The data are essentially
i.i.d. uniform; the *only* reliable signal is that across every contract
type, the empirical win rate sits 1–11 % below break-even, by an amount
equal to the explicit house edge.

---

## 3. Pre-empting the obvious objections

> *"But the 60-trade record shows 56 wins. Surely that means edge?"*

Under H0: true_p = break-even (0.9124), the probability of seeing ≥ 56 wins
in 59 settled trades is

```
P(W ≥ 56 | p = 0.9124, n = 59) = 0.229
```

That is, **about 1 in 4.4** by chance — well above the conventional 0.05
significance threshold. Even at the *nominal* p = 0.90, P(W ≥ 56) = 0.146.
The record is consistent with no edge whatsoever.

To statistically distinguish a +1 % real edge from break-even with 80 %
power and α = 0.05 you need **~4,766 trades**. Sixty isn't even close.

> *"convergence_strength is high before wins — surely that's a real
> signal?"*

Mean strength of winning trades: 0.8231. Mean strength of losing trades:
**0.8323** (losers slightly *higher*). Mann–Whitney U test that winners >
losers: U = 88, **p = 0.45**. The "signal" is not a signal.

> *"What about the tapped/untapped digit analysis (Engines 1/2/3)?"*

The tapped-vs-untapped construction is purely deterministic: for any
high–low range [L, H] with span S, the multi-set of digits in
`{floor(L*100), floor(L*100)+1, ..., floor(H*100)}` is *forced* by the
range, not by what the price actually did. It contains no information
beyond the range. Worse, "most frequent digit in the tapped set" is just
the digit closest to where the cluster started — it carries zero forward
predictive value.

> *"Could Deriv's RNG be predictable in some clever way?"*

Deriv state-publicly that their synthetic indices use CSPRNGs audited by
Gaming Laboratories International. That alone is not proof — companies
have shipped broken RNGs before. But: if there were *any* exploitable bias
at the scale we tested (24h, 86,383 ticks), we would have seen Bonferroni-
significant cells in the 1,320-cell sweep. We did not. Combined with the
exactly-1.000s metronomic tick timing and the perfect match of empirical
to nominal win rates, the prior on "Deriv has a bug we can find with
86k samples" is very low.

> *"What about microstructure features beyond the digit?"*

The only structural feature we found is the *mechanical* constraint on
the digit shift when |Δprice| < modulus — but you can only act on this
if you can predict the *next* tick's |Δprice|, and tick-magnitudes are
themselves uncorrelated under synthetic-index dynamics (volatility
clustering, the empirical regularity that lets GARCH work in real markets,
is *absent by design* in synthetic indices, and confirmed absent in T15).

> *"Couldn't we hide an edge in a more complex ML model?"*

A model can only extract structure that exists in the inputs. We tested
every reasonable conditioning the inputs can support: single, pair, triplet
of recent digits; volatility regime; hour; minute; price-mod-10 / mod-100;
last-tick-shift bucket; rolling-mode of last 10 digits; and pairwise
interactions of those. The strongest *uncorrected* signal was a +0.0144
win-rate gap that doesn't survive multi-test correction. A neural net or
gradient-boosted tree given the same inputs cannot beat the same statistical
tests — the information simply isn't there.

> *"You only tested 24 hours. What about longer time horizons?"*

86k ticks is **already enough** to detect a +0.5 % edge against break-even
with ~80 % power on any single contract type. The absence of edge in the
strongest naive cell of a 1,320-cell sweep is not a small-sample artifact.
Could there be edge at a *much* longer timescale (weeks/months)? Possibly,
but then the bot would need to operate at that timescale, and at the
required sample size to detect it (>50,000 trades), commissions and edge
decay typically destroy what shows up in backtests of synthetic data.

---

## 4. The strategy that survives

> *"If you can't beat the house, at least don't lose to it on autopilot."*

### Core thesis

There is no predictive edge to exploit. **But** the bot can still make
itself maximally useful by:

1. **Treating every potential trade as a hypothesis test.** Compare the
   conditional win-rate of an *observed* feature against break-even using a
   Sequential Probability Ratio Test (Wald). Bet only if SPRT accepts H1:
   p > break-even at α = 0.001.

2. **Restricting any live trade to the lowest-house-edge contracts.**
   That is **Differs**, equivalent to **Over 0** and **Under 9** at the
   same 1.36 % house edge. Skip every other contract.

3. **Default to OBSERVE mode.** The bot collects ticks, runs continuous
   statistical tests, and shows a live "Statistical Observatory" panel
   exposing p-values from all 15 tests. Under random data, no test ever
   trips and no trade is placed.

4. **Use fractional Kelly with hard floor at zero.** Stake size is
   `f* = (p̂ − BE) / payout` (Kelly's formula for a binary bet), clipped
   to [0, max_kelly] then scaled by a Kelly fraction (default 0.25).
   When `p̂ ≤ BE`, **Kelly says don't bet** — and we don't.

5. **Hard guardrails.**
   - Daily-loss kill switch at user-configurable threshold (default 5 %).
   - Per-trade exposure capped at 1 % of bankroll.
   - Stake floor at the Deriv minimum (default $0.35).
   - Cluster-volume gate: no more than N trades per minute, regardless of
     signal strength.
   - **All settings configurable from the UI without code changes.**

6. **Live operator transparency.** The UI shows:
   - The Wilson 99 % lower bound on win rate for each conditioning the bot
     considers.
   - The current SPRT log-likelihood ratio per strategy.
   - The current "EV gauge" — green only if estimated EV per dollar > 0.
   - The 15 statistical tests' rolling p-values.
   - Bankroll equity curve with running drawdown and risk-of-ruin estimate.

### What the bot deliberately does NOT do

- **No ML "confidence" scores that don't correspond to a calibrated
  probability** — those are exactly the signals that look strong in 60
  trades and disappear over 6,000.
- **No mean-reversion ("the digit hasn't shown up in a while") signals** —
  those rely on the gambler's fallacy, the wrongest belief in finance.
- **No betting on Matches / Over 4 / Over 7 / etc.** — even if the bot
  discovered a +2 % win-rate edge, on Over 4 it would still lose money
  because the required edge is ≥1.2 % and on Over 7 it would need ≥5 %.
- **No Martingale / anti-Martingale stake scaling.** They guarantee ruin
  given enough trades.

### What this bot will probably do in practice

Default to OBSERVE. Stay there. The operator will see all tests
continuously failing to reject the null hypothesis, which is exactly the
correct reading of the situation. **The right amount to trade against a
provably negative-EV venue is zero.**

If at some future date Deriv ships an RNG bug, the bot is the first
detector that catches it — and once it does, the SPRT triggers PROBE,
then EXPLOIT, with stake sizing scaled to the magnitude of the detected
edge.

That is the most aggressive and the most honest bot you can build for this
venue. Everything else is overfitting to noise.

---

## 5. The reproducible evidence

All raw artefacts of this analysis live alongside this document in
`research/`:

- `collect_ticks.py` — pulls a contiguous tick window from Deriv.
- `ticks_v100_1s.json` — 86,383 ticks, 24 h on Volatility 100 (1s).
- `ticks_v10_1s.json`  — 86,383 ticks, 24 h on Volatility 10 (1s).
- `audit.py` — 15 statistical tests; outputs `audit_v100.txt`,
  `audit_v10.txt`.
- `edge_search.py` — multi-feature × multi-contract edge sweep with
  Bonferroni-aware decisions; outputs `edge_v100.txt`, `edge_v10.txt`.
- `final_check.py` — round-number magnet, drift, Sharpe leaderboard,
  small-sample power; output `final_v100.txt`.
- `test_existing_bot.py` — head-to-head statistical test of the 60-trade
  dataset.

Re-run anything to regenerate the numbers above.
