# NHB — Null-Hypothesis Bot for Deriv Synthetic Indices

A statistically-honest trading bot for **Volatility 100 (1s) Index** on Deriv.

> *"You can't beat the house — but you can prove that with statistics, run
>  exactly enough trades to find out, and then stop trading."*

## What this is

A first-principles rebuild of the original `1HZ10V` digit-microstructure bot.
The strategy was redesigned after a comprehensive statistical audit
(see [`research/EDGE_ANALYSIS.md`](research/EDGE_ANALYSIS.md)) that
established three facts on 24 hours of live tick data:

1. The V100(1s) digit stream is **indistinguishable from i.i.d. uniform**
   (chi-square, autocorrelation, Markov, runs, spectral, Bonferroni-corrected
   triplet and joint-feature scans — all consistent with H0).
2. Every contract has **structurally negative expected value** — Differs / Over 0
   / Under 9 have the smallest house edge (1.36 %); Matches has the largest
   (10.71 %).
3. The 60-trade record stored in `trades.json` (56 wins, 3 losses, 1 pending)
   has **P(≥56 wins | true_p = 0.9124) ≈ 0.229** — i.e. consistent with no edge.

Given those facts, the only mathematically honest strategy is:

- **Default to OBSERVE.** No trades.
- **PROBE** with minimum-stake bets on the three lowest-house-edge contracts
  to gather enough data for a Wald Sequential Probability Ratio Test.
- **EXPLOIT** only if SPRT confirms an edge at α = 0.001 — which, on
  truly random data, essentially never happens. The bot's expected
  long-run behavior is OBSERVE.
- **HALT** if daily loss exceeds the configured limit or consecutive-loss
  streak exceeds the configured threshold.

Stake sizing uses fractional Kelly capped at 1 % of bankroll per trade and
collapses to **zero stake when estimated EV ≤ 0**.

## Quickstart

```bash
pip install -r requirements.txt
# Set your Deriv API token in the environment OR config.json (token-less
# runs operate in OBSERVE mode — the bot will see ticks but will not place
# any real trades).
export DERIV_TOKEN=your_token_here
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> for the operator UI. Keyboard shortcuts:
`o` Observe · `a` Auto · `h` Halt · `space` toggle execution.

### Runtime configuration

Settings live in `config.json` (created on first save from the UI) and
include `mode`, `bankroll_usd`, `kelly_fraction`, `max_pct_per_trade`,
`probe_period_ticks`, `daily_loss_limit_usd`, `halt_on_loss_streak`,
`edge_delta`, `sprt_alpha`, `sprt_beta`, `tradeable_contracts`, etc.
All can be tuned at runtime through `POST /config` or the UI.

## Repository layout

```
main.py                      — NHB bot (FastAPI + Deriv WS + strategy)
index.html                   — operator UI (Statistical Observatory)
research/
  EDGE_ANALYSIS.md           — full first-principles writeup
  collect_ticks.py           — pulls a contiguous tick window from Deriv
  ticks_v100_1s.json         — 24h × 86k V100(1s) ticks (research input)
  ticks_v10_1s.json          — 24h × 86k V10(1s) ticks
  audit.py                   — 15 statistical tests
  edge_search.py             — multi-feature × multi-contract edge sweep
  final_check.py             — magnet/drift/Sharpe leaderboard
  test_existing_bot.py       — statistical test of the legacy 60-trade record
  paper_trade.py             — paper-trade the NHB strategy against a tick file
  *.txt                      — pre-computed outputs for inspection
```

## Reproducing the analysis

```bash
cd research
python collect_ticks.py 100000 ticks_v100_1s.json   # pull a fresh sample
python audit.py     ticks_v100_1s.json "V100(1s)"
python edge_search.py ticks_v100_1s.json
python final_check.py
python paper_trade.py ticks_v100_1s.json            # paper-trade the bot
```

Paper-trading the bot against 24 hours of recorded ticks produces:

```
Trades placed:   10,228
Wins / Losses:   9188 / 1040
Win rate:        0.8983
Total stake:     $3,579.80
P&L:             -$55.28      (-1.54 %, matches house edge)
SPRT verdict:    no_edge on all three tradeable contracts
Mode time:       observe 75,985 · probe 10,398 · exploit 0 · halt 0
```

The bot **correctly identifies the absence of edge** and reverts to
OBSERVE for ~88 % of the run. This is the desired result on random data:
the bot stops losing money once it has statistical certainty.
