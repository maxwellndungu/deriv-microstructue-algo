"""
Paper-trade the NHB bot against the recorded 24-hour V100(1s) tick stream
in research/ticks_v100_1s.json. We use the same Strategy + Observatory
code from main.py — no shortcuts.

Contract resolution mirrors Deriv's official mechanics
(https://deriv.com/terms-and-conditions/trading-terms §2.2.3.1):

    decision tick : i
    entry  spot   : i + 1   ("next tick after our servers process the contract")
    exit   spot   : i + 1 + DURATION    (for tick-duration contracts)

For the default 1-tick contract:  decision=i  →  exit=i+2  (LAG_EXIT=2).

The CLI flag --lag overrides LAG_EXIT for what-if checks (1 reproduces the
old, incorrect simulation).

Run:  python3 paper_trade.py [path_to_ticks.json] [--lag N]
"""
import json
import math
import sys
from pathlib import Path

# Adjust path so we can import main from the parent.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import main  # type: ignore

# Disable WS / file I/O side-effects
main.CFG["mode"] = "auto"
main.CFG["token"] = ""                 # no real trades
main.CFG["probe_period_ticks"] = 1     # probe every tick to maximize sample
main.CFG["probe_min_observations"] = 1500
main.CFG["max_trades_per_minute"] = 60 # don't rate-limit in paper
# Lift the operator guardrails so the bot can accumulate a large sample
# inside one paper-trade run; the guardrails are still ENFORCED in live
# deployment — they're just disabled here to demonstrate SPRT convergence.
main.CFG["daily_loss_limit_usd"] = 1_000_000
main.CFG["halt_on_loss_streak"]  = 1_000

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else HERE / "ticks_v100_1s.json"
LAG_EXIT = 2                                          # decision i → exit i+LAG_EXIT
for a in sys.argv[1:]:
    if a.startswith("--lag"):
        LAG_EXIT = int(a.split("=", 1)[1]) if "=" in a else int(sys.argv[sys.argv.index(a)+1])
if not PATH.exists() and PATH.suffix != ".gz":
    alt = PATH.with_suffix(PATH.suffix + ".gz")
    if alt.exists():
        PATH = alt
if PATH.suffix == ".gz":
    import gzip
    ticks = json.loads(gzip.decompress(PATH.read_bytes()))
else:
    ticks = json.loads(PATH.read_text())
print(f"Paper-trading {len(ticks)} ticks from {PATH.name}  (LAG_EXIT={LAG_EXIT})")

obs = main.Observatory()
strat = main.Strategy(main.CFG)

placed = 0
wins = 0
losses = 0
pnl_running = 0.0
mode_counts = {"observe": 0, "probe": 0, "exploit": 0, "halt": 0}
per_contract = {k: {"wins": 0, "losses": 0, "stake": 0.0, "pnl": 0.0}
                for k in main.CFG["tradeable_contracts"]}
mode_at_first_exploit = None

for i, t in enumerate(ticks):
    epoch, quote = t[0], float(t[1])
    obs.push(epoch, quote)
    mode = strat.current_mode()
    mode_counts[mode] = mode_counts.get(mode, 0) + 1
    if mode in ("observe", "halt"):
        continue
    decision = strat.decide(obs, epoch)
    if decision is None:
        continue
    # Resolve at the *exit* tick (per Deriv: decision i → entry i+1 → exit i+1+duration).
    # For 1-tick duration contracts that's i+2 (LAG_EXIT=2).
    if i + LAG_EXIT >= len(ticks):
        break
    exit_quote = float(ticks[i + LAG_EXIT][1])
    exit_digit = main.last_digit(exit_quote)
    # `entry_digit` here is the digit at the decision tick, which is what the bot
    # passes as the DIGITDIFF barrier when calling Deriv's buy API.
    entry_digit = main.last_digit(quote)
    key = decision["key"]
    stake = decision["stake"]

    if key == "DIGITDIFF":
        win = exit_digit != entry_digit
    elif key == "DIGITOVER:0":
        win = exit_digit > 0
    elif key == "DIGITUNDER:9":
        win = exit_digit < 9
    else:
        win = False

    payout = main.payout_gain(key)
    profit = stake * payout if win else -stake
    pnl_running += profit
    placed += 1
    if win:
        wins += 1
    else:
        losses += 1
    pc = per_contract[key]
    pc["wins" if win else "losses"] += 1
    pc["stake"] += stake
    pc["pnl"] += profit

    strat.register_trade(epoch)
    strat.record_outcome(key=key, win=win, stake=stake, profit=profit,
                         entry_digit=entry_digit)

    if mode_at_first_exploit is None and mode == "exploit":
        mode_at_first_exploit = i

print("\n=== RESULTS ===")
print(f"Ticks consumed:   {len(ticks)}")
print(f"Trades placed:    {placed}")
print(f"Wins / Losses:    {wins} / {losses}")
if placed:
    print(f"Win rate:         {wins/placed:.4f}")
    print(f"Total stake:      ${sum(pc['stake'] for pc in per_contract.values()):.2f}")
    print(f"Total P&L:        ${pnl_running:+.2f}")
    print(f"Return on stake:  {pnl_running / sum(pc['stake'] for pc in per_contract.values()) * 100:.3f}%")
print(f"Mode time spent:  {mode_counts}")
print(f"First entered EXPLOIT at tick: {mode_at_first_exploit}")

print("\nPer-contract:")
for k, pc in per_contract.items():
    n = pc["wins"] + pc["losses"]
    if n == 0:
        print(f"  {k}: no trades")
        continue
    p = pc["wins"] / n
    be = main.breakeven_prob(k)
    lo, hi = main.wilson_interval(pc["wins"], n, 2.576)
    print(f"  {k}: n={n}  p̂={p:.4f}  CI99=[{lo:.4f},{hi:.4f}]  BE={be:.4f}  P&L=${pc['pnl']:+.2f}  SPRT={strat.sprt_state(k)}")
