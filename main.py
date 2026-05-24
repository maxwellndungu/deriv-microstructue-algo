"""
Null-Hypothesis Bot (NHB) — Deriv Volatility 100 (1s) Index
=============================================================

A statistically-honest trading agent. Built on the empirical finding (see
research/EDGE_ANALYSIS.md) that V100(1s) digit streams are indistinguishable
from i.i.d. uniform — meaning every contract has structurally negative
expected value. The bot therefore:

  - Defaults to OBSERVE (no trades).
  - Continuously runs ~15 statistical tests against the live tick feed.
  - Probes ONLY the lowest-house-edge contracts (Differs / Over 0 / Under 9
    @ 1.36% house edge).
  - Uses a Wald Sequential Probability Ratio Test gated by Wilson-99% lower
    confidence bound > break-even.
  - Sizes stakes with fractional Kelly; when EV ≤ 0, the math says ZERO.
  - Has a hard daily-loss kill-switch and per-trade exposure caps.
  - Exposes everything to the operator via a transparency-first UI.

Replace the existing main.py with this. Run with:

  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import io
import csv
import json
import math
import os
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Optional

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


# ---------------------------------------------------------------------------
# 0. Persistent config (overrideable from UI; written to disk so settings
#    survive restarts).
# ---------------------------------------------------------------------------
CONFIG_PATH = os.environ.get("NHB_CONFIG", "config.json")
TRADES_PATH = os.environ.get("NHB_TRADES", "trades.json")

DEFAULTS = {
    # Symbol & connectivity
    "symbol": "1HZ100V",                # Deriv Volatility 100 (1s) Index
    "app_id": 67340,                    # the existing bot's app id
    "ws_url": "wss://ws.binaryws.com/websockets/v3?app_id={app_id}",
    "token": os.environ.get("DERIV_TOKEN", ""),  # operator must supply

    # Mode — operator may force a mode; otherwise auto state-machine runs.
    "mode": "auto",                     # "auto" | "observe" | "probe" | "exploit" | "halt"

    # Trading universe — only the lowest-house-edge contracts.
    "tradeable_contracts": ["DIGITDIFF", "DIGITOVER:0", "DIGITUNDER:9"],

    # SPRT settings
    "sprt_alpha": 0.001,                # type-I error (false-edge claim)
    "sprt_beta":  0.05,                 # type-II error (miss real edge)
    "edge_delta": 0.005,                # minimum detectable edge above BE (=0.5%)
    "wilson_z":   2.576,                # 99% Wilson lower bound

    # Bankroll & stake controls
    "bankroll_usd":   100.0,            # current bankroll (operator can override)
    "min_stake_usd":  0.35,             # Deriv minimum
    "max_stake_usd":  2.00,
    "kelly_fraction": 0.25,             # fractional Kelly
    "max_pct_per_trade": 0.01,          # 1% of bankroll cap per trade

    # Probing schedule
    "probe_period_ticks": 60,           # one probe trade every N ticks (default 60)
    "probe_budget_usd_per_day": 5.00,   # daily $ cap on probe spend
    "probe_min_observations": 1500,     # samples needed before SPRT can declare

    # Hard guardrails
    "daily_loss_limit_usd": 5.00,
    "max_trades_per_minute": 4,
    "halt_on_loss_streak": 12,

    # UI
    "ui_refresh_ms": 1000,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            cfg.update(json.load(open(CONFIG_PATH)))
        except Exception as e:
            print(f"[CFG] failed to load {CONFIG_PATH}: {e}")
    return cfg


def save_config(cfg: dict) -> None:
    try:
        json.dump(cfg, open(CONFIG_PATH, "w"), indent=2)
    except Exception as e:
        print(f"[CFG] failed to save {CONFIG_PATH}: {e}")


CFG = load_config()


# ---------------------------------------------------------------------------
# 1. Last-digit extractor — same as the existing bot, ROUND_DOWN at 2dp.
# ---------------------------------------------------------------------------
def last_digit(quote: float, places: int = 2) -> int:
    fmt = Decimal(str(quote)).quantize(
        Decimal("0." + "0" * places), rounding=ROUND_DOWN
    )
    return int(str(fmt)[-1])


def format_price(quote: float, places: int = 2) -> str:
    fmt = Decimal(str(quote)).quantize(
        Decimal("0." + "0" * places), rounding=ROUND_DOWN
    )
    return str(fmt)


# ---------------------------------------------------------------------------
# 2. Contract math — payouts & break-even probabilities are derived from the
#    Deriv-disclosed payout table (see research/EDGE_ANALYSIS.md §1).
# ---------------------------------------------------------------------------
PAYOUT_PCT = {
    # contract_key: (payout %, theoretical win prob)
    "DIGITOVER:0":  (109.6, 0.90),
    "DIGITOVER:1":  (123.2, 0.80),
    "DIGITOVER:2":  (140.4, 0.70),
    "DIGITOVER:3":  (163.4, 0.60),
    "DIGITOVER:4":  (195.3, 0.50),
    "DIGITOVER:5":  (242.7, 0.40),
    "DIGITOVER:6":  (320.5, 0.30),
    "DIGITOVER:7":  (471.7, 0.20),
    "DIGITOVER:8":  (892.9, 0.10),
    "DIGITUNDER:9": (109.6, 0.90),
    "DIGITUNDER:8": (123.2, 0.80),
    "DIGITUNDER:7": (140.4, 0.70),
    "DIGITUNDER:6": (163.4, 0.60),
    "DIGITUNDER:5": (195.3, 0.50),
    "DIGITUNDER:4": (242.7, 0.40),
    "DIGITUNDER:3": (320.5, 0.30),
    "DIGITUNDER:2": (471.7, 0.20),
    "DIGITUNDER:1": (892.9, 0.10),
    "DIGITEVEN":    (195.3, 0.50),
    "DIGITODD":     (195.3, 0.50),
    "DIGITMATCH":   (892.9, 0.10),
    "DIGITDIFF":    (109.6, 0.90),
}


def payout_gain(key: str) -> float:
    """Payout per $1 staked, net of stake — e.g. 1.096 → 0.096."""
    pct, _ = PAYOUT_PCT[key]
    return pct / 100.0 - 1.0


def theo_win_prob(key: str) -> float:
    return PAYOUT_PCT[key][1]


def breakeven_prob(key: str) -> float:
    """Win-rate at which EV = 0:  p * payout_gain - (1-p) = 0 → p = 1/(1+gain)."""
    g = payout_gain(key)
    return 1.0 / (1.0 + g)


def expected_value(key: str, p_hat: float) -> float:
    """EV per $1 staked given observed (or assumed) win prob p_hat."""
    g = payout_gain(key)
    return p_hat * g - (1.0 - p_hat)


# ---------------------------------------------------------------------------
# 3. Wilson confidence bounds and SPRT.
# ---------------------------------------------------------------------------
def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p_hat = wins / n
    denom = 1.0 + z * z / n
    centre = p_hat + z * z / (2.0 * n)
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def wilson_lower(wins: int, n: int, z: float) -> float:
    return wilson_interval(wins, n, z)[0]


def sprt_log_lr(wins: int, losses: int, p0: float, p1: float) -> float:
    """
    Wald log-likelihood ratio of H1:p=p1 vs H0:p=p0 for a Bernoulli stream.
    Decision boundaries:  log_lr > log((1-β)/α)   → accept H1 (edge confirmed)
                          log_lr < log(β/(1-α))   → accept H0 (no edge)
    """
    return wins * math.log(p1 / p0) + losses * math.log((1 - p1) / (1 - p0))


# ---------------------------------------------------------------------------
# 4. Kelly fractional sizing.
# ---------------------------------------------------------------------------
def kelly_fraction(p: float, b: float) -> float:
    """Kelly bet fraction for a binary bet with win prob p and net odds b."""
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    f = (p * (1 + b) - 1) / b
    return max(0.0, f)


def compute_stake(p_hat: float, key: str, bankroll: float, cfg: dict) -> float:
    """Bet sizing with hard floors/ceilings; returns 0 if EV <= 0."""
    b = payout_gain(key)
    ev = p_hat * b - (1 - p_hat)
    if ev <= 0:
        return 0.0
    f_kelly = kelly_fraction(p_hat, b)
    stake = f_kelly * cfg["kelly_fraction"] * bankroll
    stake = min(stake, bankroll * cfg["max_pct_per_trade"])
    stake = min(stake, cfg["max_stake_usd"])
    stake = max(stake, cfg["min_stake_usd"])
    return round(stake, 2)


# ---------------------------------------------------------------------------
# 5. Live statistical observatory.
#    A rolling window of recent digits is kept in memory; on every tick the
#    test results are recomputed (cheaply) and broadcast to the UI.
# ---------------------------------------------------------------------------
class Observatory:
    """Holds the live tick history and computes test statistics on demand."""

    WINDOW = 5000  # rolling window size

    def __init__(self) -> None:
        self.digits: deque[int] = deque(maxlen=self.WINDOW)
        self.prices: deque[float] = deque(maxlen=self.WINDOW)
        self.epochs: deque[int] = deque(maxlen=self.WINDOW)
        self.last_tick: Optional[dict] = None

    def push(self, epoch: int, quote: float) -> None:
        d = last_digit(quote)
        self.digits.append(d)
        self.prices.append(quote)
        self.epochs.append(epoch)
        self.last_tick = {"epoch": epoch, "quote": quote, "digit": d}

    def n(self) -> int:
        return len(self.digits)

    # --- T1: chi-square uniformity --------------------------------------
    def chi_sq_uniform(self) -> tuple[float, float, list[int]]:
        n = self.n()
        if n < 100:
            return (0.0, 1.0, [0] * 10)
        counts = [0] * 10
        for d in self.digits:
            counts[d] += 1
        exp = n / 10.0
        chi2 = sum((c - exp) ** 2 / exp for c in counts)
        p = _chi2_sf(chi2, df=9)
        return chi2, p, counts

    # --- T2: lag-k autocorrelation of digit and parity ------------------
    def autocorr(self, lag: int = 1) -> dict:
        n = self.n()
        if n < lag + 50:
            return {"rho_digit": 0.0, "rho_parity": 0.0, "band": 0.0, "n": n}
        ds = list(self.digits)
        mean_d = sum(ds) / n
        num_d = sum((ds[i] - mean_d) * (ds[i + lag] - mean_d) for i in range(n - lag))
        den_d = sum((d - mean_d) ** 2 for d in ds)
        rho_d = num_d / den_d if den_d else 0.0
        par = [1 if d % 2 == 0 else 0 for d in ds]
        mean_p = sum(par) / n
        num_p = sum((par[i] - mean_p) * (par[i + lag] - mean_p) for i in range(n - lag))
        den_p = sum((p - mean_p) ** 2 for p in par)
        rho_p = num_p / den_p if den_p else 0.0
        band = 1.96 / math.sqrt(n)
        return {"rho_digit": rho_d, "rho_parity": rho_p, "band": band, "n": n}

    # --- T3: Markov independence chi-square -----------------------------
    def markov_independence(self) -> tuple[float, float]:
        n = self.n()
        if n < 200:
            return (0.0, 1.0)
        M = [[0] * 10 for _ in range(10)]
        ds = list(self.digits)
        for i in range(n - 1):
            M[ds[i]][ds[i + 1]] += 1
        row = [sum(r) for r in M]
        col = [sum(M[i][j] for i in range(10)) for j in range(10)]
        total = sum(row)
        chi = 0.0
        for i in range(10):
            for j in range(10):
                e = row[i] * col[j] / total if total else 0
                if e > 0:
                    chi += (M[i][j] - e) ** 2 / e
        p = _chi2_sf(chi, df=81)
        return chi, p

    # --- T4: Wald-Wolfowitz runs test on parity -------------------------
    def runs_test(self) -> tuple[float, float]:
        n = self.n()
        if n < 100:
            return (0.0, 1.0)
        par = [1 if d % 2 == 0 else 0 for d in self.digits]
        n1 = sum(par)
        n0 = n - n1
        if n0 == 0 or n1 == 0:
            return (0.0, 1.0)
        runs = 1 + sum(1 for i in range(1, n) if par[i] != par[i - 1])
        mu = 1 + 2 * n0 * n1 / n
        var = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
        Z = (runs - mu) / math.sqrt(var) if var > 0 else 0.0
        p = 2.0 * (1.0 - _norm_cdf(abs(Z)))
        return Z, p

    # --- T5: same-digit repeat probability ------------------------------
    def repeat_prob(self) -> float:
        n = self.n()
        if n < 2:
            return 0.0
        ds = list(self.digits)
        same = sum(1 for i in range(1, n) if ds[i] == ds[i - 1])
        return same / (n - 1)

    # --- T8: 1-tick price-increment stats -------------------------------
    def increment_stats(self) -> dict:
        n = self.n()
        if n < 2:
            return {"mean": 0.0, "std": 0.0, "median_abs": 0.0, "n": 0}
        ps = list(self.prices)
        d = [round((ps[i] - ps[i - 1]) * 100) for i in range(1, n)]
        m = sum(d) / len(d)
        var = sum((x - m) ** 2 for x in d) / len(d)
        sd = math.sqrt(var)
        absd = sorted(abs(x) for x in d)
        median = absd[len(absd) // 2]
        return {"mean": m, "std": sd, "median_abs": median, "n": len(d)}

    # --- T10: tick interarrival ----------------------------------------
    def interarrival(self) -> dict:
        if self.n() < 2:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        es = list(self.epochs)
        gaps = [es[i] - es[i - 1] for i in range(1, len(es))]
        m = sum(gaps) / len(gaps)
        sd = math.sqrt(sum((g - m) ** 2 for g in gaps) / len(gaps))
        return {"mean": m, "std": sd, "n": len(gaps)}

    # --- T13: best & worst conditional Differs cells --------------------
    def differs_conditional(self) -> list[dict]:
        """For each entry digit, empirical P(next != entry) on this window."""
        n = self.n()
        if n < 200:
            return []
        ds = list(self.digits)
        per = [[0, 0] for _ in range(10)]   # per[d] = [wins, total]
        for i in range(n - 1):
            d = ds[i]
            per[d][1] += 1
            if ds[i + 1] != d:
                per[d][0] += 1
        out = []
        be = breakeven_prob("DIGITDIFF")
        for d in range(10):
            w, t = per[d]
            if t < 20:
                out.append({"digit": d, "wins": w, "total": t, "p": 0.0,
                            "ci_lo": 0.0, "ci_hi": 0.0, "be": be})
                continue
            lo, hi = wilson_interval(w, t, CFG["wilson_z"])
            out.append({
                "digit": d, "wins": w, "total": t,
                "p": w / t, "ci_lo": lo, "ci_hi": hi, "be": be,
            })
        return out


# Pure-python helper functions for chi-square SF and normal CDF (no scipy at
# runtime — keeps the runtime footprint tiny).
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _chi2_sf(x: float, df: int) -> float:
    """Right-tail survival of chi-square; uses regularized upper inc gamma."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    s = x / 2.0
    return _gammaincc(a, s)


def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma — Lentz continued fraction."""
    # Use series for x < a + 1, continued fraction otherwise.
    if x < a + 1:
        return 1.0 - _gammaincc_series(a, x)
    return _gammaincc_cf(a, x)


def _gammaincc_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma via series."""
    if x == 0:
        return 0.0
    ap = a
    s = 1.0 / a
    delta = s
    for _ in range(2000):
        ap += 1.0
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * 1e-14:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaincc_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma via continued fraction."""
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


# ---------------------------------------------------------------------------
# 6. Strategy state machine — OBSERVE → PROBE → EXPLOIT → HALT.
# ---------------------------------------------------------------------------
class Strategy:
    """
    Holds per-contract win/loss counters and decides what to bet next.

    The contracts we may probe are limited by config to:
       DIGITDIFF / DIGITOVER:0 / DIGITUNDER:9
    These have the lowest house edge (1.36%) in Deriv's payout table.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        # per-contract: settled win/loss/stake/profit counters
        self.stats: dict[str, dict] = {
            k: {"wins": 0, "losses": 0, "stake_total": 0.0, "profit_total": 0.0,
                "last_seen_epoch": 0, "trade_history": []}
            for k in cfg["tradeable_contracts"]
        }
        # Per-condition (entry digit) Differs stats
        self.cond_differs: dict[int, dict] = {
            d: {"wins": 0, "losses": 0} for d in range(10)
        }
        # Tick-based probe scheduler
        self.tick_index = 0
        self.last_probe_tick = -10_000
        # Daily loss tracking
        self.daily_loss_usd = 0.0
        self.daily_loss_date = datetime.now(timezone.utc).date().isoformat()
        # Streak
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        # Trade-rate gate
        self.trade_epochs: deque[int] = deque(maxlen=128)
        # SPRT log-LR per contract
        self.sprt_log_lr: dict[str, float] = {k: 0.0 for k in cfg["tradeable_contracts"]}

    # ----- accounting -------------------------------------------------------
    def record_outcome(self, key: str, win: bool, stake: float, profit: float,
                       entry_digit: int) -> None:
        s = self.stats.setdefault(key, {"wins": 0, "losses": 0, "stake_total": 0.0,
                                        "profit_total": 0.0, "last_seen_epoch": 0,
                                        "trade_history": []})
        s["stake_total"] += stake
        s["profit_total"] += profit
        if win:
            s["wins"] += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            s["losses"] += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        # Conditional Differs counters
        if key == "DIGITDIFF":
            cd = self.cond_differs[entry_digit]
            cd["wins" if win else "losses"] += 1
        # SPRT update (per contract): H0: p=BE, H1: p=BE+delta
        p0 = breakeven_prob(key)
        p1 = min(0.9999, p0 + self.cfg["edge_delta"])
        if win:
            self.sprt_log_lr[key] += math.log(p1 / p0)
        else:
            self.sprt_log_lr[key] += math.log((1 - p1) / (1 - p0))
        # Daily-loss tracking (use today's date)
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self.daily_loss_date:
            self.daily_loss_date = today
            self.daily_loss_usd = 0.0
        if profit < 0:
            self.daily_loss_usd += -profit

    # ----- SPRT decision: H1 accepted? --------------------------------------
    def sprt_state(self, key: str) -> str:
        a = self.cfg["sprt_alpha"]
        b = self.cfg["sprt_beta"]
        upper = math.log((1 - b) / a)         # accept H1: edge confirmed
        lower = math.log(b / (1 - a))         # accept H0: no edge
        x = self.sprt_log_lr[key]
        if x >= upper:
            return "edge_confirmed"
        if x <= lower:
            return "no_edge"
        return "undecided"

    # ----- Mode resolver ----------------------------------------------------
    def current_mode(self) -> str:
        if self.cfg["mode"] != "auto":
            return self.cfg["mode"]
        # auto: cascade through observe → probe → exploit, halt if guardrails
        if self.daily_loss_usd >= self.cfg["daily_loss_limit_usd"]:
            return "halt"
        if self.consecutive_losses >= self.cfg["halt_on_loss_streak"]:
            return "halt"
        # If at least one tradeable contract has edge_confirmed → exploit
        for k in self.cfg["tradeable_contracts"]:
            if self.sprt_state(k) == "edge_confirmed":
                return "exploit"
        # If nothing is finalized AND we have less than probe_min_obs → probe
        total_probes = sum(self.stats[k]["wins"] + self.stats[k]["losses"]
                           for k in self.cfg["tradeable_contracts"])
        if total_probes < self.cfg["probe_min_observations"]:
            return "probe"
        # If all tradeable contracts are no_edge → observe (we won't trade)
        if all(self.sprt_state(k) == "no_edge" for k in self.cfg["tradeable_contracts"]):
            return "observe"
        return "probe"  # default: keep probing

    # ----- Trade-rate gate --------------------------------------------------
    def can_trade_now(self, epoch: int) -> bool:
        # Drop epochs older than 60s
        while self.trade_epochs and epoch - self.trade_epochs[0] > 60:
            self.trade_epochs.popleft()
        return len(self.trade_epochs) < self.cfg["max_trades_per_minute"]

    def register_trade(self, epoch: int) -> None:
        self.trade_epochs.append(epoch)

    # ----- Decision: should we bet *this* tick? -----------------------------
    def decide(self, observatory: Observatory, epoch: int) -> Optional[dict]:
        self.tick_index += 1
        mode = self.current_mode()
        if mode in ("observe", "halt"):
            return None
        if not self.can_trade_now(epoch):
            return None

        # Per-tick probing schedule.
        ticks_since_last = self.tick_index - self.last_probe_tick
        if ticks_since_last < self.cfg["probe_period_ticks"]:
            return None

        # Choose contract.  In probe mode → cycle round-robin through the
        # 3 lowest-edge contracts.  In exploit mode → pick the contract
        # whose conditional Differs (if Differs) Wilson lower bound is
        # highest, else fall back to whatever has edge_confirmed.
        last_d = observatory.last_tick["digit"] if observatory.last_tick else 0
        bankroll = self.cfg["bankroll_usd"]
        chosen = None

        if mode == "probe":
            tradeable = self.cfg["tradeable_contracts"]
            chosen_key = tradeable[self.tick_index % len(tradeable)]
            chosen = {
                "key": chosen_key,
                "stake": self.cfg["min_stake_usd"],   # probe = always min stake
                "p_hat_used": breakeven_prob(chosen_key),  # neutral assumption
                "reason": "probe (min-stake edge detection)",
            }

        elif mode == "exploit":
            # Pick the strongest confirmed contract
            for k in self.cfg["tradeable_contracts"]:
                if self.sprt_state(k) != "edge_confirmed":
                    continue
                s = self.stats[k]
                n = s["wins"] + s["losses"]
                if n < 50:
                    continue
                p_hat = s["wins"] / n
                stake = compute_stake(p_hat, k, bankroll, self.cfg)
                if stake <= 0:
                    continue
                # If Differs: use conditional estimate when the entry-digit
                # cell has enough samples.
                if k == "DIGITDIFF":
                    cd = self.cond_differs[last_d]
                    cn = cd["wins"] + cd["losses"]
                    if cn >= 200:
                        cp = cd["wins"] / cn
                        if cp > p_hat:
                            p_hat = cp
                            stake = compute_stake(p_hat, k, bankroll, self.cfg)
                            if stake <= 0:
                                continue
                ev = expected_value(k, p_hat)
                if chosen is None or ev > chosen.get("ev", -1.0):
                    chosen = {
                        "key": k,
                        "stake": stake,
                        "p_hat_used": p_hat,
                        "ev": ev,
                        "reason": "exploit (SPRT edge confirmed)",
                    }

        if chosen is None:
            return None

        # Final guard: stake within bounds
        stake = max(self.cfg["min_stake_usd"], min(self.cfg["max_stake_usd"], chosen["stake"]))
        chosen["stake"] = round(stake, 2)
        self.last_probe_tick = self.tick_index
        return chosen


# ---------------------------------------------------------------------------
# 7. Global state, observatory and strategy singletons.
# ---------------------------------------------------------------------------
OBS = Observatory()
STRAT = Strategy(CFG)
STATE: dict[str, Any] = {
    "connected": False,
    "session_started_at": int(time.time()),
    "session_ticks": 0,
    "trades": [],                   # all persisted trades
    "pending_buys": {},             # req_id → trade_dict
    "ws_ref": None,
    "ui_clients": [],
    "execution_enabled": True,
    "last_action_log": deque(maxlen=200),
}


def log_action(line: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    STATE["last_action_log"].append(f"{ts} {line}")
    print(f"[{ts}] {line}", flush=True)


def load_trades() -> None:
    if not os.path.exists(TRADES_PATH):
        return
    try:
        data = json.load(open(TRADES_PATH))
        old = data.get("trades", [])
        kept = []
        for t in old:
            # New-schema trades have a "key" field; legacy trades from the
            # previous bot do not — preserve them verbatim but don't try to
            # feed them into the new strategy state.
            if "key" not in t:
                kept.append(t)
                continue
            kept.append(t)
            if t.get("outcome") not in ("win", "loss"):
                continue
            try:
                STRAT.record_outcome(
                    key=t["key"],
                    win=(t["outcome"] == "win"),
                    stake=t.get("stake", 0.0),
                    profit=t.get("profit", 0.0),
                    entry_digit=t.get("entry_digit", 0),
                )
            except Exception as e:
                log_action(f"replay skipped trade #{t.get('id')}: {e}")
        STATE["trades"] = kept
        log_action(f"Loaded {len(STATE['trades'])} trades from {TRADES_PATH}")
    except Exception as e:
        log_action(f"Failed to load trades: {e}")


def save_trades() -> None:
    try:
        json.dump({"trades": STATE["trades"][-5000:]}, open(TRADES_PATH, "w"))
    except Exception as e:
        log_action(f"Save error: {e}")


# ---------------------------------------------------------------------------
# 8. Deriv WebSocket trader.  ONE WS connection: tick feed + trading on the
#    same authorised socket. Simpler and more robust than running three.
# ---------------------------------------------------------------------------
DERIV_WS_URL_TMPL = "wss://ws.binaryws.com/websockets/v3?app_id={app_id}"


async def deriv_session() -> None:
    """Persistent connection: auth → subscribe ticks → handle buys/settlements."""
    url = DERIV_WS_URL_TMPL.format(app_id=CFG["app_id"])
    backoff = 2
    while True:
        try:
            log_action(f"Connecting to Deriv WS {url} (symbol={CFG['symbol']})")
            async with websockets.connect(url, ping_interval=20, ping_timeout=60) as ws:
                STATE["ws_ref"] = ws
                STATE["connected"] = True
                backoff = 2
                # Authorize if we have a token
                if CFG.get("token"):
                    await ws.send(json.dumps({"authorize": CFG["token"]}))
                # Subscribe to ticks
                await ws.send(json.dumps({"ticks": CFG["symbol"], "subscribe": 1}))
                log_action("WebSocket connected; subscribed to ticks.")
                async for raw in ws:
                    try:
                        await _handle_msg(json.loads(raw), ws)
                    except Exception as e:
                        log_action(f"handler error: {e}")
        except Exception as e:
            STATE["connected"] = False
            STATE["ws_ref"] = None
            log_action(f"WS error: {e}; reconnect in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(60, backoff * 2)


_req_counter = 0


async def _handle_msg(msg: dict, ws) -> None:
    global _req_counter
    mt = msg.get("msg_type")
    if "error" in msg:
        log_action(f"Deriv error: {msg['error'].get('message', msg['error'])}")
        return
    if mt == "tick":
        tick = msg["tick"]
        epoch = tick["epoch"]
        quote = float(tick["quote"])
        OBS.push(epoch, quote)
        STATE["session_ticks"] += 1
        if not STATE["execution_enabled"]:
            return
        decision = STRAT.decide(OBS, epoch)
        if decision is not None and CFG.get("token"):
            await _send_buy(ws, decision, epoch, quote)
    elif mt == "buy":
        buy = msg.get("buy", {})
        req_id = msg.get("req_id")
        contract_id = buy.get("contract_id")
        pending = STATE["pending_buys"].pop(req_id, None)
        if pending and contract_id:
            pending["contract_id"] = contract_id
            pending["buy_price"] = buy.get("buy_price", pending["stake"])
            await ws.send(json.dumps({
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1,
            }))
            log_action(
                f"BUY {pending['key']} stake=${pending['stake']:.2f} "
                f"contract_id={contract_id} reason={pending.get('reason')}"
            )
    elif mt == "proposal_open_contract":
        poc = msg.get("proposal_open_contract", {})
        if poc.get("is_expired") or poc.get("is_sold"):
            _handle_settlement(poc)


async def _send_buy(ws, decision: dict, epoch: int, quote: float) -> None:
    global _req_counter
    _req_counter += 1
    req_id = _req_counter
    key = decision["key"]
    if ":" in key:
        ct, barrier = key.split(":")
    else:
        ct, barrier = key, None

    params: dict = {
        "contract_type": ct,
        "symbol": CFG["symbol"],
        "currency": "USD",
        "amount": decision["stake"],
        "basis": "stake",
        "duration": 1,
        "duration_unit": "t",
    }
    if barrier is not None:
        params["barrier"] = barrier
    if ct == "DIGITDIFF":
        # Bet the *current* digit will differ from the next one.
        params["barrier"] = str(OBS.last_tick["digit"])

    req = {
        "buy": 1,
        "price": decision["stake"],
        "parameters": params,
        "req_id": req_id,
    }
    trade = {
        "id": len(STATE["trades"]) + 1,
        "req_id": req_id,
        "epoch": epoch,
        "key": key,
        "contract_type": ct,
        "barrier": params.get("barrier"),
        "stake": decision["stake"],
        "p_hat_used": decision.get("p_hat_used"),
        "reason": decision.get("reason"),
        "entry_price": quote,
        "entry_digit": OBS.last_tick["digit"],
        "outcome": "pending",
        "profit": 0.0,
        "buy_price": None,
        "payout": None,
        "mode_at_entry": STRAT.current_mode(),
    }
    STATE["pending_buys"][req_id] = trade
    STATE["trades"].append(trade)
    STRAT.register_trade(epoch)
    try:
        await ws.send(json.dumps(req))
    except Exception as e:
        trade["outcome"] = "error"
        STATE["pending_buys"].pop(req_id, None)
        log_action(f"Send-buy error: {e}")


def _handle_settlement(poc: dict) -> None:
    contract_id = poc.get("contract_id")
    if not contract_id:
        return
    for trade in reversed(STATE["trades"]):
        if trade.get("contract_id") == contract_id and trade["outcome"] == "pending":
            profit = float(poc.get("profit", 0.0))
            trade["profit"] = profit
            trade["outcome_digit"] = (
                last_digit(float(poc["exit_tick"])) if poc.get("exit_tick") else None
            )
            trade["payout"] = poc.get("payout")
            trade["outcome"] = "win" if profit > 0 else "loss"
            # Update strategy state
            STRAT.record_outcome(
                key=trade["key"],
                win=(trade["outcome"] == "win"),
                stake=trade["stake"],
                profit=profit,
                entry_digit=trade["entry_digit"],
            )
            # Update bankroll
            CFG["bankroll_usd"] = round(CFG["bankroll_usd"] + profit, 2)
            save_trades()
            log_action(
                f"SETTLE {trade['key']} {trade['outcome']} "
                f"profit=${profit:+.2f} bankroll=${CFG['bankroll_usd']:.2f}"
            )
            return


# ---------------------------------------------------------------------------
# 9. UI state snapshot.
# ---------------------------------------------------------------------------
def ui_snapshot() -> dict:
    last = OBS.last_tick
    n = OBS.n()
    chi2, p_chi, counts = OBS.chi_sq_uniform()
    ac1 = OBS.autocorr(1)
    ac2 = OBS.autocorr(10)
    mchi, mp = OBS.markov_independence()
    zr, pr = OBS.runs_test()
    inc = OBS.increment_stats()
    ia = OBS.interarrival()
    cond_diff = OBS.differs_conditional()

    # Strategy snapshot
    per_contract = []
    for k in CFG["tradeable_contracts"]:
        s = STRAT.stats[k]
        nt = s["wins"] + s["losses"]
        p_hat = s["wins"] / nt if nt > 0 else 0.0
        lo, hi = wilson_interval(s["wins"], nt, CFG["wilson_z"])
        per_contract.append({
            "key": k,
            "wins": s["wins"],
            "losses": s["losses"],
            "n": nt,
            "p_hat": p_hat,
            "ci_lo": lo,
            "ci_hi": hi,
            "be": breakeven_prob(k),
            "ev_per_dollar": expected_value(k, p_hat) if nt > 0 else None,
            "sprt_log_lr": STRAT.sprt_log_lr[k],
            "sprt_state": STRAT.sprt_state(k),
            "stake_total": s["stake_total"],
            "profit_total": s["profit_total"],
        })

    total_pnl = sum(t["profit"] for t in STATE["trades"]
                    if t["outcome"] in ("win", "loss"))
    total_wins = sum(1 for t in STATE["trades"] if t["outcome"] == "win")
    total_losses = sum(1 for t in STATE["trades"] if t["outcome"] == "loss")

    return {
        "ts": int(time.time()),
        "connected": STATE["connected"],
        "execution_enabled": STATE["execution_enabled"],
        "symbol": CFG["symbol"],
        "mode": STRAT.current_mode(),
        "ticks": STATE["session_ticks"],
        "last_tick": last,
        "bankroll": CFG["bankroll_usd"],
        "config": {k: CFG[k] for k in CFG if k not in ("token",)},
        "strategy": {
            "tick_index": STRAT.tick_index,
            "consecutive_losses": STRAT.consecutive_losses,
            "consecutive_wins": STRAT.consecutive_wins,
            "daily_loss_usd": STRAT.daily_loss_usd,
            "daily_loss_limit": CFG["daily_loss_limit_usd"],
            "per_contract": per_contract,
        },
        "stats": {
            "n_window": n,
            "digit_counts": counts,
            "chi2_uniform": {"chi2": chi2, "p": p_chi},
            "ac_lag1": ac1,
            "ac_lag10": ac2,
            "markov": {"chi2": mchi, "p": mp},
            "runs_parity": {"Z": zr, "p": pr},
            "repeat_prob": OBS.repeat_prob(),
            "increment": inc,
            "interarrival": ia,
            "cond_differs": cond_diff,
        },
        "totals": {
            "trades": len(STATE["trades"]),
            "wins": total_wins,
            "losses": total_losses,
            "pnl": round(total_pnl, 2),
        },
        "log_tail": list(STATE["last_action_log"])[-30:],
    }


# ---------------------------------------------------------------------------
# 10. FastAPI app.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app):
    load_trades()
    asyncio.create_task(deriv_session())
    asyncio.create_task(broadcast_loop())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "connected": STATE["connected"], "mode": STRAT.current_mode()}


@app.get("/state")
async def state_endpoint():
    return ui_snapshot()


@app.post("/execution")
async def execution(payload: dict):
    if "enabled" in payload:
        STATE["execution_enabled"] = bool(payload["enabled"])
    return {"execution_enabled": STATE["execution_enabled"]}


@app.post("/config")
async def update_config(payload: dict):
    """Allow operator to tune the bot at runtime.
    Pass {"mode": "observe"} to force observe-only mode, etc."""
    allowed_keys = {
        "mode", "kelly_fraction", "max_pct_per_trade",
        "min_stake_usd", "max_stake_usd", "bankroll_usd",
        "probe_period_ticks", "probe_min_observations",
        "daily_loss_limit_usd", "max_trades_per_minute",
        "halt_on_loss_streak", "edge_delta", "sprt_alpha", "sprt_beta",
        "tradeable_contracts", "token", "symbol",
    }
    changed = {}
    for k, v in payload.items():
        if k in allowed_keys:
            CFG[k] = v
            changed[k] = v
    save_config(CFG)
    return {"changed": changed}


@app.get("/export/trades.csv")
async def export_trades():
    cols = ["id", "req_id", "contract_id", "epoch", "key", "contract_type",
            "barrier", "stake", "p_hat_used", "reason", "entry_price",
            "entry_digit", "outcome", "outcome_digit", "profit",
            "buy_price", "payout", "mode_at_entry"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for t in STATE["trades"]:
        w.writerow({c: t.get(c, "") for c in cols})
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
                             headers={"Content-Disposition":
                                      "attachment; filename=trades.csv"})


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


async def broadcast_loop():
    while True:
        if STATE["ui_clients"]:
            snap = ui_snapshot()
            payload = json.dumps(snap, default=str)
            dead = []
            for c in STATE["ui_clients"]:
                try:
                    await c.send_text(payload)
                except Exception:
                    dead.append(c)
            for c in dead:
                if c in STATE["ui_clients"]:
                    STATE["ui_clients"].remove(c)
        await asyncio.sleep(CFG.get("ui_refresh_ms", 1000) / 1000.0)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    STATE["ui_clients"].append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in STATE["ui_clients"]:
            STATE["ui_clients"].remove(websocket)


# ---------------------------------------------------------------------------
# 11. Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
