"""
Digit Microstructure Trading System — Phase 1: Live Feed + Visualization
Deriv Volatility 10 (1s) Index — 1HZ10V
"""

import asyncio
import csv
import io
import json
import time
from decimal import Decimal, ROUND_DOWN
from collections import Counter
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
import websockets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=67340"
DERIV_TOKEN = "iGwTSVES9MsY9bv"
SYMBOL = "1HZ10V"
DECIMAL_PLACES = 2
TICKS_PER_MINI = 10
MINIS_PER_CLUSTER = 6
TICKS_PER_CLUSTER = TICKS_PER_MINI * MINIS_PER_CLUSTER
MAX_TRADES_PER_RETEST = 1
TRADES_FILE = "trades.json"

# Stake configuration
MIN_STAKE = 0.35
MAX_STAKE = 2.00

# ---------------------------------------------------------------------------
# Digit extraction (handles trailing-zero bug)
# ---------------------------------------------------------------------------
def extract_digit(quote_float: float, decimal_places: int = DECIMAL_PLACES) -> int:
    fmt = Decimal(str(quote_float)).quantize(
        Decimal('0.' + '0' * decimal_places), rounding=ROUND_DOWN
    )
    return int(str(fmt)[-1])


def format_price(quote_float: float, decimal_places: int = DECIMAL_PLACES) -> str:
    fmt = Decimal(str(quote_float)).quantize(
        Decimal('0.' + '0' * decimal_places), rounding=ROUND_DOWN
    )
    return str(fmt)

# ---------------------------------------------------------------------------
# Probability and Stake Calculators
# ---------------------------------------------------------------------------
def calculate_probability(contract_type: str, barrier) -> float:
    """Calculate win probability for a given contract and barrier."""
    if contract_type == "DIGITMATCH":
        return 0.10
    if contract_type == "DIGITDIFF":
        return 0.90
    if contract_type in ("DIGITEVEN", "DIGITODD"):
        return 0.50
    if contract_type == "DIGITOVER" and barrier is not None:
        return (9 - int(barrier)) / 10
    if contract_type == "DIGITUNDER" and barrier is not None:
        return int(barrier) / 10
    return 0.50  # fallback


def calculate_stake(contract_type: str, barrier, engine_id: str) -> float:
    """
    Calculate stake based on contract probability and engine P&L state.
    Range: $0.35 minimum to $2.00 maximum.
    """
    # MATCH — always minimum
    if contract_type == "DIGITMATCH":
        return MIN_STAKE

    # DIFFER — always maximum
    if contract_type == "DIGITDIFF":
        return MAX_STAKE

    # EVEN/ODD — depends on engine P&L state
    if contract_type in ("DIGITEVEN", "DIGITODD"):
        pnl = state["engine_pnl"][engine_id]["net_pnl"]
        if pnl > 0:
            return 1.00
        elif pnl < 0:
            return 0.50
        else:
            return 0.75

    # OVER/UNDER — linear scaling by probability
    prob = calculate_probability(contract_type, barrier)
    stake = MIN_STAKE + (prob * (MAX_STAKE - MIN_STAKE))
    stake = max(MIN_STAKE, min(MAX_STAKE, stake))
    return round(stake, 2)


def get_active_trade_count() -> int:
    """Count trades currently pending settlement."""
    return sum(1 for t in state["trades"] if t["outcome"] == "pending")


def get_adjusted_stake(base_stake: float) -> float:
    """
    Adjust stake based on global exposure rule.
    Max total exposure per cycle = $2.50
    """
    MAX_EXPOSURE = 2.50
    active = get_active_trade_count()
    if active == 0:
        return base_stake
    adjusted = base_stake / active
    return round(max(MIN_STAKE, adjusted), 2)

# ---------------------------------------------------------------------------
# Mini candle builder
# ---------------------------------------------------------------------------
def build_mini_candle(ticks: list, mini_id: int) -> dict:
    prices = [t["quote"] for t in ticks]
    epochs = [t["epoch"] for t in ticks]
    high = max(prices)
    low = min(prices)
    range_val = round(high - low, DECIMAL_PLACES)

    # Tapped prices (unique, formatted)
    tapped_set = set()
    for p in prices:
        tapped_set.add(format_price(p))

    # All possible prices from low to high at 0.01 increments
    all_prices = set()
    current = Decimal(str(low)).quantize(Decimal('0.01'), rounding=ROUND_DOWN)
    high_d = Decimal(str(high)).quantize(Decimal('0.01'), rounding=ROUND_DOWN)
    while current <= high_d:
        all_prices.add(str(current))
        current += Decimal('0.01')

    untapped_set = all_prices - tapped_set

    tapped_digits = [int(p[-1]) for p in sorted(tapped_set)]
    untapped_digits = [int(p[-1]) for p in sorted(untapped_set)]

    # Digit frequency from tapped prices (including duplicates from all ticks)
    all_tick_digits = [extract_digit(p) for p in prices]
    digit_freq = Counter(all_tick_digits)

    untapped_digit_freq = {d: untapped_digits.count(d) for d in range(10)}

    # Full range digits (all possible prices low→high)
    full_range_digits = [int(p[-1]) for p in sorted(all_prices)]

    # Run engines
    e1_signals = run_engine1(tapped_digits)
    e2_signals = run_engine2(untapped_digits)
    e3_signals = run_engine3(full_range_digits)

    return {
        "mini_id": mini_id,
        "epoch_start": min(epochs),
        "epoch_end": max(epochs),
        "high": high,
        "low": low,
        "range": range_val,
        "tick_count": len(ticks),
        "tapped_prices": sorted(tapped_set),
        "tapped_digits": tapped_digits,
        "untapped_prices": sorted(untapped_set),
        "untapped_digits": untapped_digits,
        "digit_freq": {d: digit_freq.get(d, 0) for d in range(10)},
        "untapped_digit_freq": untapped_digit_freq,
        "e1_signals": e1_signals,
        "e2_signals": e2_signals,
        "e3_signals": e3_signals,
        "ticks": ticks,
    }

# ---------------------------------------------------------------------------
# Over/Under dynamic barrier helper
# ---------------------------------------------------------------------------
def get_over_under_signals(digits: list, engine_label: str) -> list:
    """
    Dynamically finds the strongest Over/Under barrier based on
    actual digit distribution. Tests all valid barriers 0-8.
    Returns the single strongest OVER and single strongest UNDER signal.
    """
    if not digits:
        return []

    total = len(digits)
    signals = []
    best_over = None
    best_under = None

    for barrier in range(0, 9):  # barriers 0 through 8
        # OVER barrier: digits strictly greater than barrier
        over_count = sum(1 for d in digits if d > barrier)
        over_pct = over_count / total

        # UNDER barrier: digits strictly less than barrier + 1
        under_count = sum(1 for d in digits if d < barrier + 1)
        under_pct = under_count / total

        # Over signal strength — how far above 50% is it
        if over_pct > 0.5:
            over_strength = round((over_pct - 0.5) * 2, 3)  # 0.0 to 1.0
            if best_over is None or over_strength > best_over["signal_strength"]:
                best_over = {
                    "engine": engine_label,
                    "contract_type": "DIGITOVER",
                    "barrier": str(barrier),
                    "signal": f"{over_count}/{total} digits > {barrier} ({over_pct:.0%})",
                    "signal_strength": over_strength,
                }

        # Under signal strength
        if under_pct > 0.5:
            under_strength = round((under_pct - 0.5) * 2, 3)
            if best_under is None or under_strength > best_under["signal_strength"]:
                best_under = {
                    "engine": engine_label,
                    "contract_type": "DIGITUNDER",
                    "barrier": str(barrier + 1),
                    "signal": f"{under_count}/{total} digits < {barrier + 1} ({under_pct:.0%})",
                    "signal_strength": under_strength,
                }

    if best_over:
        signals.append(best_over)
    if best_under:
        signals.append(best_under)

    return signals

# ---------------------------------------------------------------------------
# Engine 1 — Tapped Price Analysis
# ---------------------------------------------------------------------------
def run_engine1(tapped_digits: list) -> list:
    """
    Engine 1 — Tapped Price Analysis
    Analyses digit frequency in tapped prices and generates signals.
    Returns list of signal dicts. Returns empty list if insufficient data.
    """
    if len(tapped_digits) < 3:
        return []

    signals = []
    freq = Counter(tapped_digits)
    total = len(tapped_digits)

    # Signal 1: Most frequent digit in tapped -> MATCH (dominated, may repeat)
    most_digit, most_count = freq.most_common(1)[0]
    most_strength = round(most_count / total, 3)
    signals.append({
        "engine": "E1_TAPPED",
        "contract_type": "DIGITMATCH",
        "barrier": str(most_digit),
        "signal": f"digit {most_digit} appears {most_count}x in tapped (most frequent)",
        "signal_strength": most_strength,
    })

    # Signal 2: Digits that appeared 0 times in tapped -> DIFFER
    for d in range(10):
        if freq.get(d, 0) == 0:
            signals.append({
                "engine": "E1_TAPPED",
                "contract_type": "DIGITDIFF",
                "barrier": str(d),
                "signal": f"digit {d} absent from tapped (0 appearances)",
                "signal_strength": round(1.0 - (0 / total), 3),
            })

    # Signal 3: Even/Odd balance of tapped
    evens = sum(1 for d in tapped_digits if d % 2 == 0)
    odds = total - evens
    if evens != odds:
        dominant = "even" if evens > odds else "odd"
        eo_strength = round(abs(evens - odds) / total, 3)
        if eo_strength > 0.1:
            signals.append({
                "engine": "E1_TAPPED",
                "contract_type": "DIGITEVEN" if dominant == "even" else "DIGITODD",
                "barrier": None,
                "signal": f"tapped skews {dominant} ({evens}E / {odds}O)",
                "signal_strength": eo_strength,
            })

    # Signal 4: Over/Under — dynamic barrier selection
    signals.extend(get_over_under_signals(tapped_digits, "E1_TAPPED"))

    signals.sort(key=lambda x: x["signal_strength"], reverse=True)
    return signals

# ---------------------------------------------------------------------------
# Engine 3 — Full Range Analysis
# ---------------------------------------------------------------------------
def run_engine3(full_range_digits: list) -> list:
    """
    Engine 3 — Full Range Analysis
    Analyses digit frequency across all possible prices in the range.
    Returns list of signal dicts. Returns empty list if insufficient data.
    """
    if len(full_range_digits) < 3:
        return []

    signals = []
    freq = Counter(full_range_digits)
    total = len(full_range_digits)

    # Signal 1: Most frequent digit across full range -> MATCH
    most_digit, most_count = freq.most_common(1)[0]
    most_strength = round(most_count / total, 3)
    signals.append({
        "engine": "E3_FULL",
        "contract_type": "DIGITMATCH",
        "barrier": str(most_digit),
        "signal": f"digit {most_digit} appears {most_count}x in full range (most frequent)",
        "signal_strength": most_strength,
    })

    # Signal 2: Least frequent digit across full range -> DIFFER
    least_digit, least_count = freq.most_common()[-1]
    least_strength = round(1.0 - (least_count / total), 3)
    if least_strength > 0.5:
        signals.append({
            "engine": "E3_FULL",
            "contract_type": "DIGITDIFF",
            "barrier": str(least_digit),
            "signal": f"digit {least_digit} appears {least_count}x in full range (least frequent)",
            "signal_strength": least_strength,
        })

    # Signal 3: Even/Odd balance
    evens = sum(1 for d in full_range_digits if d % 2 == 0)
    odds = total - evens
    if evens != odds:
        dominant = "even" if evens > odds else "odd"
        eo_strength = round(abs(evens - odds) / total, 3)
        if eo_strength > 0.1:
            signals.append({
                "engine": "E3_FULL",
                "contract_type": "DIGITEVEN" if dominant == "even" else "DIGITODD",
                "barrier": None,
                "signal": f"full range skews {dominant} ({evens}E / {odds}O)",
                "signal_strength": eo_strength,
            })

    # Signal 4: Over/Under — dynamic barrier selection
    signals.extend(get_over_under_signals(full_range_digits, "E3_FULL"))

    signals.sort(key=lambda x: x["signal_strength"], reverse=True)
    return signals


# ---------------------------------------------------------------------------
# Engine 2 — Untapped Price Analysis
# ---------------------------------------------------------------------------
def run_engine2(untapped_digits: list) -> list:
    """
    Engine 2 — Untapped Price Analysis
    Analyses digit frequency in untapped prices and generates signals.
    Returns list of signal dicts. Returns empty list if insufficient data.
    """
    if len(untapped_digits) < 5:
        return []

    signals = []
    freq = Counter(untapped_digits)
    total = len(untapped_digits)

    # Signal 1: Most frequent digit in untapped -> MATCH (price will fill these gaps)
    most_digit, most_count = freq.most_common(1)[0]
    most_strength = round(most_count / total, 3)
    signals.append({
        "engine": "E2_UNTAPPED",
        "contract_type": "DIGITMATCH",
        "barrier": str(most_digit),
        "signal": f"digit {most_digit} appears {most_count}x in untapped (most frequent — expect fill)",
        "signal_strength": most_strength,
    })

    # Signal 2: Least frequent digit in untapped -> DIFFER (rarely in gaps, unlikely to appear)
    least_digit, least_count = freq.most_common()[-1]
    least_strength = round(1.0 - (least_count / total), 3)
    if least_strength > 0.5:
        signals.append({
            "engine": "E2_UNTAPPED",
            "contract_type": "DIGITDIFF",
            "barrier": str(least_digit),
            "signal": f"digit {least_digit} appears {least_count}x in untapped (least frequent — unlikely to fill)",
            "signal_strength": least_strength,
        })

    # Signal 3: Even/Odd balance
    evens = sum(1 for d in untapped_digits if d % 2 == 0)
    odds = total - evens
    if evens != odds:
        dominant = "even" if evens > odds else "odd"
        eo_strength = round(abs(evens - odds) / total, 3)
        if eo_strength > 0.1:
            signals.append({
                "engine": "E2_UNTAPPED",
                "contract_type": "DIGITEVEN" if dominant == "even" else "DIGITODD",
                "barrier": None,
                "signal": f"untapped skews {dominant} ({evens}E / {odds}O)",
                "signal_strength": eo_strength,
            })

    # Signal 4: Over/Under — dynamic barrier selection
    signals.extend(get_over_under_signals(untapped_digits, "E2_UNTAPPED"))

    signals.sort(key=lambda x: x["signal_strength"], reverse=True)
    return signals

# ---------------------------------------------------------------------------
# Cluster candle builder
# ---------------------------------------------------------------------------
def build_cluster_candle(minis: list, cluster_id: int) -> dict:
    all_ticks = []
    for m in minis:
        all_ticks.extend(m["ticks"])

    prices = [t["quote"] for t in all_ticks]
    epochs = [t["epoch"] for t in all_ticks]
    high = max(prices)
    low = min(prices)
    range_val = round(high - low, DECIMAL_PLACES)

    # Combined tapped / untapped
    combined_tapped = set()
    combined_untapped = set()
    for m in minis:
        combined_tapped.update(m["tapped_prices"])
        combined_untapped.update(m["untapped_prices"])

    # Digit frequency across all ticks in cluster
    all_digits = [extract_digit(p) for p in prices]
    digit_freq = Counter(all_digits)
    most_digit = digit_freq.most_common(1)[0][0] if digit_freq else 0
    least_digit = digit_freq.most_common()[-1][0] if digit_freq else 0

    # Prepare mini summaries (strip raw ticks to keep payload smaller)
    mini_summaries = []
    for m in minis:
        summary = {k: v for k, v in m.items() if k != "ticks"}
        mini_summaries.append(summary)

    # Collect all engine signals from minis
    all_e1_signals = []
    all_e2_signals = []
    all_e3_signals = []
    for m in minis:
        all_e1_signals.extend(m.get("e1_signals", []))
        all_e2_signals.extend(m.get("e2_signals", []))
        all_e3_signals.extend(m.get("e3_signals", []))

    return {
        "cluster_id": cluster_id,
        "epoch_start": min(epochs),
        "epoch_end": max(epochs),
        "high": high,
        "low": low,
        "range": range_val,
        "tick_count": len(all_ticks),
        "tapped_count": len(combined_tapped),
        "untapped_count": len(combined_untapped),
        "most_digit": most_digit,
        "least_digit": least_digit,
        "digit_freq": {d: digit_freq.get(d, 0) for d in range(10)},
        "minis": mini_summaries,
        "e1_signals": all_e1_signals,
        "e2_signals": all_e2_signals,
        "e3_signals": all_e3_signals,
    }

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
state = {
    "connected": False,
    "current_price": None,
    "current_digit": None,
    "current_cluster_id": 0,
    "current_mini_id": 0,
    "ticks_in_current_mini": [],
    "ticks_in_current_cluster": 0,
    "completed_minis_in_cluster": [],
    "clusters": [],
    "active_cluster": None,
    "session_ticks": 0,
    "synced": False,
    "sync_waiting": True,
    "last_epoch": None,
    "cluster_epoch_start": None,
    "retest_events": [],
    "active_retests": [],
    "retest_total": 0,
    "execution_enabled": True,
    "trades": [],
    "trade_stats": {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0},
    "traded_minis": {},  # key: "C{cid}M{mid}" -> trade count
    "engine_pnl": {
        "E1_TAPPED":     {"total_profit": 0.0, "total_loss": 0.0, "net_pnl": 0.0},
        "E2_UNTAPPED":   {"total_profit": 0.0, "total_loss": 0.0, "net_pnl": 0.0},
        "E3_FULL_RANGE": {"total_profit": 0.0, "total_loss": 0.0, "net_pnl": 0.0},
    },
    "active_trade_count": 0,
}


def save_trades_to_disk():
    """Persist trades and stats to disk so they survive restarts."""
    try:
        with open(TRADES_FILE, "w") as f:
            json.dump({"trades": state["trades"], "trade_stats": state["trade_stats"]}, f)
    except Exception as e:
        print(f"[SAVE] Error saving trades: {e}")


def load_trades_from_disk():
    """Load trades from disk on startup if file exists."""
    import os
    if not os.path.exists(TRADES_FILE):
        return
    try:
        with open(TRADES_FILE, "r") as f:
            data = json.load(f)
        state["trades"] = data.get("trades", [])
        state["trade_stats"] = data.get("trade_stats", {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0})
        print(f"[LOAD] Restored {len(state['trades'])} trades from {TRADES_FILE}")
    except Exception as e:
        print(f"[LOAD] Error loading trades: {e}")

# Global reference to Deriv WS for trade execution
deriv_ws_ref = None
buy_response_queue: asyncio.Queue = None  # initialized in lifespan

# Browser WebSocket clients
ui_clients: list[WebSocket] = []

# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------
async def execute_engine_trade(engine_id: str, signal: dict, stake: float,
                                probability: float, cluster_id: int, mini_id: int,
                                price: float, digit: int, epoch: int):
    """Place a trade via the Deriv WS connection. Non-blocking."""
    global deriv_ws_ref
    if deriv_ws_ref is None:
        print("[EXEC] No Deriv WS connection available")
        return

    ct = signal["contract_type"]
    barrier = signal.get("barrier")
    signal_strength = signal["signal_strength"]

    # Determine P&L state
    net = state["engine_pnl"][engine_id]["net_pnl"]
    pnl_state = "positive" if net > 0 else ("negative" if net < 0 else "neutral")

    # Build buy request
    buy_req = {
        "buy": 1,
        "price": stake,
        "parameters": {
            "contract_type": ct,
            "symbol": SYMBOL,
            "currency": "USD",
            "amount": stake,
            "basis": "stake",
            "duration": 1,
            "duration_unit": "t",
        }
    }
    if barrier is not None:
        buy_req["parameters"]["barrier"] = str(barrier)

    trade_record = {
        "id": len(state["trades"]) + 1,
        "epoch": epoch,
        "cluster_id": cluster_id,
        "mini_id": mini_id,
        "engine": engine_id,
        "contract_type": ct,
        "barrier": barrier,
        "stake": stake,
        "probability": probability,
        "pnl_state": pnl_state,
        "signal_strength": signal_strength,
        "entry_price": price,
        "entry_digit": digit,
        "contract_id": None,
        "outcome": "pending",
        "profit": 0.0,
        "buy_price": None,
        "payout": None,
    }
    state["trades"].append(trade_record)
    state["trade_stats"]["total"] += 1

    try:
        # Send buy — response will be routed via the main message loop
        await deriv_ws_ref.send(json.dumps(buy_req))
        print(f"[EXEC] Buy sent: {ct} b:{barrier}")

        # Wait for buy response on the queue (routed by deriv_feed)
        data = await asyncio.wait_for(buy_response_queue.get(), timeout=10)

        if "error" in data:
            print(f"[EXEC] Buy error: {data['error']['message']}")
            trade_record["outcome"] = "error"
            trade_record["profit"] = 0.0
            return

        if "buy" in data:
            buy = data["buy"]
            trade_record["contract_id"] = buy.get("contract_id")
            trade_record["buy_price"] = buy.get("buy_price")
            print(f"[EXEC] Trade placed: {ct} b:{barrier} contract:{trade_record['contract_id']}")

            # Subscribe to settlement — just send, response handled by main loop
            if trade_record["contract_id"]:
                sub_req = {
                    "proposal_open_contract": 1,
                    "contract_id": trade_record["contract_id"],
                    "subscribe": 1,
                }
                await deriv_ws_ref.send(json.dumps(sub_req))

    except asyncio.TimeoutError:
        print("[EXEC] Buy request timed out")
        trade_record["outcome"] = "timeout"
    except Exception as e:
        print(f"[EXEC] Trade error: {e}")
        trade_record["outcome"] = "error"


def handle_proposal_open_contract(data: dict):
    """Process settlement updates from proposal_open_contract subscription."""
    poc = data.get("proposal_open_contract", {})
    contract_id = poc.get("contract_id")
    if not contract_id:
        return

    is_settled = poc.get("is_expired", False) or poc.get("is_sold", False)
    if not is_settled:
        return

    # Find the matching trade
    for trade in state["trades"]:
        if trade["contract_id"] == contract_id and trade["outcome"] == "pending":
            profit = float(poc.get("profit", 0))
            trade["profit"] = profit
            payout = poc.get("payout")
            trade["payout"] = payout
            buy_price = float(trade.get("buy_price") or trade["stake"])
            trade["payout_pct"] = round(((float(payout) / buy_price) - 1) * 100, 1) if payout and buy_price else 0
            if profit > 0:
                trade["outcome"] = "win"
                state["trade_stats"]["wins"] += 1
            else:
                trade["outcome"] = "loss"
                state["trade_stats"]["losses"] += 1
            state["trade_stats"]["pnl"] = round(state["trade_stats"]["pnl"] + profit, 2)
            
            # Update per-engine P&L
            engine_id = trade.get("engine")
            if engine_id and engine_id in state["engine_pnl"]:
                ep = state["engine_pnl"][engine_id]
                if profit > 0:
                    ep["total_profit"] = round(ep["total_profit"] + profit, 2)
                else:
                    ep["total_loss"] = round(ep["total_loss"] + abs(profit), 2)
                ep["net_pnl"] = round(ep["total_profit"] - ep["total_loss"], 2)
            
            print(f"[EXEC] Settled: {engine_id} {trade['contract_type']} b:{trade['barrier']} -> {trade['outcome']} ${profit:.2f}")
            save_trades_to_disk()
            break



# ---------------------------------------------------------------------------
# Retest Monitor
# ---------------------------------------------------------------------------
def get_all_sealed_minis() -> list:
    """Collect all sealed minis from last 30 clusters + current cluster."""
    sealed = []
    # Only check last 30 clusters (30 minutes of history)
    recent_clusters = state["clusters"][-30:] if len(state["clusters"]) > 30 else state["clusters"]
    for cl in recent_clusters:
        for m in cl["minis"]:
            sealed.append({
                "cluster_id": cl["cluster_id"],
                "mini_id": m["mini_id"],
                "high": m["high"],
                "low": m["low"],
                "e1_signals": m.get("e1_signals", []),
                "e2_signals": m.get("e2_signals", []),
                "e3_signals": m.get("e3_signals", []),
            })
    for m in state["completed_minis_in_cluster"]:
        sealed.append({
            "cluster_id": state["current_cluster_id"],
            "mini_id": m["mini_id"],
            "high": m["high"],
            "low": m["low"],
            "e1_signals": m.get("e1_signals", []),
            "e2_signals": m.get("e2_signals", []),
            "e3_signals": m.get("e3_signals", []),
        })
    return sealed


def check_retests(epoch: int, quote: float, digit: int):
    """Check current tick against all sealed mini candle ranges."""
    sealed = get_all_sealed_minis()
    active = []

    for sm in sealed:
        if sm["low"] <= quote <= sm["high"]:
            # Price is inside this sealed mini's range
            active.append({
                "cluster_id": sm["cluster_id"],
                "mini_id": sm["mini_id"],
                "low": sm["low"],
                "high": sm["high"],
            })

            # Check if we already logged a retest for this mini in the last tick
            # (avoid spamming — only log on first entry or after exit)
            already_active = any(
                r["cluster_id"] == sm["cluster_id"] and r["mini_id"] == sm["mini_id"]
                for r in state["active_retests"]
            )
            if not already_active:
                # New retest entry — log it
                event = {
                    "epoch": epoch,
                    "cluster_id": sm["cluster_id"],
                    "mini_id": sm["mini_id"],
                    "price": quote,
                    "digit": digit,
                    "range_low": sm["low"],
                    "range_high": sm["high"],
                }
                state["retest_events"].append(event)
                state["retest_total"] += 1
                
                # Trim retest_events to last 500 entries
                if len(state["retest_events"]) > 500:
                    state["retest_events"] = state["retest_events"][-500:]

                # Fire each engine independently — no convergence, no merging
                if not state["execution_enabled"]:
                    continue
                
                for engine_id, signals in [
                    ("E1_TAPPED",     sm.get("e1_signals", [])),
                    ("E2_UNTAPPED",   sm.get("e2_signals", [])),
                    ("E3_FULL_RANGE", sm.get("e3_signals", [])),
                ]:
                    if not signals:
                        continue
                    
                    # Check trade limit for this mini per engine
                    mini_key = f"C{sm['cluster_id']}M{sm['mini_id']}_{engine_id}"
                    if state["traded_minis"].get(mini_key, 0) >= MAX_TRADES_PER_RETEST:
                        continue
                    
                    # Take the strongest signal from this engine
                    best = max(signals, key=lambda s: s["signal_strength"])
                    prob  = calculate_probability(best["contract_type"], best.get("barrier"))
                    stake = calculate_stake(best["contract_type"], best.get("barrier"), engine_id)
                    stake = get_adjusted_stake(stake)
                    
                    # Track trades per mini per engine
                    state["traded_minis"][mini_key] = state["traded_minis"].get(mini_key, 0) + 1
                    
                    asyncio.ensure_future(
                        execute_engine_trade(engine_id, best, stake, prob, sm["cluster_id"], sm["mini_id"], quote, digit, epoch)
                    )

    state["active_retests"] = active


# ---------------------------------------------------------------------------
# Tick processing
# ---------------------------------------------------------------------------
def process_tick(epoch: int, quote: float):
    state["current_price"] = quote
    state["current_digit"] = extract_digit(quote)
    state["last_epoch"] = epoch
    state["session_ticks"] += 1

    # Waiting for sync?
    if state["sync_waiting"]:
        if epoch % 60 == 0:
            state["synced"] = True
            state["sync_waiting"] = False
            state["current_cluster_id"] = 1
            state["current_mini_id"] = 1
            state["cluster_epoch_start"] = epoch
        else:
            return  # discard tick, still waiting for boundary

    # Add tick
    tick = {"epoch": epoch, "quote": quote, "digit": extract_digit(quote)}
    state["ticks_in_current_mini"].append(tick)
    state["ticks_in_current_cluster"] += 1

    # Check if mini candle is sealed (10 ticks)
    if len(state["ticks_in_current_mini"]) == TICKS_PER_MINI:
        mini = build_mini_candle(state["ticks_in_current_mini"], state["current_mini_id"])
        state["completed_minis_in_cluster"].append(mini)
        state["ticks_in_current_mini"] = []

        # Check if cluster is sealed (6 minis)
        if len(state["completed_minis_in_cluster"]) == MINIS_PER_CLUSTER:
            cluster = build_cluster_candle(state["completed_minis_in_cluster"], state["current_cluster_id"])
            state["clusters"].append(cluster)
            state["completed_minis_in_cluster"] = []
            state["ticks_in_current_cluster"] = 0
            state["current_cluster_id"] += 1
            state["current_mini_id"] = 1
            state["cluster_epoch_start"] = None
            
            # Clean up traded_minis dict - only keep last 30 clusters
            # Keys are now "C{cid}M{mid}_{engine_id}"
            current_cid = state["current_cluster_id"]
            state["traded_minis"] = {
                k: v for k, v in state["traded_minis"].items()
                if int(k.split('C')[1].split('M')[0].split('_')[0]) >= current_cid - 30
            }
        else:
            state["current_mini_id"] += 1

    # Retest monitor — check every synced tick against sealed minis
    if state["synced"]:
        check_retests(epoch, quote, extract_digit(quote))


def get_active_cluster_snapshot() -> Optional[dict]:
    """Build a snapshot of the cluster currently being assembled."""
    minis = state["completed_minis_in_cluster"]
    current_ticks = state["ticks_in_current_mini"]
    if not minis and not current_ticks:
        return None

    # Build partial mini from current ticks
    partial_mini = None
    if current_ticks:
        prices = [t["quote"] for t in current_ticks]
        partial_mini = {
            "mini_id": state["current_mini_id"],
            "tick_count": len(current_ticks),
            "ticks": current_ticks,
            "high": max(prices),
            "low": min(prices),
        }

    # Summaries for completed minis
    mini_summaries = []
    for m in minis:
        summary = {k: v for k, v in m.items() if k != "ticks"}
        mini_summaries.append(summary)

    return {
        "cluster_id": state["current_cluster_id"],
        "minis_completed": len(minis),
        "minis": mini_summaries,
        "partial_mini": partial_mini,
        "ticks_in_cluster": state["ticks_in_current_cluster"],
        "epoch_start": state["cluster_epoch_start"],
    }


def get_ui_state() -> dict:
    return {
        "connected": state["connected"],
        "synced": state["synced"],
        "sync_waiting": state["sync_waiting"],
        "current_price": state["current_price"],
        "current_digit": state["current_digit"],
        "current_cluster_id": state["current_cluster_id"],
        "current_mini_id": state["current_mini_id"],
        "ticks_in_mini": len(state["ticks_in_current_mini"]),
        "session_ticks": state["session_ticks"],
        "last_epoch": state["last_epoch"],
        "clusters": state["clusters"],
        "active_cluster": get_active_cluster_snapshot(),
        "retest_events": state["retest_events"][-10:],
        "active_retests": state["active_retests"],
        "retest_total": state["retest_total"],
        "execution_enabled": state["execution_enabled"],
        "trades": state["trades"][-20:],
        "trade_stats": state["trade_stats"],
        "engine_pnl": state["engine_pnl"],
    }

# ---------------------------------------------------------------------------
# Deriv WebSocket feed
# ---------------------------------------------------------------------------
async def deriv_feed():
    global deriv_ws_ref
    print(f"[DERIV_FEED] Starting... URL={DERIV_WS_URL}")
    while True:
        try:
            print(f"[DERIV_FEED] Connecting to Deriv WebSocket...")
            # Add ping_interval and ping_timeout for keepalive during laptop sleep
            async with websockets.connect(
                DERIV_WS_URL,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=60,   # Wait up to 60 seconds for pong
                close_timeout=10
            ) as ws:
                state["connected"] = True
                deriv_ws_ref = ws
                print(f"[DERIV_FEED] Connected!")

                # Authorize
                print(f"[DERIV_FEED] Authorizing with token...")
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                auth_resp = await ws.recv()
                auth_data = json.loads(auth_resp)
                if "error" in auth_data:
                    print(f"[DERIV_FEED] Auth error: {auth_data['error']['message']}")
                    state["connected"] = False
                    deriv_ws_ref = None
                    await asyncio.sleep(5)
                    continue

                print(f"[DERIV_FEED] Auth successful! Subscribing to {SYMBOL}...")
                # Subscribe to ticks
                await ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

                last_tick_time = asyncio.get_event_loop().time()
                async for message in ws:
                    last_tick_time = asyncio.get_event_loop().time()  # Update on every message
                    data = json.loads(message)
                    if "tick" in data:
                        tick = data["tick"]
                        epoch = tick["epoch"]
                        quote = tick["quote"]

                        # Track cluster start epoch
                        if state["synced"] and state["cluster_epoch_start"] is None:
                            state["cluster_epoch_start"] = epoch

                        process_tick(epoch, quote)

                    # Route buy responses to the execution queue
                    elif "buy" in data or ("error" in data and data.get("msg_type") == "buy"):
                        await buy_response_queue.put(data)

                    # Handle trade settlement updates
                    elif "proposal_open_contract" in data:
                        handle_proposal_open_contract(data)

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"[DERIV_FEED] Connection closed: {e}")
            state["connected"] = False
            deriv_ws_ref = None
            # If reconnected mid-candle, discard partial
            if state["synced"]:
                state["ticks_in_current_mini"] = []
                state["sync_waiting"] = True
                state["synced"] = False
                state["completed_minis_in_cluster"] = []
                state["ticks_in_current_cluster"] = 0
            print(f"[DERIV_FEED] Retrying in 3 seconds...")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"[DERIV_FEED] Unexpected error: {e}")
            state["connected"] = False
            deriv_ws_ref = None
            if state["synced"]:
                state["ticks_in_current_mini"] = []
                state["sync_waiting"] = True
                state["synced"] = False
                state["completed_minis_in_cluster"] = []
                state["ticks_in_current_cluster"] = 0
            print(f"[DERIV_FEED] Retrying in 5 seconds...")
            await asyncio.sleep(5)

# ---------------------------------------------------------------------------
# UI broadcast
# ---------------------------------------------------------------------------
async def broadcast_loop():
    while True:
        if ui_clients:
            payload = json.dumps(get_ui_state())
            disconnected = []
            for client in ui_clients:
                try:
                    await client.send_text(payload)
                except Exception:
                    disconnected.append(client)
            # Safe removal - only remove if still in list
            for c in disconnected:
                if c in ui_clients:
                    ui_clients.remove(c)
        await asyncio.sleep(1)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application):
    global buy_response_queue
    print("[APP] Starting lifespan event...")
    buy_response_queue = asyncio.Queue()
    # Start fresh each time - don't load previous trades
    print("[APP] Starting with clean slate (no trades loaded)")
    print("[APP] Creating deriv_feed task...")
    asyncio.create_task(deriv_feed())
    print("[APP] Creating broadcast_loop task...")
    asyncio.create_task(broadcast_loop())
    print("[APP] Startup complete!")
    yield
    print("[APP] Shutdown...")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint for Render and other platforms."""
    return {"status": "ok", "connected": state["connected"], "synced": state["synced"]}


@app.head("/health")
async def health_check_head():
    """Health check HEAD endpoint for Render."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/export/trades")
async def export_trades():
    cols = ["id", "epoch", "cluster_id", "mini_id", "engine", "contract_type", "barrier",
            "probability", "stake", "pnl_state", "signal_strength", "entry_price",
            "entry_digit", "outcome", "profit", "payout"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for t in state["trades"]:
        writer.writerow({c: t.get(c, "") for c in cols})
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trades.csv"},
    )


@app.post("/toggle_execution")
async def toggle_execution():
    state["execution_enabled"] = not state["execution_enabled"]
    status = "ENABLED" if state["execution_enabled"] else "DISABLED"
    print(f"[EXEC] Execution {status}")
    return {"execution_enabled": state["execution_enabled"]}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    ui_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in ui_clients:
            ui_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
