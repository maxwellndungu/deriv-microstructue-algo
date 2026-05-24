"""
Collect a long, contiguous tick stream from Deriv V100 (1s) for statistical analysis.

We pull `history` (up to 5000 ticks per call) repeatedly walking the epoch
backwards so we end up with a long contiguous window. No auth required for
`ticks_history`.
"""
import asyncio
import json
import sys
import websockets
import time

URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"  # public app_id from Deriv docs
SYMBOL = "1HZ100V"  # Volatility 100 (1s) Index

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
OUT = sys.argv[2] if len(sys.argv) > 2 else "ticks_v100_1s.json"


async def fetch_history(ws, end_epoch=None, count=5000):
    req = {
        "ticks_history": SYMBOL,
        "style": "ticks",
        "count": count,
        "end": "latest" if end_epoch is None else str(end_epoch),
        "adjust_start_time": 1,
    }
    await ws.send(json.dumps(req))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("msg_type") == "history":
            return msg["history"]
        if "error" in msg:
            raise RuntimeError(msg["error"]["message"])


async def main():
    all_ticks = []  # list of (epoch, price_string)
    async with websockets.connect(URL, ping_interval=20, ping_timeout=60) as ws:
        end = None
        while len(all_ticks) < TARGET:
            hist = await fetch_history(ws, end_epoch=end, count=5000)
            times = hist["times"]
            prices = hist["prices"]
            chunk = list(zip(times, prices))
            if not chunk:
                break
            # newest first as we walk back: prepend earlier-fetched chunks
            new_part = [t for t in chunk if not all_ticks or t[0] < all_ticks[0][0]]
            if not new_part:
                break
            all_ticks = new_part + all_ticks
            print(f"collected {len(all_ticks)} ticks, oldest epoch {all_ticks[0][0]}", flush=True)
            end = all_ticks[0][0] - 1
            await asyncio.sleep(0.4)  # be polite to the API

    # Sort by epoch ascending
    all_ticks.sort(key=lambda x: x[0])
    with open(OUT, "w") as f:
        json.dump(all_ticks, f)
    span_s = all_ticks[-1][0] - all_ticks[0][0]
    print(f"saved {len(all_ticks)} ticks to {OUT}, span {span_s}s = {span_s/3600:.2f}h")


if __name__ == "__main__":
    asyncio.run(main())
