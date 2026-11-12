"""Hourly candles from market OPEN to tip for this season's Kalshi NBA prop events (one event-candlesticks call per
event). -> work/oc_<SERIES>.jsonl: [ticker, open_ts, [[end_ts, bid, ask, vol], ...]]"""
import sys, json, time; sys.path.insert(0, ".")
import pandas as pd
from tools.kalshi_candles import kget, norm, PACE

G = pd.read_pickle("work/G.pkl"); tips = dict(zip(G.gid, G.tip))
from edge.kalshi import parse_event, game_index
gi = game_index(G)
PACE.gap = 1 / 8
for ser in sys.argv[1:]:
    ms = [json.loads(l) for l in open(f"work/kmk_{ser}.jsonl")]
    ev = {}
    for m in ms:
        if not m.get("hist") and m.get("result") in ("yes", "no"):
            ev.setdefault(m["event_ticker"], []).append(m)
    out = open(f"work/oc_{ser}.jsonl", "w"); n = 0
    for et, mm in ev.items():
        pe = parse_event(et); gid = gi.get(pe) if pe else None
        if gid is None:
            continue
        tip = int(tips[gid].timestamp())
        op = {m["ticker"]: int(pd.Timestamp(m["open_time"]).timestamp()) for m in mm}
        start = min(op.values()) - 3600
        d = kget(f"/series/{ser}/events/{et}/candlesticks", {"start_ts": start, "end_ts": tip, "period_interval": 60})
        for t, c in zip(d.get("market_tickers") or [], d.get("market_candlesticks") or []):
            out.write(json.dumps([t, op.get(t), norm(c)], separators=(",", ":")) + "\n")
        n += 1
    out.close(); print(ser, "events", n, flush=True)
