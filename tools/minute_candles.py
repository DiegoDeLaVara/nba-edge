"""1-minute Kalshi candles for this season's NBA events, from HOURS before tip to tip (event candlesticks are capped at
~5,000 market-minutes per call, so page forward with adjusted_end_ts).
usage: minute_candles.py HOURS SERIES...   -> work/mc_<SERIES>.jsonl: [ticker, [[end_ts, bid, ask, vol], ...]]"""
import sys, json; sys.path.insert(0, ".")
import pandas as pd
from tools.kalshi_candles import kget, norm, PACE
from edge.kalshi import parse_event, game_index

G = pd.read_pickle("work/G.pkl"); tips = dict(zip(G.gid, G.tip)); gi = game_index(G)
PACE.gap = 1 / 8
hours = float(sys.argv[1])
for ser in sys.argv[2:]:
    evs = {}
    for l in open(f"work/kmk_{ser}.jsonl"):
        m = json.loads(l)
        if not m.get("hist"):
            evs[m["event_ticker"]] = 1
    out = open(f"work/mc_{ser}.jsonl", "w"); n = 0
    for et in evs:
        pe = parse_event(et); gid = gi.get(pe) if pe else None
        if gid is None:
            continue
        tip = int(tips[gid].timestamp()); a = int(tip - hours * 3600)
        acc = {}
        while a < tip:
            d = kget(f"/series/{ser}/events/{et}/candlesticks", {"start_ts": a, "end_ts": tip, "period_interval": 1})
            for t, c in zip(d.get("market_tickers") or [], d.get("market_candlesticks") or []):
                acc.setdefault(t, []).extend(norm(c))
            e = d.get("adjusted_end_ts")
            if not e or e <= a:
                break
            a = e
        for t, c in acc.items():
            out.write(json.dumps([t, c], separators=(",", ":")) + "\n")
        n += 1
        if n % 20 == 0:
            print(ser, n, len(evs), flush=True); out.flush()
    out.close(); print(ser, "events", n, "done", flush=True)
