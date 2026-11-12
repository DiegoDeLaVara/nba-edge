"""Pre-game market selection for the historical refill: keep prop lines our walk-forward projection prices 4-96%."""
import sys, json; sys.path.insert(0, ".")
import pandas as pd
from edge.kalshi import load_markets, parse_event, game_index, norm_name
from edge.dist import Dist, GRID
G = pd.read_pickle("work/G.pkl"); P = pd.read_pickle("work/P.pkl")
gi = game_index(G); pl = P[P.played].copy(); pl["nn"] = pl.name.map(norm_name)
keep = {}
for st, ser in (("reb", "KXNBAREB"), ("ast", "KXNBAAST"), ("fg3m", "KXNBA3PT")):
    W = pd.read_pickle(f"work/wf_{st}.pkl"); pre = W[W.tip < "2025-11-01"]
    dist = Dist().fit(pre.mu.values, pre[st].values.astype(float), GRID[st])
    mu = {(g, p): m for g, p, m in zip(W.gid, W.pid, W.mu)}
    nm = {(g, n): p for g, n, p in zip(pl.gid, pl.nn, pl.pid)}
    k_all = n = 0; ks = []
    for m in load_markets(ser):
        if not m.get("hist") or m.get("result") not in ("yes", "no"): continue
        k_all += 1
        pe = parse_event(m["event_ticker"]); gid = gi.get(pe) if pe else None
        name, _, k = (m.get("yes_sub_title") or "").rpartition(":")
        pid = nm.get((gid, norm_name(name)))
        if pid is None or (gid, pid) not in mu:
            continue
        p = float(dist.p_over([mu[(gid, pid)]], [float(k.strip().rstrip("+"))])[0])
        if 0.04 <= p <= 0.96:
            ks.append(m["ticker"])
    keep[ser] = ks
    print(ser, "hist settled", k_all, "kept", len(ks))
json.dump(keep, open("work/select.json", "w"))
