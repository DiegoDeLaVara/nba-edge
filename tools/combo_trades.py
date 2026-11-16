"""Pre-tip trade prices for a random, outcome-blind sample of all-NBA combos (combos_full.jsonl): the price actually
paid before any game started, instead of the last trade (which for losers is often a near-zero in-game print).
-> work/combo_trades.jsonl: ticker, vwap_pre (YES price), contracts_pre, n_trades_pre"""
import sys, json, time, zlib, threading; sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
from edge.kalshi import parse_event, game_index

API = "https://api.elections.kalshi.com/trade-api/v2"
S = requests.Session(); S.headers.update({"User-Agent": "nba-edge-research/1.0"})
S.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16))
G = pd.read_pickle("work/G.pkl"); gi = game_index(G); tip = dict(zip(G.gid, G.tip))
lock = threading.Lock(); nxt = [0.0]


def pace(rate=8):
    with lock:
        now = time.monotonic(); t = max(now, nxt[0] + 1 / rate); nxt[0] = t
    time.sleep(max(0, t - now))


def first_tip(legs):
    ts = []
    for t, _ in legs:
        pe = parse_event("-".join(t.split("-")[:2])); g = gi.get(pe) if pe else None
        if g is None:
            return None
        ts.append(tip[g].timestamp())
    return min(ts)


def one(m):
    ft = first_tip(m["legs"])
    if ft is None or pd.Timestamp(m["open"]).timestamp() >= ft - 300:
        return None
    for i in range(6):
        pace()
        try:
            r = S.get(API + "/markets/trades", params={"ticker": m["ticker"], "limit": 1000, "max_ts": int(ft)}, timeout=30)
            if r.status_code == 200:
                tr = r.json().get("trades") or []
                c = sum(float(t["count_fp"]) for t in tr)
                v = sum(float(t["count_fp"]) * float(t["yes_price_dollars"]) for t in tr)
                return {"ticker": m["ticker"], "vwap_pre": v / c if c else None, "contracts_pre": c, "n_pre": len(tr),
                        "taker_yes": sum(float(t["count_fp"]) for t in tr if t.get("taker_side") == "yes")}
            if r.status_code == 429:
                time.sleep(1 + 2 * i); continue
            return None
        except requests.RequestException:
            time.sleep(2 + 2 * i)
    return None


if __name__ == "__main__":
    share = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    done = set()
    try:
        done = {json.loads(l)["ticker"] for l in open("work/combo_trades.jsonl")}
    except FileNotFoundError:
        pass
    ms = [json.loads(l) for l in open("work/combos_full.jsonl")]
    ms = [m for m in ms if m["result"] in ("yes", "no") and m["ticker"] not in done and (zlib.crc32(m["ticker"].encode()) // 100) % 1000 < share * 1000]
    print("combos to fetch", len(ms), flush=True)
    with open("work/combo_trades.jsonl", "a") as fh, ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(one, ms)):
            if r:
                fh.write(json.dumps(r) + "\n")
            if i % 1000 == 0:
                print(i, time.strftime("%H:%M:%S"), flush=True); fh.flush()
    print("done", flush=True)
