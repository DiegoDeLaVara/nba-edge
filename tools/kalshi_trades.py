"""Pre-game trade tapes for this season's Kalshi NBA prop markets (who took, which side, price, size).
-> work/kt_trades.jsonl: [ticker, [[ts, taker_side, yes_price, count], ...]] for trades before tip. Resumable."""
import sys, json, time, threading; sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests

API = "https://api.elections.kalshi.com/trade-api/v2"
S = requests.Session(); S.headers.update({"User-Agent": "nba-edge-research/1.0"})
S.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16))
OUT = "work/kt_trades.jsonl"
lock = threading.Lock(); nxt = [0.0]


def pace(rate):
    with lock:
        now = time.monotonic(); t = max(now, nxt[0] + 1 / rate); nxt[0] = t
    time.sleep(max(0, t - now))


def ts(s):
    return pd.Timestamp(s).timestamp()


def one(ticker, tip, rate):
    out, cur = [], ""
    for _ in range(20):
        p = {"ticker": ticker, "limit": 1000, "max_ts": tip}
        if cur:
            p["cursor"] = cur
        d = None
        for i in range(6):
            pace(rate)
            try:
                r = S.get(API + "/markets/trades", params=p, timeout=30)
                if r.status_code == 200:
                    d = r.json(); break
                if r.status_code == 429:
                    time.sleep(1 + 2 * i); continue
                break
            except requests.RequestException:
                time.sleep(2 + 2 * i)
        if not d:
            break
        for t in d.get("trades") or []:
            out.append([round(ts(t["created_time"])), t.get("taker_side"), float(t.get("yes_price_dollars") or 0), float(t.get("count_fp") or 0)])
        cur = d.get("cursor") or ""
        if not cur or not d.get("trades"):
            break
    return [ticker, out]


if __name__ == "__main__":
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = []
    for st in ("fg3m", "pts", "reb", "ast"):
        R = pd.read_pickle(f"work/pt_{st}.pkl")
        R = R[pd.to_datetime(R.tip, unit="s") >= "2026-10-01"]
        rows += list(zip(R.ticker, R.tip.astype(int)))
    done = set()
    try:
        done = {json.loads(l)[0] for l in open(OUT)}
    except FileNotFoundError:
        pass
    jobs = [(t, tp) for t, tp in rows if t not in done]
    print("markets", len(jobs), flush=True)
    with open(OUT, "a") as fh, ThreadPoolExecutor(10) as ex:
        for i, r in enumerate(ex.map(lambda j: one(j[0], j[1], rate), jobs)):
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            if i % 2000 == 0:
                print(i, time.strftime("%H:%M:%S"), flush=True); fh.flush()
    print("done", flush=True)
