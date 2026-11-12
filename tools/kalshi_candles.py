"""Hourly bid/ask/volume candles for settled Kalshi NBA markets, from 14h before expected expiration to expiration.
Live (post-cutoff) events: one event-candlesticks call covers all markets; historical: one call per market.

  python -m tools.kalshi_candles --series KXNBAGAME KXNBAPTS ... [--minvol 1000] [--rate 8]
Writes work/kc_<SERIES>.jsonl (ticker -> candles), resumable.
"""
import argparse, json, threading, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.elections.kalshi.com/trade-api/v2"
S = requests.Session(); S.headers.update({"User-Agent": "nba-edge-research/1.0"})
S.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16))


class Pace:
    def __init__(self, rate):
        self.gap, self.t, self.lock = 1 / rate, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic(); t = max(now, self.t + self.gap); self.t = t
        time.sleep(max(0, t - now))


PACE = Pace(8)


def kget(path, params):
    for i in range(8):
        PACE.wait()
        try:
            r = S.get(API + path, params=params, timeout=40)
        except requests.RequestException:
            time.sleep(2 + 2 * i); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(1 + 2 * i); continue
        return {}
    return {}


def ts(s):
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def norm(c):
    def f(x, side):
        v = x.get(side) or {}
        v = v.get("close_dollars", v.get("close"))
        try:
            return float(v) if v not in (None, "") else None
        except ValueError:
            return None
    return [[x["end_period_ts"], f(x, "yes_bid"), f(x, "yes_ask"), round(float(x.get("volume_fp") or x.get("volume") or 0), 2)] for x in c]


def window(m):
    end = ts(m.get("expected_expiration_time") or m["close_time"])
    return end - 14 * 3600, end


def hist_one(series, m):
    a, b = window(m)
    d = kget(f"/historical/markets/{m['ticker']}/candlesticks", {"start_ts": a, "end_ts": b, "period_interval": 60})
    return [(m["ticker"], norm(d.get("candlesticks") or []))]


def live_event(series, et, ms):
    a = min(window(m)[0] for m in ms); b = max(window(m)[1] for m in ms)
    d = kget(f"/series/{series}/events/{et}/candlesticks", {"start_ts": a, "end_ts": b, "period_interval": 60})
    out = []
    for t, c in zip(d.get("market_tickers") or [], d.get("market_candlesticks") or []):
        out.append((t, norm(c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", required=True)
    ap.add_argument("--minvol", type=float, default=1000)
    ap.add_argument("--rate", type=float, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--live-only", action="store_true", help="only post-cutoff events (one call per event)")
    ap.add_argument("--sample", type=float, default=1.0, help="random share of historical games (by game, outcome-blind)")
    a = ap.parse_args()
    PACE.gap = 1 / a.rate
    for s in a.series:
        ms = [json.loads(l) for l in open(ROOT / "work" / f"kmk_{s}.jsonl")]
        ms = [m for m in ms if m.get("result") in ("yes", "no") and float(m.get("volume_fp") or m.get("volume") or 0) >= a.minvol]
        out = ROOT / "work" / f"kc_{s}.jsonl"
        done = set()
        if out.exists():
            done = {json.loads(l)[0] for l in out.open()}
        ms = [m for m in ms if m["ticker"] not in done]
        live = {}
        for m in ms:
            if not m.get("hist"):
                live.setdefault(m["event_ticker"], []).append(m)
        import zlib
        def keep(et):   # stable pseudo-random by game (date+teams), independent of results
            return (zlib.crc32(et.split("-", 1)[1].encode()) % 1000) / 1000 < a.sample
        hist = [] if a.live_only else [m for m in ms if m.get("hist") and keep(m["event_ticker"])]
        print(s, "hist markets", len(hist), "live events", len(live), flush=True)
        jobs = [(hist_one, (s, m)) for m in hist] + [(live_event, (s, et, v)) for et, v in live.items()]
        n = 0
        with out.open("a") as fh, ThreadPoolExecutor(a.workers) as ex:
            for res in ex.map(lambda j: j[0](*j[1]), jobs):
                for t, c in res:
                    fh.write(json.dumps([t, c], separators=(",", ":")) + "\n")
                n += 1
                if n % 1000 == 0:
                    print(s, n, "/", len(jobs), time.strftime("%H:%M:%S"), flush=True); fh.flush()
        print(s, "done", n, flush=True)


if __name__ == "__main__":
    main()
