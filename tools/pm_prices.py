"""Polymarket price just before tip (and 3h before) for every matched token, from the public CLOB price history.
-> work/pm_prices.jsonl (resumable): token, p_tip, t_tip, p_3h, n_points"""
import sys, json, time, threading; sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests

S = requests.Session(); S.headers.update({"User-Agent": "nba-edge-research/1.0"})
S.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16))
OUT = "work/pm_prices2.jsonl"
lock = threading.Lock(); gap = [0.0]


def pace(rate):
    with lock:
        now = time.monotonic(); t = max(now, gap[0] + 1 / rate); gap[0] = t
    time.sleep(max(0, t - now))


def one(tok, tip, rate):
    for i in range(6):
        pace(rate)
        try:
            r = S.get("https://clob.polymarket.com/prices-history", params={"market": tok, "startTs": tip - 8 * 3600, "endTs": tip + 600, "fidelity": 10}, timeout=30)
            if r.status_code == 200:
                h = r.json().get("history") or []
                pre = [x for x in h if x["t"] <= tip]
                pre3 = [x for x in h if x["t"] <= tip - 3 * 3600]
                return {"token": tok, "p_tip": pre[-1]["p"] if pre else None, "t_tip": pre[-1]["t"] if pre else None,
                        "p_3h": pre3[-1]["p"] if pre3 else None, "n": len(h), "h": [[x["t"], x["p"]] for x in pre]}
            if r.status_code == 429:
                time.sleep(2 + 2 * i); continue
            return {"token": tok, "err": r.status_code}
        except requests.RequestException:
            time.sleep(2 + 2 * i)
    return {"token": tok, "err": "fail"}


if __name__ == "__main__":
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    M = pd.read_pickle("work/pm_matched.pkl")[["token", "tip"]].drop_duplicates("token")
    done = set()
    try:
        done = {json.loads(l)["token"] for l in open(OUT)}
    except FileNotFoundError:
        pass
    jobs = [(t, int(tp)) for t, tp in zip(M.token, M.tip) if t not in done]
    print("tokens", len(jobs), flush=True)
    with open(OUT, "a") as fh, ThreadPoolExecutor(12) as ex:
        for i, r in enumerate(ex.map(lambda j: one(j[0], j[1], rate), jobs)):
            fh.write(json.dumps(r) + "\n")
            if i % 2000 == 0:
                print(i, time.strftime("%H:%M:%S"), flush=True); fh.flush()
    print("done", flush=True)
