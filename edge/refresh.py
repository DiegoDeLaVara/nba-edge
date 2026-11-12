"""One board refresh (cloud, every ~20 min):

  python -m edge.refresh --data pub [--train]

history update (ESPN) -> daily model retrain -> today's projections (ESPN schedule + injury report) -> Kalshi open
prop markets -> model probability, EV after Kalshi fee -> picks (proven strategies stake, others paper) -> pick log,
grading, record -> pub/board.json. Rolling state lives in --data (published to gh-pages)."""
import argparse, json, math, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from edge import history, train
from edge.dist import Dist
from edge.kalshi import parse_event, norm_name, fee, K2E
from edge.live import project

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.elections.kalshi.com/trade-api/v2"
KS = requests.Session(); KS.headers.update({"User-Agent": "nba-edge/1.0 (github actions)"})
SERIES = {"fg3m": "KXNBA3PT", "pts": "KXNBAPTS", "reb": "KXNBAREB", "ast": "KXNBAAST", "pra": "KXNBAPRA"}
LABEL = {"fg3m": "threes", "pts": "points", "reb": "rebounds", "ast": "assists", "pra": "pts+reb+ast"}
CFG = json.loads((ROOT / "config.json").read_text())
CORR = {"same_player": {"pts-reb": 0.25, "pts-ast": 0.14, "reb-ast": 0.18}, "teammates": -0.06, "opponents": 0.0,
        "player_vs_game_total": 0.17}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kget(path, params, tries=5):
    for i in range(tries):
        try:
            r = KS.get(API + path, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 + 2 * i); continue
        except requests.RequestException:
            time.sleep(2 + 2 * i)
    return {}


def kalshi_open(series):
    out, cur = [], ""
    for _ in range(10):
        p = {"series_ticker": series, "status": "open", "limit": 1000}
        if cur:
            p["cursor"] = cur
        d = kget("/markets", p)
        ms = d.get("markets") or []
        out += ms; cur = d.get("cursor") or ""
        if not cur or not ms:
            break
        time.sleep(0.3)
    return out


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_json(p, default):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def kelly_units(p, cost, mult, cap):
    """Kelly stake on a binary contract bought at `cost` (incl. fee): f = (p - cost) / (1 - cost); units = % bankroll."""
    if cost <= 0 or cost >= 1 or p <= cost:
        return 0.0
    f = (p - cost) / (1 - cost) * mult
    return round(min(cap, 100 * f), 2)


def price_props(D, games, dists):
    """Match open Kalshi props to projections; model P(stat >= k) and EV for both sides."""
    by_game = {}
    tipmap = {g["gid"]: g for g in games}
    et = {}
    for g in games:
        tip = pd.Timestamp(g["tip"]).tz_convert("America/New_York").tz_localize(None).normalize()
        et[(tip, g["away"], g["home"])] = g["gid"]
    D = D.copy(); D["nn"] = D["name"].map(norm_name)
    for gid, d in D.groupby("gid"):
        by_game[gid] = d
    rows = []
    for st, ser in SERIES.items():
        for m in kalshi_open(ser):
            pe = parse_event(m.get("event_ticker", ""))
            gid = et.get(pe) if pe else None
            if gid is None or gid not in by_game:
                continue
            name, _, k = (m.get("yes_sub_title") or "").rpartition(":")
            try:
                k = float(k.strip().rstrip("+"))
            except ValueError:
                continue
            d = by_game[gid]; hit = d[d["nn"] == norm_name(name)]
            if len(hit) != 1:
                continue
            r = hit.iloc[0]
            bid, ask = fnum(m.get("yes_bid_dollars")), fnum(m.get("yes_ask_dollars"))
            if bid is None or ask is None or not (0 < ask <= 1) or not (0 <= bid < ask):
                continue
            mu = float(r[f"mu_{st}"])
            p_model = float(dists[st].p_over([mu], [k])[0])
            p = p_model
            bl = CFG.get("blend", {}).get(st)
            if bl:                                   # fair price = blend of Kalshi mid and model (logit space)
                lg = lambda x: math.log(min(max(x, 1e-4), 1 - 1e-4) / (1 - min(max(x, 1e-4), 1 - 1e-4)))
                z = bl["c"] + bl["a_mid"] * lg((bid + ask) / 2) + bl["b_model"] * lg(p_model)
                p = 1 / (1 + math.exp(-z))
            cy = ask + fee(ask); cn = (1 - bid) + fee(1 - bid)
            rows.append({"ticker": m["ticker"], "gid": gid, "pid": str(r["pid"]), "player": r["name"], "team": r["team"],
                         "opp": r["opp"], "stat": st, "label": LABEL[st], "k": k, "bid": bid, "ask": ask, "mu": round(mu, 2),
                         "p": round(p, 4), "p_model": round(p_model, 4), "ev_yes": round((p - cy) / cy, 4), "ev_no": round(((1 - p) - cn) / cn, 4),
                         "cost_yes": round(cy, 4), "cost_no": round(cn, 4), "inj": r.get("inj", ""),
                         "tip": tipmap[gid]["tip"], "vol": fnum(m.get("volume_fp")) or 0.0})
    return pd.DataFrame(rows)


def picks_from(props, now):
    """One pick per player-stat per strategy (a player's lines move together): the line with the biggest edge."""
    out = []
    for s in CFG["strategies"]:
        if props.empty:
            break
        X = props[props["stat"].isin(s["stats"])]
        X = X[((X["ask"] - X["bid"]) <= s.get("max_spread", 0.10))]
        mid = (X["bid"] + X["ask"]) / 2
        X = X[(mid >= s.get("mid_lo", 0)) & (mid <= s.get("mid_hi", 1))]
        best = {}
        for r in X.itertuples(index=False):
            hours = (pd.Timestamp(r.tip) - now).total_seconds() / 3600
            if hours <= 0 or hours > s.get("max_hours", 30):
                continue
            for side, ev, cost, p in (("YES", r.ev_yes, r.cost_yes, r.p), ("NO", r.ev_no, r.cost_no, 1 - r.p)):
                edge = p - cost
                if edge > s["min_edge"]:
                    units = kelly_units(p, cost, s["kelly"], s["max_units"])     # paper strategies log would-be units
                    pk = {"strategy": s["name"], "staked": s["staked"], "ticker": r.ticker, "side": side, "gid": r.gid,
                          "pid": r.pid, "player": r.player, "team": r.team, "opp": r.opp, "stat": r.stat, "label": r.label,
                          "k": r.k, "pay": round(r.ask if side == "YES" else 1 - r.bid, 4), "cost": round(cost, 4),
                          "p": round(p, 4), "edge": round(edge, 4), "ev": round(ev, 4), "units": units, "tip": r.tip,
                          "mu": r.mu, "inj": r.inj, "hours": round(hours, 1)}
                    key = (r.gid, r.pid, r.stat)
                    if key not in best or edge > best[key]["edge"]:
                        best[key] = pk
        out += list(best.values())
    return out


def update_log(logp, picks, props, now, P):
    """Pick log: first sighting price, latest pre-tip price (closing line), result when the game is final."""
    L = load_json(logp, {})
    quote = {r.ticker: (r.bid, r.ask) for r in props.itertuples(index=False)} if not props.empty else {}
    for pk in picks:
        key = f"{pk['ticker']}|{pk['side']}|{pk['strategy']}"
        e = L.get(key)
        if e is None:
            e = dict(pk); e["t0"] = now.isoformat(); e["entry"] = pk["pay"]; e["entry_p"] = pk["p"]
            L[key] = e
        if e.get("result") is None and pd.Timestamp(e["tip"]) > now:
            e["p_last"] = pk["p"]; e["units_last"] = pk["units"]
    for key, e in L.items():
        if e.get("result") is None and pd.Timestamp(e["tip"]) > now and e["ticker"] in quote:
            b, a = quote[e["ticker"]]
            e["close"] = a if e["side"] == "YES" else 1 - b      # last pre-tip price we saw
    # grade from ESPN box scores in the history
    Pg = P[P["gid"].isin({e["gid"] for e in L.values() if e.get("result") is None})]
    stat_box = {}
    if len(Pg):
        for st in ("pts", "reb", "ast", "fg3m"):
            for g, p, pl, v in zip(Pg["gid"], Pg["pid"], Pg["played"], Pg[st]):
                stat_box[(g, str(p), st)] = (pl, v)
        for g, p, pl, a, b, c in zip(Pg["gid"], Pg["pid"], Pg["played"], Pg["pts"], Pg["reb"], Pg["ast"]):
            stat_box[(g, str(p), "pra")] = (pl, (a or 0) + (b or 0) + (c or 0))
    for e in L.values():
        if e.get("result") is not None:
            continue
        hit = stat_box.get((e["gid"], e["pid"], e["stat"]))
        if hit is None:
            continue
        played, v = hit
        if not played:
            e["result"] = "void"; e["pnl"] = 0.0; continue
        yes = float(v) >= e["k"]
        won = yes if e["side"] == "YES" else not yes
        e["result"] = "won" if won else "lost"; e["actual"] = v
        e["pnl"] = round((1 - e["cost"]) / e["cost"] if won else -1.0, 4)     # per unit staked
    Path(logp).write_text(json.dumps(L))
    return L


def record(L):
    out = {}
    for e in L.values():
        s = out.setdefault(e["strategy"], {"n": 0, "won": 0, "lost": 0, "void": 0, "open": 0, "roi_sum": 0.0, "units": 0.0,
                                           "units_staked": 0.0, "clv_sum": 0.0, "clv_n": 0})
        if e.get("result") is None:
            s["open"] += 1; continue
        if e["result"] == "void":
            s["void"] += 1; continue
        s["n"] += 1; s["won" if e["result"] == "won" else "lost"] += 1
        s["roi_sum"] += e["pnl"]
        u = e.get("units_last", e.get("units", 0)) or 0
        s["units"] += u * e["pnl"]; s["units_staked"] += u
        if e.get("close") is not None:
            s["clv_sum"] += e["close"] - e["entry"]; s["clv_n"] += 1
    for s in out.values():
        s["roi"] = round(s["roi_sum"] / s["n"], 4) if s["n"] else None
        s["clv"] = round(s["clv_sum"] / s["clv_n"], 4) if s["clv_n"] else None
        s["units"] = round(s["units"], 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out")
    ap.add_argument("--train", action="store_true", help="force a model retrain")
    a = ap.parse_args()
    data = Path(a.data); data.mkdir(parents=True, exist_ok=True)
    state = ROOT / "state"
    hist_dir = data if (data / "hist_games.csv.gz").exists() else state
    if hist_dir is state:                       # first run: seed rolling history from the repo
        import shutil
        for f in ("hist_games.csv.gz", "hist_players.csv.gz"):
            shutil.copy(state / f, data / f)
    now = pd.Timestamp.now(tz="UTC")
    G, P, n_new = history.update(data, log=log)
    G, P = history.frames(G, P)
    mp = data / "models.pkl"
    meta = load_json(data / "models.json", {})
    stale = (not mp.exists() or a.train or n_new > 0 and time.time() - meta.get("ts", 0) > 18 * 3600
             or time.time() - meta.get("ts", 0) > 36 * 3600)
    if stale:
        models = train.fit_all(G, P, log=log); train.save(models, mp)
        (data / "models.json").write_text(json.dumps({"ts": time.time(), "trained": models["trained"]}))
    else:
        models = train.load(mp)
    dists = {st: Dist.from_json(json.loads((ROOT / "model" / f"dist_{st}.json").read_text())) for st in SERIES}
    today = now.tz_convert("America/New_York").date()
    games = [g for d in (today, today + dt.timedelta(days=1)) for g in history.scoreboard(d)]
    games = [g for g in games if not g["completed"] and g["home"] and g["away"]]
    inj = history.injuries()
    D, X = project(G, P, games, inj, models) if games else (pd.DataFrame(), pd.DataFrame())
    if len(D):
        D = D[D["n_prev"] >= 3]
    props = price_props(D, games, dists) if len(D) else pd.DataFrame()
    log(f"games {len(games)}, players {len(D)}, priced props {len(props)}")
    picks = picks_from(props, now)
    L = update_log(data / "picklog.json", picks, props, now, P)
    rec = record(L)
    gm = []
    for g in games:
        x = X[X["gid"] == g["gid"]] if len(X) else X
        gm.append({**g, "xmargin": round(float(x["xmargin"].iloc[0]), 1) if len(x) else None,
                   "xtotal": round(float(x["xtotal"].iloc[0]), 1) if len(x) else None})
    proj = []
    if len(D):
        for r in D.itertuples(index=False):
            proj.append({"gid": r.gid, "pid": str(r.pid), "player": r.name, "team": r.team, "opp": r.opp, "inj": r.inj,
                         "min": round(float(r.min_e10), 1), **{st: round(float(getattr(r, f"mu_{st}")), 1) for st in SERIES}})
    recent = sorted([e for e in L.values() if e.get("result") is not None], key=lambda e: e["tip"], reverse=True)[:150]
    board = {"generated": now.isoformat(), "model_trained": (load_json(data / "models.json", {})).get("trained"),
             "config": CFG, "corr": CORR, "games": gm, "projections": proj,
             "props": props[props["stat"].isin({st for x in CFG["strategies"] for st in x["stats"]})].to_dict("records") if len(props) else [], "picks": picks, "record": rec,
             "recent": recent, "open": [e for e in L.values() if e.get("result") is None]}
    (data / "board.json").write_text(json.dumps(board, default=str))
    log(f"picks {len(picks)} ({sum(1 for p in picks if p['staked'])} staked); record {json.dumps(rec)}")


if __name__ == "__main__":
    main()
