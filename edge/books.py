"""NBA line shopping: US sportsbooks vs a sharp fair price.

Fair price = Kalshi's mid for the same line (free: game lines, spreads, totals and every prop threshold), averaged
with Pinnacle's no-vig line when the Odds API returns it. A bet is logged when a book pays more than fair by min_ev
(after nothing — books have no fee; the margin is in the price). Closing-line value is measured free against
Kalshi's mid at the last refresh before tip. Credits come from the ODDS_API_KEY secret (shared with Tennis Edge), so
the scanner spends a small daily budget and keeps a reserve.
"""
import json, math, os, time, datetime as dt
from pathlib import Path
import requests

from edge.kalshi import parse_event, norm_name, K2E

API = "https://api.the-odds-api.com/v4/sports/basketball_nba"
US_BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet", "fanatics", "hardrockbet", "ballybet"]
BOOKS = ["pinnacle"] + US_BOOKS                       # 10 bookmakers = one region of credits
NICE = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
        "betrivers": "BetRivers", "espnbet": "ESPN BET", "fanatics": "Fanatics", "hardrockbet": "Hard Rock",
        "ballybet": "Bally Bet", "pinnacle": "Pinnacle"}
PROP_MKT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m",
            "player_points_rebounds_assists": "pra"}
TEAM = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET", "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO",
        "New York Knicks": "NY", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SA",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTAH", "Washington Wizards": "WSH"}
DEFAULTS = {"daily_credits": 4, "reserve": 80, "main_every_hours": 6, "main_window_hours": [0.5, 10], "props_window_hours": [0.3, 4],
            "props_markets": ["player_points", "player_threes", "player_rebounds"], "props_events_per_day": 0,
            "min_ev": 0.02, "min_ev_one_source": 0.03, "max_ev": 0.15, "max_odds": 3.0, "max_kalshi_spread": 0.05,
            "kelly": 0.25, "max_units": 1.0, "staked": True}


def cfg(CFG):
    c = dict(DEFAULTS); c.update(CFG.get("books", {})); return c


def key():
    return os.environ.get("ODDS_API_KEY")


def devig(prices):
    """Decimal odds -> fair probabilities (power method)."""
    imp = [1 / p for p in prices]
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if sum(i ** k for i in imp) > 1:
            lo = k
        else:
            hi = k
    out = [i ** k for i in imp]; t = sum(out)
    return [o / t for o in out]


def american(d):
    return f"+{(d - 1) * 100:.0f}" if d >= 2 else f"-{100 / (d - 1):.0f}"


# ------------------------------------------------------------------ Kalshi fair prices (free)
def kalshi_lines(kget, games):
    """{(gid, kind, team_or_None, strike): mid} from open KXNBAGAME / KXNBASPREAD / KXNBATOTAL markets.
    kind 'ml': P(team wins); 'spread': P(team wins by more than strike); 'total': P(total > strike)."""
    import pandas as pd
    et = {}
    for g in games:
        tip = pd.Timestamp(g["tip"]).tz_convert("America/New_York").tz_localize(None).normalize()
        et[(tip, g["away"], g["home"])] = g["gid"]
    out = {}
    for ser, kind in (("KXNBAGAME", "ml"), ("KXNBASPREAD", "spread"), ("KXNBATOTAL", "total")):
        cur = ""
        for _ in range(5):
            p = {"series_ticker": ser, "status": "open", "limit": 1000}
            if cur:
                p["cursor"] = cur
            d = kget("/markets", p)
            for m in d.get("markets") or []:
                pe = parse_event(m.get("event_ticker", ""))
                gid = et.get(pe) if pe else None
                if gid is None:
                    continue
                try:
                    b, a = float(m.get("yes_bid_dollars")), float(m.get("yes_ask_dollars"))
                except (TypeError, ValueError):
                    continue
                if not (0 < a <= 1 and 0 <= b < a):
                    continue
                suf = m["ticker"].split("-")[-1]
                if kind == "ml":
                    team = K2E.get(suf, suf); strike = None
                elif kind == "spread":
                    code = suf.rstrip("0123456789"); team = K2E.get(code, code); strike = float(m.get("floor_strike"))
                else:
                    team = None; strike = float(m.get("floor_strike"))
                out[(gid, kind, team, strike)] = ((a + b) / 2, a - b)
            cur = d.get("cursor") or ""
            if not cur:
                break
    return out


def kalshi_fair(line, KL, props_idx, gid, home, away, c):
    """Fair probability of one Odds API outcome from Kalshi, or None."""
    mk, name, pt, desc = line
    def get(k):
        v = KL.get(k)
        return v[0] if v and v[1] <= c["max_kalshi_spread"] else None
    if mk == "h2h":
        t = TEAM.get(name)
        p = get((gid, "ml", t, None))
        if p is None:                                   # use the other side
            o = home if t == away else away
            q = get((gid, "ml", o, None)); p = None if q is None else 1 - q
        return p
    if mk == "spreads" and pt is not None:
        t = TEAM.get(name); o = home if t == away else away; s = abs(pt)
        if pt < 0:                                      # team -s covers  <=> team wins by more than s
            return get((gid, "spread", t, s))
        q = get((gid, "spread", o, s))                  # team +s covers <=> opponent does NOT win by more than s
        return None if q is None else 1 - q
    if mk == "totals" and pt is not None:
        q = get((gid, "total", None, pt))
        if q is None:
            return None
        return q if name == "Over" else 1 - q
    if mk in PROP_MKT and pt is not None:
        st = PROP_MKT[mk]
        r = props_idx.get((gid, norm_name(desc or ""), st, pt + 0.5))
        if r is None or r[1] > c["max_kalshi_spread"]:
            return None
        return r[0] if name == "Over" else 1 - r[0]
    return None


# ------------------------------------------------------------------ Odds API
def odds_get(path, params, state, log):
    k = key()
    if not k:
        return None
    try:
        r = requests.get(API + path, params={"apiKey": k, **params}, timeout=40)
    except requests.RequestException as e:
        log(f"books: request failed {type(e).__name__}"); return None
    used = int(float(r.headers.get("x-requests-last", 0) or 0))
    rem = r.headers.get("x-requests-remaining")
    day = dt.datetime.utcnow().strftime("%Y-%m-%d")
    state.setdefault("spend", {})[day] = state["spend"].get(day, 0) + used
    if rem is not None:
        state["remaining"] = int(float(rem))
    if r.status_code != 200:
        log(f"books: HTTP {r.status_code}"); return None
    return r.json()


def budget_left(state, c):
    day = dt.datetime.utcnow().strftime("%Y-%m-%d")
    spent = state.get("spend", {}).get(day, 0)
    rem = state.get("remaining")
    room = c["daily_credits"] - spent
    if rem is not None:
        room = min(room, rem - c["reserve"])
    return room


def best_lines(event, c):
    """Per outcome: best US-book price and the Pinnacle no-vig fair (if any)."""
    pin, best = {}, {}
    for b in event.get("bookmakers", []):
        for m in b.get("markets", []):
            groups = {}
            for o in m.get("outcomes", []):
                pt = o.get("point")
                gk = (m["key"], o.get("description"), abs(pt) if (m["key"] == "spreads" and pt is not None) else pt)
                groups.setdefault(gk, []).append(o)
                if b["key"] in US_BOOKS:
                    k = (m["key"], o["name"], pt, o.get("description"))
                    if k not in best or o["price"] > best[k][0]:
                        best[k] = (o["price"], b["key"])
            if b["key"] == "pinnacle":
                for gk, outs in groups.items():
                    if len(outs) == 2:
                        f = devig([x["price"] for x in outs])
                        for x, p in zip(outs, f):
                            pin[(m["key"], x["name"], x.get("point"), x.get("description"))] = p
    return best, pin


def find(event, game, KL, props_idx, c, now):
    gid, home, away = game["gid"], game["home"], game["away"]
    best, pin = best_lines(event, c)
    out = []
    for line, (price, book) in best.items():
        fk = kalshi_fair(line, KL, props_idx, gid, home, away, c)
        fp = pin.get(line)
        srcs = [x for x in (fk, fp) if x is not None]
        if not srcs:
            continue
        fair = sum(srcs) / len(srcs)
        ev = fair * price - 1
        need = c["min_ev"] if len(srcs) == 2 else c["min_ev_one_source"]
        if price > c["max_odds"] or ev < need:
            continue
        mk, name, pt, desc = line
        kelly = ev / (price - 1)
        units = max(0.25, min(c["max_units"], math.floor(c["kelly"] * kelly * 100 * 4) / 4))
        flag = "check the line (very high EV: stale or limited)" if ev > c["max_ev"] else ""
        if flag:
            units = min(units, 0.25)
        if mk == "h2h":
            label = f"{name} to win"
        elif mk == "spreads":
            label = f"{name} {pt:+g}"
        elif mk == "totals":
            label = f"{name} {pt:g}"
        else:
            label = f"{desc} {name.lower()} {pt:g} {PROP_MKT.get(mk, mk)}"
        out.append({"key": f"{gid}|{mk}|{name}|{pt}|{desc}", "gid": gid, "game": f"{away} @ {home}", "tip": game["tip"],
                    "market": mk, "outcome": name, "point": pt, "player": desc, "label": label, "book": book,
                    "book_name": NICE.get(book, book), "price": price, "american": american(price), "fair": round(fair, 4),
                    "fair_kalshi": None if fk is None else round(fk, 4), "fair_pinnacle": None if fp is None else round(fp, 4),
                    "ev": round(ev, 4), "units": units if c["staked"] else 0.0, "flag": flag, "seen": now})
    return out


# ------------------------------------------------------------------ one scan per refresh
def run(data, games, props, kget, CFG, log, now_ts=None, fake_events=None):
    """Scan within budget, log new +EV bets, refresh closing fair from Kalshi, grade finished bets. Returns board dict."""
    c = cfg(CFG)
    now = now_ts or time.time()
    state_p, log_p = Path(data) / "books_state.json", Path(data) / "books_log.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    L = json.loads(log_p.read_text()) if log_p.exists() else {}
    import pandas as pd
    upcoming = [g for g in games if pd.Timestamp(g["tip"]).timestamp() > now]
    gmap = {}
    for g in upcoming:
        gmap[(g["home"], g["away"])] = g
    props_idx = {}
    if props is not None and len(props):
        for r in props.itertuples(index=False):
            props_idx[(r.gid, norm_name(r.player), r.stat, r.k)] = ((r.bid + r.ask) / 2, r.ask - r.bid)
    KL = kalshi_lines(kget, upcoming) if upcoming else {}
    has_key = bool(key()) or fake_events is not None
    found = []
    if has_key and upcoming:
        hrs = lambda g: (pd.Timestamp(g["tip"]).timestamp() - now) / 3600
        lo, hi = c["main_window_hours"]
        due = any(lo <= hrs(g) <= hi for g in upcoming) and now - state.get("last_main", 0) > c["main_every_hours"] * 3600
        events = []
        if fake_events is not None:
            events = fake_events
        elif due and budget_left(state, c) >= 3:
            d = odds_get("/odds", {"bookmakers": ",".join(BOOKS), "markets": "h2h,spreads,totals", "oddsFormat": "decimal"}, state, log)
            if d is not None:
                events = d; state["last_main"] = now
        # props: a few events per day, the ones starting soonest
        plo, phi = c["props_window_hours"]
        if fake_events is None and c["props_events_per_day"] > 0 and c["props_markets"]:
            day = dt.datetime.utcnow().strftime("%Y-%m-%d")
            done = state.setdefault("props_done", {}).setdefault(day, [])
            ev_list = odds_get("/events", {}, state, log) or []          # free endpoint
            for e in ev_list:
                g = gmap.get((TEAM.get(e["home_team"]), TEAM.get(e["away_team"])))
                if not g or e["id"] in done or len(done) >= c["props_events_per_day"] or not (plo <= hrs(g) <= phi):
                    continue
                if budget_left(state, c) < len(c["props_markets"]):
                    break
                d = odds_get(f"/events/{e['id']}/odds", {"bookmakers": ",".join(BOOKS), "markets": ",".join(c["props_markets"]),
                                                          "oddsFormat": "decimal"}, state, log)
                if d:
                    events.append(d); done.append(e["id"])
        for e in events:
            g = gmap.get((TEAM.get(e.get("home_team")), TEAM.get(e.get("away_team"))))
            if g:
                found += find(e, g, KL, props_idx, c, now)
    for b in found:                                   # log at first sighting; keep the best price seen
        e = L.get(b["key"])
        if e is None:
            L[b["key"]] = dict(b)
        elif e.get("result") is None and b["price"] > e["price"]:
            e.update({k: b[k] for k in ("price", "american", "book", "book_name", "ev", "units", "fair")})
    # closing fair: Kalshi mid at the last refresh before tip (free)
    for e in L.values():
        if e.get("result") is None and pd.Timestamp(e["tip"]).timestamp() > now:
            g = {"gid": e["gid"]}
            gm = next((x for x in upcoming if x["gid"] == e["gid"]), None)
            if gm:
                fk = kalshi_fair((e["market"], e["outcome"], e["point"], e["player"]), KL, props_idx, gm["gid"], gm["home"], gm["away"], c)
                if fk is not None:
                    e["close_fair"] = round(fk, 4); e["clv"] = round(fk * e["price"] - 1, 4)
    state_p.write_text(json.dumps(state)); log_p.write_text(json.dumps(L))
    return state, L, found, c, has_key


def grade(L, G, P):
    """Settle logged bets from ESPN scores / box scores."""
    import pandas as pd
    Gi = G.set_index("gid")
    stat_box = {}
    Pp = P[P["gid"].isin({e["gid"] for e in L.values() if e.get("result") is None})]
    for r in Pp.itertuples(index=False):
        d = {"pts": r.pts, "reb": r.reb, "ast": r.ast, "fg3m": r.fg3m}
        d["pra"] = (r.pts or 0) + (r.reb or 0) + (r.ast or 0)
        stat_box[(r.gid, norm_name(r.name))] = (bool(r.played), d)
    for e in L.values():
        if e.get("result") is not None or e["gid"] not in Gi.index:
            continue
        g = Gi.loc[e["gid"]]
        if pd.isna(g["hs"]):
            continue
        hs, as_ = float(g["hs"]), float(g["as"]); home, away = g["home"], g["away"]
        mk, name, pt = e["market"], e["outcome"], e["point"]
        res = None
        if mk == "h2h":
            t = TEAM.get(name); res = (hs > as_) if t == home else (as_ > hs)
        elif mk == "spreads":
            t = TEAM.get(name); m = (hs - as_) if t == home else (as_ - hs)
            res = "push" if m + pt == 0 else (m + pt > 0)
        elif mk == "totals":
            tot = hs + as_; res = "push" if tot == pt else ((tot > pt) if name == "Over" else (tot < pt))
        elif mk in PROP_MKT:
            hit = stat_box.get((e["gid"], norm_name(e["player"] or "")))
            if hit is None:
                continue
            played, d = hit
            if not played:
                res = "void"
            else:
                v = d[PROP_MKT[mk]]; res = (v > pt) if name == "Over" else (v < pt)
        if res is None:
            continue
        if res in ("push", "void"):
            e["result"] = res; e["pnl"] = 0.0
        else:
            e["result"] = "won" if res else "lost"; e["pnl"] = round(e["price"] - 1 if res else -1.0, 4)
    return L


def summary(L, state, c, has_key):
    done = [e for e in L.values() if e.get("result") in ("won", "lost")]
    n = len(done)
    staked = sum(e.get("units", 0) for e in done)
    units = sum(e.get("units", 0) * e["pnl"] for e in done)
    clv = [e["clv"] for e in L.values() if e.get("clv") is not None]
    day = dt.datetime.utcnow().strftime("%Y-%m-%d")
    return {"has_key": has_key, "n": n, "won": sum(e["result"] == "won" for e in done),
            "roi": round(sum(e["pnl"] for e in done) / n, 4) if n else None,
            "units": round(units, 2), "units_staked": round(staked, 2), "clv": round(sum(clv) / len(clv), 4) if clv else None,
            "clv_n": len(clv), "credits_left": state.get("remaining"), "spent_today": state.get("spend", {}).get(day, 0),
            "daily_budget": c["daily_credits"], "reserve": c["reserve"], "last_scan": state.get("last_main")}
