"""Are Kalshi NBA combos (parlays) mispriced? For settled combos whose legs are all NBA props / game lines:
price paid (last trade) vs how often they hit, by structure: same game vs different games, all unders / all overs,
'safe' legs (each leg >= 70% likely), number of legs. Leg prices = Kalshi mid at the hour the combo was created."""
import sys, json, glob; sys.path.insert(0, ".")
import numpy as np, pandas as pd

PROP = {"KXNBAPTS": "pts", "KXNBAREB": "reb", "KXNBAAST": "ast", "KXNBA3PT": "fg3m", "KXNBAPRA": "pra", "KXNBA2D": "dd", "KXNBA3D": "td"}
GAME = {"KXNBAGAME": "ml", "KXNBASPREAD": "spread", "KXNBATOTAL": "total", "KXNBATEAMTOTAL": "teamtot"}

# leg price paths
C = {}
for f in glob.glob("work/oc_*.jsonl"):
    for l in open(f):
        t, op, c = json.loads(l); C[t] = c
for s in ("KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL", "KXNBATEAMTOTAL", "KXNBA2D", "KXNBAPRA"):
    try:
        for l in open(f"work/kc_{s}.jsonl"):
            t, c = json.loads(l)
            C.setdefault(t, c)
    except FileNotFoundError:
        pass


def mid_at(c, ts):
    best = None
    for e, b, a, v in c:
        if e <= ts and b is not None and a is not None and 0 <= b < a <= 1:
            best = (e, (a + b) / 2)
    return best[1] if best and ts - best[0] <= 3 * 3600 else None


from edge.kalshi import parse_event, game_index
_G = pd.read_pickle("work/G.pkl"); _gi = game_index(_G); _tip = dict(zip(_G.gid, _G.tip))
rows = []
for l in open("work/combos_full.jsonl"):
    m = json.loads(l)
    if m["result"] not in ("yes", "no") or not m.get("last"):
        continue
    legs = m["legs"]
    sers = [x.split("-")[0] for x, _ in legs]
    if not all(s in PROP or s in GAME for s in sers):
        continue
    ts = pd.Timestamp(m["open"]).timestamp()
    tips = []
    for t, _ in legs:
        pe = parse_event("-".join(t.split("-")[:2])); g = _gi.get(pe) if pe else None
        tips.append(_tip[g].timestamp() if g else None)
    if None in tips or ts >= min(tips) - 300:          # bought before every game in it started (5 min margin)
        continue
    probs, kinds, sides, games = [], [], [], []
    ok = True
    for (t, side), s in zip(legs, sers):
        c = C.get(t)
        mid = mid_at(c, ts) if c else None
        if mid is None:
            ok = False; break
        probs.append(mid if side == "yes" else 1 - mid)
        kinds.append(PROP.get(s) or GAME.get(s)); sides.append(side)
        games.append(t.split("-")[1][-6:] + t.split("-")[1][:7])
    if not ok:
        continue
    price = float(m["last"])
    if not (0.005 < price < 0.995):
        continue
    props = [k in ("pts", "reb", "ast", "fg3m", "pra", "dd", "td") for k in kinds]
    rows.append({"ticker": m["ticker"], "ts": ts, "y": int(m["result"] == "yes"), "price": price, "n": len(legs),
                 "fair_ind": float(np.prod(probs)), "min_leg": min(probs), "mean_leg": float(np.mean(probs)),
                 "same_game": len(set(games)) == 1, "games": len(set(games)),
                 "all_props": all(props), "all_under": all(s == "no" for s, p in zip(sides, props) if p) and any(props) and all(props),
                 "all_over": all(s == "yes" for s, p in zip(sides, props) if p) and all(props),
                 "night": pd.to_datetime(ts - 10 * 3600, unit="s").date()})
X = pd.DataFrame(rows)
# price actually paid before tip (volume-weighted pre-tip trades), not the last (often in-game) print
TR = pd.DataFrame([json.loads(l) for l in open("work/combo_trades.jsonl")])
X = X.merge(TR[["ticker", "vwap_pre", "contracts_pre"]], on="ticker", how="inner")
X = X[X.vwap_pre.between(0.005, 0.995)].copy()
X["last_price"] = X.price; X["price"] = X.vwap_pre
X = X.sort_values("ts", kind="stable").reset_index(drop=True)
X["roi"] = (X.y - X.price) / X.price
X["H1"] = X.ts <= X.ts.quantile(0.5)
print(f"all-NBA combos with leg prices: {len(X)} over {X.night.nunique()} nights; legs {X.n.value_counts().sort_index().to_dict()}")
print(f"buyers overall: paid {X.price.mean():.3f} avg, hit {X.y.mean():.3f}, ROI {X.roi.mean():+.3f} (bet-weighted); "
      f"price / independent fair = {(X.price / X.fair_ind).median():.2f} (median)")


def show(name, S):
    if len(S) < 40:
        print(f"  {name:42s} n={len(S)} (too few)"); return
    nights = S.night.values; un = np.unique(nights)
    rng = np.random.default_rng(1)
    idx = {d: np.where(nights == d)[0] for d in un}
    bs = []
    for _ in range(400):
        ii = np.concatenate([idx[d] for d in rng.choice(un, len(un))])
        bs.append(S.y.values[ii].sum() / S.price.values[ii].sum() - 1)
    roi_d = S.y.sum() / S.price.sum() - 1         # per dollar staked (equal stake per combo would be S.roi.mean())
    print(f"  {name:42s} n={len(S):6d} paid {S.price.mean():.3f} hit {S.y.mean():.3f} | ROI per $ {roi_d:+.3f} "
          f"(H1 {S[S.H1].y.sum() / S[S.H1].price.sum() - 1:+.3f} H2 {S[~S.H1].y.sum() / S[~S.H1].price.sum() - 1:+.3f}) | "
          f"indep. fair {S.fair_ind.mean():.3f} | P(ROI<=0) {np.mean(np.array(bs) <= 0):.2f}")


show("ALL", X)
show("same game", X[X.same_game]); show("different games", X[~X.same_game])
show("all props, all UNDERS", X[X.all_under]); show("all props, all OVERS", X[X.all_over])
show("same game + all overs", X[X.same_game & X.all_over])
show("SAFE legs (every leg >= 70%)", X[X.min_leg >= 0.70]); show("safe legs >= 80%", X[X.min_leg >= 0.80])
show("safe >= 70% + same game", X[(X.min_leg >= 0.70) & X.same_game])
show("safe >= 70% + different games", X[(X.min_leg >= 0.70) & ~X.same_game])
show("safe >= 80% + same game", X[(X.min_leg >= 0.80) & X.same_game])
for k in (2, 3, 4, 5, 6):
    show(f"{k} legs", X[X.n == k])
show("6+ legs", X[X.n >= 6])
X["band"] = pd.cut(X.price, [0, .05, .15, .3, .5, 1])
for b, S in X.groupby("band", observed=True):
    show(f"combo price {b}", S)
X.to_pickle("work/combo_rows.pkl")
