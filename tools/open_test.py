"""Opening prices on Kalshi NBA props (2026-27): are they softer than the tip price, which way do they drift, and can
we profit by betting at the open (no model, or model+Kalshi blend)? Model rules are also run on games with NO known
absences on either team, because the model's absence feature uses who actually sat out (known at tip, maybe not at
the open)."""
import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from edge.kalshi import fee
from tools.value_hunt import logit

SER = {"pts": "KXNBAPTS", "reb": "KXNBAREB", "ast": "KXNBAAST", "fg3m": "KXNBA3PT"}
D = pd.read_pickle("work/D.pkl")[["gid", "pid", "miss_min", "opp_miss_min", "min_e10"]]
old = pd.read_pickle("work/vh_rows_v1.pkl")
rows = []
for st, ser in SER.items():
    oc = {}
    for l in open(f"work/oc_{ser}.jsonl"):
        t, op, c = json.loads(l); oc[t] = (op, c)
    R = pd.read_pickle(f"work/pt_{st}.pkl")
    R = R[pd.to_datetime(R.tip, unit="s") >= "2026-10-01"].copy()
    # blend coefficients fitted only on data before this season
    S = old[(old.stat == st) & ~old.fresh]
    lr = LogisticRegression(C=100).fit(np.c_[logit(S.mid), logit(S.p)], S.y)
    for r in R.itertuples(index=False):
        if r.ticker not in oc:
            continue
        op, c = oc[r.ticker]
        first = next(((e, b, a) for e, b, a, v in c if b is not None and a is not None and 0 < b < a < 1 and a - b <= 0.10 and (op is None or e >= op)), None)
        if not first or first[0] >= r.tip - 3600:
            continue
        e, b, a = first
        rows.append({"stat": st, "gid": r.gid, "pid": r.pid, "k": r.k, "y": r.y, "tip": r.tip, "p": r.p, "ob": b, "oa": a, "ots": e,
                     "tb": r.bid, "ta": r.ask, "hrs": (r.tip - e) / 3600,
                     "pb_open": lr.predict_proba(np.c_[logit((a + b) / 2), logit(r.p)])[0, 1],
                     "pb_tip": lr.predict_proba(np.c_[logit((r.bid + r.ask) / 2), logit(r.p)])[0, 1]})
X = pd.DataFrame(rows).merge(D, on=["gid", "pid"], how="left")
X["om"] = (X.ob + X.oa) / 2; X["tm"] = (X.tb + X.ta) / 2; X["drift"] = X.tm - X.om
X = X.sort_values("tip", kind="stable").reset_index(drop=True)
X["H1"] = X.tip <= X.tip.quantile(0.5)
X["clean"] = (X.miss_min.fillna(0) == 0) & (X.opp_miss_min.fillna(0) == 0)


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def boot(x, g, nb=400, rng=np.random.default_rng(4)):
    ug, inv = np.unique(g, return_inverse=True); s = np.bincount(inv, weights=x); c = np.bincount(inv)
    bs = [(s * w).sum() / (c * w).sum() for w in (np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(nb))]
    return float(np.mean(np.array(bs) <= 0))


print(f"markets with an opening quote >=1h before tip: {len(X)} ({X.gid.nunique()} games); median open {X.hrs.median():.1f}h before tip; "
      f"open spread {np.mean(X.oa - X.ob):.3f} vs tip spread {np.mean(X.ta - X.tb):.3f}")
print(f"log-loss  open mid {ll(X.om, X.y):.4f}   tip mid {ll(X.tm, X.y):.4f}   model {ll(X.p, X.y):.4f}   "
      f"blend at open {ll(X.pb_open, X.y):.4f}   blend at tip {ll(X.pb_tip, X.y):.4f}")
print(f"mean drift open->tip {X.drift.mean():+.4f}; mean |drift| {X.drift.abs().mean():.4f}")
print("drift by stat:", X.groupby("stat").drift.mean().round(4).to_dict())
X["band"] = pd.cut(X.om, [0, .2, .4, .6, .8, 1])
print("drift by opening price:", X.groupby("band", observed=True).drift.agg(["size", "mean"]).round(4).to_dict("index"))


def bet(S, q, thr, at="open"):
    b, a = (S.ob, S.oa) if at == "open" else (S.tb, S.ta)
    cy = a + fee(a); cn = (1 - b) + fee(1 - b)
    ey, en = q - cy, (1 - q) - cn
    side = np.where(ey >= en, "YES", "NO"); e = np.maximum(ey, en)
    x = np.where(side == "YES", (S.y - cy) / cy, ((1 - S.y) - cn) / cn)
    entry = np.where(side == "YES", a, 1 - b)
    clv = np.where(side == "YES", S.tm - a, (1 - S.tm) - (1 - b))
    return e > thr, x, clv


print("\nNo-model rules at the open (always buy one side in a price band):")
for lo, hi in ((.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95)):
    for side in ("YES", "NO"):
        S = X[(X.om >= lo) & (X.om < hi)]
        cost = (S.oa + fee(S.oa)) if side == "YES" else ((1 - S.ob) + fee(1 - S.ob))
        win = S.y if side == "YES" else 1 - S.y
        x = (win - cost) / cost
        print(f"  {side:3s} {lo:.2f}-{hi:.2f}: n={len(S):5d} ROI open {x.mean():+.3f} (H1 {x[S.H1].mean():+.3f} H2 {x[~S.H1].mean():+.3f})", end="")
        cost_t = (S.ta + fee(S.ta)) if side == "YES" else ((1 - S.tb) + fee(1 - S.tb))
        print(f" | same at tip {((win - cost_t) / cost_t).mean():+.3f}")

print("\nModel + Kalshi blend, bet at the OPEN (CLV = tip mid minus entry, cents):")
for nm, S in (("all games", X), ("no absences either team (leak-safe)", X[X.clean])):
    for thr in (0.02, 0.04, 0.06):
        m, x, clv = bet(S, S.pb_open, thr, "open")
        if m.sum() < 50:
            continue
        print(f"  {nm:36s} edge>{thr:.2f}: n={m.sum():5d} ROI {x[m].mean():+.3f} | H1 {x[m & S.H1.values].mean():+.3f} H2 {x[m & ~S.H1.values].mean():+.3f} "
              f"| CLV {100 * clv[m].mean():+.2f}c (positive {np.mean(clv[m] > 0):.2f}) | P(ROI<=0) {boot(x[m], S.gid.values[m]):.3f}")
    m, x, clv = bet(S, S.pb_tip, 0.02, "tip")
    print(f"  {nm:36s} same rule at TIP edge>0.02: n={m.sum()} ROI {x[m].mean():+.3f}")
X.to_pickle("work/open_rows.pkl")
