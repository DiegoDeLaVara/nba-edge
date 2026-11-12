"""Pre-registered follow-up from the points study: role players (min_e10 < 24) where the model disagrees with Kalshi
by > 3% after fee. Tested on every stat with luck runs (outcomes redrawn from the Kalshi mid, per player-game)."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from tools.prop_test import bets

D = pd.read_pickle("work/D.pkl")[["gid", "pid", "min_e10"]]
rng = np.random.default_rng(11)
for st in sys.argv[1:]:
    R = pd.read_pickle(f"work/pt_{st}.pkl").merge(D, on=["gid", "pid"], how="left")
    R = R[R.min_e10 < 24].sort_values("tip", kind="stable").reset_index(drop=True)
    H1 = (R.tip <= R.tip.quantile(.5)).values
    by, bn, pnl = bets(R, R.p, 0.03); m = ~np.isnan(pnl)
    key = R.groupby(["gid", "pid"]).ngroup().values
    luck = []
    for _ in range(500):
        u = rng.random(key.max() + 1)[key]
        _, _, pl = bets(R.assign(y=(u < R.mid).astype(int)), R.p, 0.03)
        luck.append(np.nanmean(pl))
    luck = np.array(luck); obs = np.nanmean(pnl)
    print(f"{st}: role-player model bets n={m.sum()} ROI {obs:+.3f} | H1 {np.nanmean(pnl[m & H1]):+.3f} H2 {np.nanmean(pnl[m & ~H1]):+.3f}"
          f" | luck mean {luck.mean():+.3f}, 95th pct {np.quantile(luck, .95):+.3f}, p {np.mean(luck >= obs):.3f}")
