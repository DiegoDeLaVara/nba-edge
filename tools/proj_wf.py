"""Walk-forward player projections for every stat; accuracy vs naive baselines; NB dispersion fitted pre-Nov-2025."""
import sys, time, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from edge.props import walk_forward, fit_r, STATS
D = pd.read_pickle("work/D.pkl")
out, disp = {}, {}
for st in STATS:
    t = time.time()
    W = walk_forward(D, st)
    W = W.merge(D[["gid", "pid", "min_e10", f"{st}_e5" if st != "pra" else "pra_e5", "min"]], on=["gid", "pid"])
    base = W[f"{st}_e5" if st != "pra" else "pra_e5"]
    ok = base.notna()
    mae_m = (W[st] - W.mu).abs()[ok].mean(); mae_b = (W[st] - base).abs()[ok].mean()
    rm = np.sqrt(((W[st] - W.mu) ** 2)[ok].mean()); rb = np.sqrt(((W[st] - base) ** 2)[ok].mean())
    pre = W[W.tip < "2025-11-01"]
    r = fit_r(pre.mu.values, pre[st].values.astype(int))
    disp[st] = r
    print(f"{st}: n={len(W)} MAE model {mae_m:.3f} vs last-5 EWMA {mae_b:.3f} | RMSE {rm:.3f} vs {rb:.3f} | NB r={r:.1f}  ({time.time()-t:.0f}s)", flush=True)
    W.to_pickle(f"work/wf_{st}.pkl")
json.dump(disp, open("work/nb_r.json", "w"))
