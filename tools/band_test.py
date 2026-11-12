"""Price-band rules (no model): back YES or NO at the real quote inside a price band. Halves + luck runs."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from edge.kalshi import fee

def test(R, label, rng=np.random.default_rng(3)):
    R = R.sort_values("tip", kind="stable").reset_index(drop=True)
    half = R.tip.quantile(0.5); H1 = (R.tip <= half).values
    key = R.groupby(["gid", "pid"]).ngroup().values if "pid" in R else np.arange(len(R))
    cy = (R.ask + fee(R.ask)).values; cn = ((1 - R.bid) + fee(1 - R.bid)).values
    rows = []
    for side, cost in (("YES", cy), ("NO", cn)):
        px = R.ask.values if side == "YES" else (1 - R.bid).values
        for lo, hi in ((0.01, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 0.90), (0.90, 0.97)):
            m = (px >= lo) & (px < hi)
            if m.sum() < 100: continue
            def pnl(y):
                win = y if side == "YES" else 1 - y
                return (win - cost) / cost
            p = pnl(R.y.values)
            luck = []
            for _ in range(300):
                u = rng.random(key.max() + 1)[key]
                luck.append(pnl((u < R.mid.values).astype(int))[m].mean())
            luck = np.array(luck)
            rows.append({"set": label, "side": side, "band": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()), "hit": (R.y.values[m] if side == "YES" else 1 - R.y.values[m]).mean(),
                         "px": px[m].mean(), "roi": p[m].mean(), "H1": p[m & H1].mean(), "H2": p[m & ~H1].mean(),
                         "luck": luck.mean(), "p": (luck >= p[m].mean()).mean()})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = []
    for st in sys.argv[1:]:
        R = pd.read_pickle(f"work/pt_{st}.pkl")
        out.append(test(R, st))
    T = pd.concat(out); pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
