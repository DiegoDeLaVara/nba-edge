import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd
from edge.kalshi import fee
from tools.prop_test import bets
def phase(t):
    d=pd.to_datetime(t,unit="s")
    return np.where(d<"2026-02-01","25-26 Nov-Jan",np.where(d<"2026-04-13","25-26 Feb-Apr",np.where(d<"2026-07-01","playoffs 26","26-27 season")))
D=pd.read_pickle("work/D.pkl")[["gid","pid","min_e10"]]
for st in sys.argv[1:]:
    R=pd.read_pickle(f"work/pt_{st}.pkl").merge(D,on=["gid","pid"],how="left"); R["ph"]=phase(R.tip)
    cy=R.ask+fee(R.ask); cn=(1-R.bid)+fee(1-R.bid)
    R["yes"]=(R.y-cy)/cy; R["no"]=((1-R.y)-cn)/cn
    by,bn,pnl=bets(R,R.p,0.03); R["mod"]=pnl
    R["yes_mid"]=np.where((R.ask>=.2)&(R.ask<.65),R.yes,np.nan)
    R["bench_mod"]=np.where(R.min_e10<24,R["mod"],np.nan)
    g=R.groupby("ph").agg(n=("y","size"),mid=("mid","mean"),hit=("y","mean"),spread=("ask",lambda s:(s-R.loc[s.index,"bid"]).mean()),
        all_yes=("yes","mean"),yes_20_65=("yes_mid","mean"),model=("mod","mean"),n_model=("mod","count"),bench_model=("bench_mod","mean"),n_bench=("bench_mod","count"))
    print(st); print(g.round(3).to_string())
