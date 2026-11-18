"""Rekomendacijų nesėkmių analizė: kada modelio pirmas pasiūlymas nėra geriausias."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
r = pd.read_csv(ROOT / "outputs" / "recommendation_eval.csv")
rows = []
for s, g in r.groupby("seg"):
    if len(g) < 2:
        continue
    t, o = g.loc[g.hgb_saving.idxmax()], g.loc[g.true_saving.idxmax()]
    rows.append(dict(seg=int(s), pick=t.action, pick_pred=t.hgb_saving, pick_true=t.true_saving,
                     best=o.action, best_true=o.true_saving))
d = pd.DataFrame(rows)
d["regret"] = d.best_true - d.pick_true
ct = pd.crosstab(d.pick, d.best)
out = dict(n=len(d), mean_regret=float(d.regret.mean()),
           zero_regret_share=float((d.regret < 1e-9).mean()),
           crosstab=ct.to_dict(),
           pick_counts=d.pick.value_counts().to_dict(),
           pick_correct={a: int(ct.loc[a, a]) if a in ct.columns else 0 for a in ct.index},
           worst=d.sort_values("regret").tail(3).round(2).to_dict("records"))
json.dump(out, open(ROOT / "outputs" / "failures.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
