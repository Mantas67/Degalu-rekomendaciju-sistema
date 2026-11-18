"""
Pagrindinis eksperimentų skriptas:
  * duomenų valymas ir padalijimas pagal vairuotojus (GroupShuffleSplit);
  * A užduotis – degalų sąnaudų regresija: Baseline (vidurkis, LinearRegression)
    vs Proposed (HistGradientBoosting su monotoniškumo apribojimais);
  * B užduotis – vairavimo stiliaus klasifikacija (eco/normalus/agresyvus):
    Baseline (taisyklė pagal accel_rms) vs Proposed (HGB klasifikatorius);
  * C – rekomendacijų teisingumo patikra su fizikiniu „orakulu“;
  * D – latencijos matavimas, XAI (permutation importance, PDP);
  * E – realaus laiko kelionės demonstracija ir maršruto pasirinkimas.

Paleidimas:  python src/train_evaluate.py
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from physics import FEATURE_COLS, fuel_l_per_100km
from recommender import recommend, recommend_route, _candidate_actions, _safe

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; OUT = ROOT / "outputs"; MOD = ROOT / "models"
for p in (FIG, OUT, MOD):
    p.mkdir(exist_ok=True)
SEED = 42
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C_BASE, C_PROP, C_ALT = "#8c8c8c", "#1f6fb2", "#e07b39"
metrics = {}

# ------------------------------------------------------------------ duomenys
df = pd.read_csv(ROOT / "data" / "segments.csv")
n_raw = len(df)
q1 = df.groupby("road_type")["fuel_l_100km"].transform(lambda s: s.quantile(.25))
q3 = df.groupby("road_type")["fuel_l_100km"].transform(lambda s: s.quantile(.75))
iqr = q3 - q1
mask = (df.fuel_l_100km <= q3 + 3 * iqr) & (df.fuel_l_100km > 0.2)
df = df[mask].reset_index(drop=True)
metrics["data"] = dict(raw_segments=n_raw, removed_outliers=int(n_raw - len(df)),
                       segments=len(df), trips=int(df.trip_id.nunique()),
                       drivers=int(df.driver_id.nunique()),
                       fuel_mean=float(df.fuel_l_100km.mean()),
                       fuel_std=float(df.fuel_l_100km.std()),
                       style_counts=df["style"].value_counts().to_dict())

# padalijimas pagal vairuotojus, stratifikuotas pagal stilių: testui – po 4
# kiekvieno stiliaus vairuotojus (12 iš 60), kurių modelis niekada nematė
rng_split = np.random.default_rng(SEED)
drv_style = df.groupby("driver_id")["style_id"].first()
test_drivers = np.concatenate([rng_split.choice(drv_style[drv_style == s].index, 4,
                                                replace=False) for s in range(3)])
is_test = df.driver_id.isin(test_drivers)
train, test = df[~is_test].copy(), df[is_test].copy()
Xtr, ytr = train[FEATURE_COLS], train.fuel_l_100km
Xte, yte = test[FEATURE_COLS], test.fuel_l_100km
metrics["split"] = dict(train_segments=len(train), test_segments=len(test),
                        train_drivers=int(train.driver_id.nunique()),
                        test_drivers=int(test.driver_id.nunique()))

# ---- 1 pav. duomenų apžvalga
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for rt, c in zip(["miestas", "priemiestis", "magistralė"], [C_ALT, "#6aa84f", C_PROP]):
    ax[0].hist(df.loc[df.road_type == rt, "fuel_l_100km"], bins=60, alpha=.55,
               label=rt, color=c, range=(0, 25))
ax[0].set_xlabel("Degalų sąnaudos, l/100 km"); ax[0].set_ylabel("Segmentų sk.")
ax[0].legend(frameon=False); ax[0].set_title("Sąnaudų pasiskirstymas pagal kelio tipą")
corr_cols = ["fuel_l_100km", "avg_speed_kmh", "accel_rms", "harsh_acc_per_km",
             "stops_per_km", "idle_share", "avg_rpm", "congestion", "temp_c", "grade_pct"]
cm = df[corr_cols].corr().values
im = ax[1].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
short = ["degalai", "greitis", "accel_rms", "staig.gr.", "sustoj.", "tušč.eiga",
         "apsukos", "spūstis", "temp.", "nuolydis"]
ax[1].set_xticks(range(len(short))); ax[1].set_xticklabels(short, rotation=60, ha="right")
ax[1].set_yticks(range(len(short))); ax[1].set_yticklabels(short)
for i in range(len(short)):
    for j in range(len(short)):
        ax[1].text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=6,
                   color="white" if abs(cm[i, j]) > .5 else "black")
ax[1].set_title("Pirsono koreliacijų matrica")
fig.colorbar(im, ax=ax[1], fraction=.04)
fig.tight_layout(); fig.savefig(FIG / "fig1_duomenys.png"); plt.close(fig)

# ------------------------------------------------- A. regresija (sąnaudos)
def reg_metrics(y, p):
    return dict(RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)), R2=float(r2_score(y, p)),
                MAPE=float(np.mean(np.abs((y - p) / y)) * 100))

mono = {c: 0 for c in FEATURE_COLS}
mono.update({"accel_rms": 1, "harsh_acc_per_km": 1, "stops_per_km": 1,
             "idle_share": 1, "avg_rpm": 1, "load_kg": 1, "grade_pct": 1,
             "headwind_ms": 1, "ac_on": 1})
mono_cst = [mono[c] for c in FEATURE_COLS]

# hiperparametrų paieška su GroupKFold (vairuotojai nesikerta tarp fold'ų)
grid = [dict(learning_rate=lr, max_leaf_nodes=ln, l2_regularization=l2)
        for lr in (0.05, 0.1) for ln in (15, 31, 63) for l2 in (0.0, 1.0)]
gkf = GroupKFold(n_splits=4)
cv_rows = []
for g in grid:
    scores = []
    for a, b in gkf.split(Xtr, ytr, groups=train.driver_id):
        m = HistGradientBoostingRegressor(max_iter=400, early_stopping=True,
                                          validation_fraction=0.1, random_state=SEED,
                                          monotonic_cst=mono_cst, **g)
        m.fit(Xtr.iloc[a], ytr.iloc[a])
        scores.append(np.sqrt(mean_squared_error(ytr.iloc[b], m.predict(Xtr.iloc[b]))))
    cv_rows.append({**g, "cv_rmse": float(np.mean(scores))})
cv_df = pd.DataFrame(cv_rows).sort_values("cv_rmse")
cv_df.to_csv(OUT / "cv_grid.csv", index=False)
best = cv_df.iloc[0][["learning_rate", "max_leaf_nodes", "l2_regularization"]].to_dict()
best["max_leaf_nodes"] = int(best["max_leaf_nodes"])
metrics["best_hparams"] = best

models = {
    "Vidurkio prognozė": None,
    "Baseline – LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
    "HGB be monotoniškumo": HistGradientBoostingRegressor(
        max_iter=400, early_stopping=True, random_state=SEED, **best),
    "Proposed – HGB + monotoniškumas": HistGradientBoostingRegressor(
        max_iter=400, early_stopping=True, random_state=SEED,
        monotonic_cst=mono_cst, **best),
}
preds, reg_res = {}, {}
for name, m in models.items():
    t0 = time.perf_counter()
    if m is None:
        p = np.full(len(yte), ytr.mean())
    else:
        m.fit(Xtr, ytr); p = m.predict(Xte)
    reg_res[name] = {**reg_metrics(yte.values, p),
                     "fit_s": float(time.perf_counter() - t0)}
    preds[name] = p
metrics["regression"] = reg_res
lr_model = models["Baseline – LinearRegression"]
hgb_free = models["HGB be monotoniškumo"]
hgb = models["Proposed – HGB + monotoniškumas"]
joblib.dump(hgb, MOD / "fuel_hgb_mono.joblib")
joblib.dump(lr_model, MOD / "fuel_linreg.joblib")

# kelionės lygiu (litrai per kelionę) – praktiškai svarbiausia
tl = test.assign(p_lr=preds["Baseline – LinearRegression"],
                 p_hgb=preds["Proposed – HGB + monotoniškumas"])
trip = tl.groupby("trip_id")[["fuel_l_100km", "p_lr", "p_hgb"]].mean()
metrics["trip_level"] = {
    "Baseline – LinearRegression": reg_metrics(trip.fuel_l_100km.values, trip.p_lr.values),
    "Proposed – HGB + monotoniškumas": reg_metrics(trip.fuel_l_100km.values, trip.p_hgb.values)}

# ---- 2 pav. prognozė vs tikrovė
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for a, (name, c) in zip(ax, [("Baseline – LinearRegression", C_BASE),
                             ("Proposed – HGB + monotoniškumas", C_PROP)]):
    a.scatter(yte, preds[name], s=3, alpha=.25, color=c)
    a.plot([0, 25], [0, 25], "k--", lw=.8)
    r = reg_res[name]
    a.set_title(f"{name}\nRMSE={r['RMSE']:.2f}  R²={r['R2']:.3f}")
    a.set_xlim(0, 25); a.set_ylim(0, 25)
    a.set_xlabel("Tikros sąnaudos, l/100 km")
ax[0].set_ylabel("Prognozuotos sąnaudos, l/100 km")
fig.tight_layout(); fig.savefig(FIG / "fig2_prognoze_vs_tikrove.png"); plt.close(fig)

# ---- 3 pav. XAI – permutacinė svarba
pi = permutation_importance(hgb, Xte, yte, n_repeats=8, random_state=SEED,
                            scoring="neg_root_mean_squared_error")
imp = pd.Series(pi.importances_mean, index=FEATURE_COLS).sort_values()
metrics["permutation_importance"] = imp.sort_values(ascending=False).round(4).to_dict()
lt = {"road_city": "kelias: miestas", "road_suburban": "kelias: priemiestis",
      "road_highway": "kelias: magistralė", "speed_limit_kmh": "leist. greitis",
      "grade_pct": "nuolydis %", "congestion": "spūstis", "temp_c": "temperatūra",
      "headwind_ms": "priešpriešinis vėjas", "precip": "krituliai", "load_kg": "krovinys",
      "avg_speed_kmh": "vid. greitis", "accel_rms": "greitėjimo intens. (RMS)",
      "harsh_acc_per_km": "staigūs greitėjimai/km", "harsh_brake_per_km": "staigūs stabdymai/km",
      "stops_per_km": "sustojimai/km", "idle_share": "tuščiosios eigos dalis",
      "avg_rpm": "vid. apsukos", "ac_on": "kondicionierius"}
controllable = {"avg_speed_kmh", "accel_rms", "harsh_acc_per_km", "harsh_brake_per_km",
                "idle_share", "avg_rpm", "ac_on", "load_kg"}
fig, ax = plt.subplots(figsize=(6.5, 4.6))
ax.barh([lt[i] for i in imp.index], imp.values,
        color=[C_PROP if i in controllable else C_BASE for i in imp.index])
ax.set_xlabel("RMSE padidėjimas sumaišius požymį (l/100 km)")
ax.set_title("Permutacinė požymių svarba (Proposed modelis)\n"
             "mėlyna – vairuotojo valdomi, pilka – aplinkos požymiai")
fig.tight_layout(); fig.savefig(FIG / "fig3_pozymiu_svarba.png"); plt.close(fig)

# ---- 4 pav. XAI – PDP
fig, ax = plt.subplots(1, 4, figsize=(11, 2.9))
PartialDependenceDisplay.from_estimator(
    hgb, Xte.sample(3000, random_state=SEED),
    ["avg_speed_kmh", "accel_rms", "avg_rpm", "temp_c"], ax=ax,
    line_kw={"color": C_PROP})
for a, t in zip(ax, ["Vid. greitis, km/h", "Greitėjimo RMS, m/s²",
                     "Vid. apsukos, aps./min", "Temperatūra, °C"]):
    a.set_xlabel(t)
for a in ax:
    a.set_ylabel("")
ax[0].set_ylabel("Poveikis prognozei, l/100 km")
fig.suptitle("Dalinės priklausomybės (PDP) grafikai – kaip modelis reaguoja į požymį", y=1.02)
fig.tight_layout(); fig.savefig(FIG / "fig4_pdp.png", bbox_inches="tight"); plt.close(fig)

# --------------------------------------- B. vairavimo stiliaus klasifikacija
train["speed_ratio"] = train.avg_speed_kmh / train.speed_limit_kmh
test["speed_ratio"] = test.avg_speed_kmh / test.speed_limit_kmh
STY = ["speed_ratio", "accel_rms", "harsh_acc_per_km", "harsh_brake_per_km",
       "idle_share", "avg_rpm", "avg_speed_kmh", "congestion",
       "road_city", "road_suburban", "road_highway"]
labels = ["eco", "normalus", "agresyvus"]
# Baseline: taisyklė – accel_rms tercilių slenksčiai iš mokymo aibės
t1, t2 = train.accel_rms.quantile([1 / 3, 2 / 3])
base_cls = np.where(test.accel_rms < t1, 0, np.where(test.accel_rms < t2, 1, 2))
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                     random_state=SEED, early_stopping=True)
clf.fit(train[STY], train.style_id)
prop_cls = clf.predict(test[STY])
proba = clf.predict_proba(test[STY])
joblib.dump(clf, MOD / "style_hgb.joblib")
# kelionės lygiu: vidutinės tikimybės per kelionę
tp = pd.DataFrame(proba, index=test.index).groupby(test.trip_id).mean()
trip_true = test.groupby("trip_id").style_id.first()
trip_pred = tp.values.argmax(1)
tb = pd.Series(base_cls, index=test.index).groupby(test.trip_id).agg(
    lambda s: s.value_counts().idxmax())
metrics["classification"] = {
    "Baseline – accel_rms taisyklė (segmentas)": dict(
        acc=float(accuracy_score(test.style_id, base_cls)),
        f1_macro=float(f1_score(test.style_id, base_cls, average="macro"))),
    "Proposed – HGB (segmentas)": dict(
        acc=float(accuracy_score(test.style_id, prop_cls)),
        f1_macro=float(f1_score(test.style_id, prop_cls, average="macro"))),
    "Baseline – accel_rms taisyklė (kelionė)": dict(
        acc=float(accuracy_score(trip_true, tb.loc[trip_true.index])),
        f1_macro=float(f1_score(trip_true, tb.loc[trip_true.index], average="macro"))),
    "Proposed – HGB (kelionė)": dict(
        acc=float(accuracy_score(trip_true, trip_pred)),
        f1_macro=float(f1_score(trip_true, trip_pred, average="macro"))),
    "thresholds_accel_rms": [float(t1), float(t2)],
    "report_segment": classification_report(test.style_id, prop_cls,
                                            target_names=labels, output_dict=True),
}
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
ConfusionMatrixDisplay.from_predictions(test.style_id, base_cls, display_labels=labels,
                                        ax=ax[0], cmap="Greys", colorbar=False,
                                        normalize="true", values_format=".2f")
ax[0].set_title("Baseline (accel_rms taisyklė)")
ConfusionMatrixDisplay.from_predictions(test.style_id, prop_cls, display_labels=labels,
                                        ax=ax[1], cmap="Blues", colorbar=False,
                                        normalize="true", values_format=".2f")
ax[1].set_title("Proposed (HGB klasifikatorius)")
for a in ax:
    a.set_xlabel("Prognozuota klasė"); a.set_ylabel("Tikroji klasė")
fig.tight_layout(); fig.savefig(FIG / "fig5_konfuzijos_matrica.png"); plt.close(fig)

# ---------------------- C. rekomendacijų patikra su fizikiniu orakulu
sample = test.sample(2500, random_state=SEED)
rows = []
for _, r in sample.iterrows():
    x = r[FEATURE_COLS].astype(float)
    acts = [a for a in _candidate_actions(x) if _safe(a[0], x, a[1])]
    if not acts:
        continue
    batch = pd.DataFrame([x] + [a[1] for a in acts])[FEATURE_COLS]
    true = fuel_l_per_100km(batch, vehicle_factor=r.vehicle_factor)
    p_lr, p_free, p_h = lr_model.predict(batch), hgb_free.predict(batch), hgb.predict(batch)
    for k, (aid, _, _) in enumerate(acts, start=1):
        rows.append(dict(seg=r.name, action=aid, style=r["style"],
                         true_saving=true[0] - true[k],
                         lr_saving=p_lr[0] - p_lr[k],
                         free_saving=p_free[0] - p_free[k],
                         hgb_saving=p_h[0] - p_h[k],
                         base_true=true[0]))
rec = pd.DataFrame(rows)
rec.to_csv(OUT / "recommendation_eval.csv", index=False)

def reco_quality(col):
    sign_ok = np.mean((rec[col] > 0.01) == (rec.true_saving > 0.01))
    nonphys = np.mean(rec[col] < -0.01)      # „veiksmas padidins sąnaudas“ – fiziškai neteisinga
    zero = np.mean(np.abs(rec[col]) <= 0.01)  # modelis „nemato“ efekto (plokščia sritis)
    mae = mean_absolute_error(rec.true_saving, rec[col])
    rho = spearmanr(rec[col], rec.true_saving).correlation
    # top-1 tikslumas: ar modelio pirmas pasiūlymas yra tikrai geriausias
    g = rec.groupby("seg")
    top_hit = np.mean([grp.loc[grp[col].idxmax(), "action"] ==
                       grp.loc[grp.true_saving.idxmax(), "action"]
                       for _, grp in g if len(grp) > 1])
    realized = np.mean([grp.loc[grp[col].idxmax(), "true_saving"] for _, grp in g])
    realized_pct = np.mean([grp.loc[grp[col].idxmax(), "true_saving"] /
                            grp.base_true.iloc[0] * 100 for _, grp in g])
    return dict(sign_accuracy=float(sign_ok), nonphysical_share=float(nonphys),
                zero_effect_share=float(zero),
                zero_by_action=(np.abs(rec[col]) <= 0.01).groupby(rec.action).mean().round(3).to_dict(),
                neg_by_action=(rec[col] < -0.01).groupby(rec.action).mean().round(3).to_dict(),
                saving_MAE=float(mae), spearman=float(rho), top1_hit=float(top_hit),
                realized_saving_l100=float(realized),
                realized_saving_pct=float(realized_pct))

metrics["recommendations"] = {
    "n_segments": int(rec.seg.nunique()), "n_action_evals": len(rec),
    "action_counts": rec.action.value_counts().to_dict(),
    "Baseline – LinearRegression": reco_quality("lr_saving"),
    "HGB be monotoniškumo": reco_quality("free_saving"),
    "Proposed – HGB + monotoniškumas": reco_quality("hgb_saving"),
    "oracle_best_possible_l100": float(rec.groupby("seg").true_saving.max().mean()),
    "by_action_true_saving_mean": rec.groupby("action").true_saving.mean().round(3).to_dict(),
    "by_action_hgb_saving_mean": rec.groupby("action").hgb_saving.mean().round(3).to_dict(),
    "by_action_lr_saving_mean": rec.groupby("action").lr_saving.mean().round(3).to_dict(),
}

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for a, (col, name, c) in zip(ax, [("lr_saving", "Baseline – LinearRegression", C_BASE),
                                  ("hgb_saving", "Proposed – HGB + monotoniškumas", C_PROP)]):
    a.scatter(rec.true_saving, rec[col], s=4, alpha=.3, color=c)
    lim = (-1, 4)
    a.plot(lim, lim, "k--", lw=.8); a.axhline(0, color="red", lw=.6)
    q = metrics["recommendations"][name]
    a.set_title(f"{name}\nkrypties tikslumas {q['sign_accuracy']*100:.1f} %, "
                f"Spearman ρ={q['spearman']:.2f}")
    a.set_xlim(*lim); a.set_ylim(-1.5, 4)
    a.set_xlabel("Tikras (fizikinis) sutaupymas, l/100 km")
ax[0].set_ylabel("Modelio prognozuotas sutaupymas, l/100 km")
fig.tight_layout(); fig.savefig(FIG / "fig6_rekomendaciju_patikra.png"); plt.close(fig)

# ------------------------------------------------------------ D. latencija
lat = []
xs = test.sample(1000, random_state=1)
for _, r in xs.iterrows():
    _, ms = recommend(hgb, r)
    lat.append(ms)
lat = np.array(lat)
metrics["latency_ms"] = dict(median=float(np.median(lat)), p95=float(np.percentile(lat, 95)),
                             max=float(lat.max()), requirement_ms=200)

# --------------------------------------- E. realaus laiko kelionės demo
agg_trips = test[test["style"] == "agresyvus"].groupby("trip_id").filter(
    lambda g: len(g) >= 28 and g.road_type.nunique() == 3)
demo_id = agg_trips.trip_id.iloc[0] if len(agg_trips) else test.trip_id.iloc[0]
demo = test[test.trip_id == demo_id].sort_values("seg_idx")
log, actual, followed, predicted = [], [], [], []
for _, r in demo.iterrows():
    shown, ms = recommend(hgb, r)
    x = r[FEATURE_COLS].astype(float)
    y = x.copy()
    for s in shown:                      # vairuotojas paklauso rodomų patarimų
        for k in FEATURE_COLS:
            if s["new_features"][k] != x[k]:
                y[k] = s["new_features"][k]
    tv = fuel_l_per_100km(pd.DataFrame([x, y]), vehicle_factor=r.vehicle_factor)
    actual.append(r.fuel_l_100km); predicted.append(hgb.predict(pd.DataFrame([x]))[0])
    followed.append(r.fuel_l_100km * tv[1] / tv[0])
    log.append(dict(km=int(r.seg_idx) + 1, kelias=r.road_type,
                    greitis=round(r.avg_speed_kmh), sanaudos=round(r.fuel_l_100km, 2),
                    patarimai=" | ".join(f"{s['text']} (−{s['saving_l100']:.2f} l/100 km)"
                                          for s in shown),
                    latency_ms=round(ms, 2)))
pd.DataFrame(log).to_csv(OUT / "demo_trip_log.csv", index=False)
metrics["demo_trip"] = dict(trip_id=int(demo_id), km=len(demo),
                            litres_actual=float(np.sum(actual) / 100),
                            litres_if_followed=float(np.sum(followed) / 100),
                            saving_pct=float((1 - np.sum(followed) / np.sum(actual)) * 100),
                            log_first=log[:6])
fig, ax = plt.subplots(figsize=(10, 3.4))
km = np.arange(1, len(demo) + 1)
ax.plot(km, actual, "-o", ms=3, color=C_ALT, label="Faktinės sąnaudos (agresyvus vairuotojas)")
ax.plot(km, followed, "-o", ms=3, color=C_PROP, label="Jei laikomasi AI patarimų (fizikinis įvertis)")
for i, l in enumerate(log):
    if l["patarimai"]:
        ax.axvspan(km[i] - .5, km[i] + .5, color="#ffe9a8", alpha=.5, lw=0)
for rt, c in [("miestas", "#f3f3f3")]:
    pass
ax.set_xlabel("Kelionės kilometras"); ax.set_ylabel("l/100 km")
ax.set_title(f"Realaus laiko demonstracija – kelionė #{demo_id}: "
             f"{metrics['demo_trip']['litres_actual']:.2f} l → "
             f"{metrics['demo_trip']['litres_if_followed']:.2f} l "
             f"(−{metrics['demo_trip']['saving_pct']:.1f} %); geltona – rodytas patarimas")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig7_realaus_laiko_demo.png"); plt.close(fig)

# ---------------- maršruto pasirinkimas (prieš kelionę): 3 alternatyvos
def mk_route(n_city, n_sub, n_hw, cong, temp=5.0, precip=0):
    seg = []
    for rt, n, lim in [(0, n_city, 50), (1, n_sub, 70), (2, n_hw, 110)]:
        for _ in range(n):
            seg.append(dict(road_city=int(rt == 0), road_suburban=int(rt == 1),
                            road_highway=int(rt == 2), speed_limit_kmh=lim, grade_pct=0.0,
                            congestion=cong[rt], temp_c=temp, headwind_ms=0.0,
                            precip=precip, load_kg=75, ac_on=0,
                            stops_per_km=[1.6, .6, .05][rt] * (0.5 + 2.2 * cong[rt])))
    return pd.DataFrame(seg)

routes = {"A: per miestą (trumpiausias, 18 km)": mk_route(14, 4, 0, [0.65, 0.4, 0.1]),
          "B: aplinkkelis (24 km)": mk_route(4, 6, 14, [0.4, 0.2, 0.1]),
          "C: priemiesčio keliai (21 km)": mk_route(5, 16, 0, [0.4, 0.25, 0.1])}
normal_style = dict(speed_factor=1.0, rpm_offset=550, accel_rms=0.5, harsh_acc_per_km=0.4,
                    harsh_brake_per_km=0.4, idle_share=0.07)
best_route, rres = recommend_route(hgb, routes, normal_style)
for name, seg in routes.items():               # patikra su fizikiniu modeliu
    s = seg.copy()
    s["avg_speed_kmh"] = s.speed_limit_kmh * (1 - 0.75 * s.congestion)
    s["avg_rpm"] = 1500 + 550 + 9 * s.avg_speed_kmh
    for k in ["accel_rms", "harsh_acc_per_km", "harsh_brake_per_km"]:
        s[k] = normal_style[k]
    s["idle_share"] = normal_style["idle_share"] * s.congestion * 3
    rres[name]["litres_physics"] = float(fuel_l_per_100km(s[FEATURE_COLS]).sum() / 100)
metrics["route_choice"] = dict(best=best_route, routes=rres)

# ---- 8 pav. architektūros schema
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off")
def box(x, y, w, h, t, c):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="#333", lw=.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
def arr(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1))
box(0.0, 3.0, 2.0, 0.8, "Maršrutas\n(GPS, žemėlapis, nuolydis)", "#e8f0fa")
box(0.0, 2.0, 2.0, 0.8, "Eismas\n(spūsčių API)", "#e8f0fa")
box(0.0, 1.0, 2.0, 0.8, "Oras\n(OpenWeatherMap API)", "#e8f0fa")
box(0.0, 0.0, 2.0, 0.8, "Automobilis (OBD-II / CAN)\ngreitis, apsukos, IMU", "#e8f0fa")
box(2.6, 1.2, 2.0, 1.6, "Požymių formavimas\n1 km slenkantis langas\n(18 požymių)", "#fff4e0")
box(5.2, 2.3, 2.1, 1.0, "Sąnaudų modelis\nHGB + monotoniškumas", "#dbe9f6")
box(5.2, 0.7, 2.1, 1.0, "Stiliaus klasifikatorius\n(eco/norm./agres.)", "#dbe9f6")
box(7.9, 1.5, 2.0, 1.3, "Rekomendacijų variklis\nwhat-if + saugos\nfiltras (<200 ms)", "#e3f1df")
box(7.9, 0.0, 2.0, 0.9, "Vairuotojui: balsu /\nekrane (tik patarimas)", "#f6e3e3")
for y in (3.4, 2.4, 1.4, 0.4):
    arr(2.0, y, 2.6, 2.0)
arr(4.6, 2.2, 5.2, 2.8); arr(4.6, 1.8, 5.2, 1.2)
arr(7.3, 2.8, 7.9, 2.3); arr(7.3, 1.2, 7.9, 1.9); arr(8.9, 1.5, 8.9, 0.9)
ax.set_xlim(-0.1, 10); ax.set_ylim(-0.2, 3.9)
fig.tight_layout(); fig.savefig(FIG / "fig8_architektura.png"); plt.close(fig)

with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps({k: v for k, v in metrics.items()
                  if k not in ("permutation_importance",)}, ensure_ascii=False,
                 indent=1, default=str)[:9000])
