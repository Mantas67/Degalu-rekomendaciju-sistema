"""
Realaus laiko degalų taupymo rekomendacijų variklis.

Principas („what-if“ / kontrafaktinė analizė):
  1. Paimamas einamojo ~1 km lango požymių vektorius x (maršrutas, eismas,
     oras, vairavimo stilius).
  2. Sugeneruojami kandidatiniai veiksmai, keičiantys TIK vairuotojo
     valdomus požymius (greitis, greitėjimas, apsukos, tuščiosios eigos
     laikas, kondicionierius, krovinys).
  3. Visi kandidatai įvertinami VIENU modelio predict() kvietimu
     (todėl latencija – kelios milisekundės).
  4. Saugos filtras atmeta veiksmus, kurie būtų nesaugūs ar neteisėti
     (pvz. greitis virš leistino, per lėtas važiavimas magistralėje,
     greičio mažinimas spūstyje, kondicionieriaus išjungimas karštyje).
  5. Grąžinamos 1–2 naudingiausios rekomendacijos lietuvių kalba.

Sistema yra INFORMACINĖ – ji tik pataria ir niekada nevaldo automobilio.
"""
import time
import numpy as np
import pandas as pd

from physics import FEATURE_COLS

MIN_SAVING_L100 = 0.15   # mažesnis sutaupymas vairuotojui nerodomas


def _candidate_actions(x):
    """Grąžina sąrašą (veiksmo_id, pakeistas_vektorius, tekstas)."""
    acts = []
    v, lim = x["avg_speed_kmh"], x["speed_limit_kmh"]

    # 1) Greitis: jei viršijamas leistinas arba važiuojama > 100 km/h
    if v > lim + 2 or v > 100:
        new_v = max(min(v - 10, lim), 0.85 * lim)
        if new_v < v - 3:
            y = x.copy(); y["avg_speed_kmh"] = new_v
            y["avg_rpm"] = x["avg_rpm"] - 9 * (v - new_v)
            acts.append(("greitis", y,
                         f"Sumažinkite greitį iki ~{new_v:.0f} km/h "
                         f"(dabar {v:.0f} km/h)."))

    # 2) Švelnesnis greitėjimas
    if x["accel_rms"] > 0.45 or x["harsh_acc_per_km"] >= 1:
        y = x.copy()
        y["accel_rms"] = max(0.25, x["accel_rms"] * 0.70)
        y["harsh_acc_per_km"] = 0.0
        y["harsh_brake_per_km"] = max(0.0, x["harsh_brake_per_km"] - 1)
        acts.append(("greitejimas", y,
                     "Greitėkite tolygiau – spauskite akceleratorių švelniau "
                     "ir anksčiau numatykite stabdymą."))

    # 3) Ankstesnis pavarų perjungimas
    if x["avg_rpm"] > 2400:
        y = x.copy(); y["avg_rpm"] = max(1800, x["avg_rpm"] - 500)
        acts.append(("apsukos", y,
                     f"Perjunkite aukštesnę pavarą anksčiau – laikykite apsukas "
                     f"apie {y['avg_rpm']:.0f} aps./min."))

    # 4) Tuščiosios eigos mažinimas (tik kai stovima ilgiau)
    if x["idle_share"] > 0.08:
        y = x.copy(); y["idle_share"] = x["idle_share"] * 0.4
        acts.append(("tuscioji_eiga", y,
                     "Ilgiau stovėdami (>60 s, pvz. prie pervažos) išjunkite "
                     "variklį arba naudokite Start-Stop."))

    # 5) Kondicionierius – tik kai lauke nekaršta
    if x["ac_on"] == 1 and x["temp_c"] <= 24:
        y = x.copy(); y["ac_on"] = 0
        acts.append(("kondicionierius", y,
                     "Lauke vėsu – kondicionierių galite išjungti, naudokite "
                     "vėdinimą."))
    return acts


def _safe(action_id, x, y):
    """Saugos taisyklės (sprendimo kritiškumas – informacinis)."""
    if action_id == "greitis":
        if x["congestion"] > 0.5:          # spūstyje greičio nepatariame
            return False
        if x["precip"] == 2:               # sniege greičio sprendžia vairuotojas
            return False
        if y["avg_speed_kmh"] < 0.85 * x["speed_limit_kmh"] and x["road_highway"]:
            return False                   # per lėtas važiavimas magistralėje
    return True


def recommend(model, x_row, top_k=2, return_all=False):
    """x_row – pd.Series su FEATURE_COLS. Grąžina rekomendacijų sąrašą."""
    t0 = time.perf_counter()
    x = x_row[FEATURE_COLS].astype(float).copy()
    acts = [a for a in _candidate_actions(x) if _safe(a[0], x, a[1])]
    if not acts:
        return [], (time.perf_counter() - t0) * 1000
    batch = pd.DataFrame([x] + [a[1] for a in acts])[FEATURE_COLS]
    preds = model.predict(batch)
    base = preds[0]
    out = []
    for (aid, y, txt), p in zip(acts, preds[1:]):
        out.append(dict(action=aid, text=txt, current_l100=float(base),
                        predicted_l100=float(p), saving_l100=float(base - p),
                        new_features=y))
    out.sort(key=lambda r: -r["saving_l100"])
    latency_ms = (time.perf_counter() - t0) * 1000
    if return_all:
        return out, latency_ms
    shown = [r for r in out if r["saving_l100"] >= MIN_SAVING_L100][:top_k]
    return shown, latency_ms


def recommend_route(model, routes, driver_style):
    """Maršruto pasirinkimas prieš kelionę.
    routes: dict {pavadinimas: DataFrame su segmentų maršruto/eismo/oro požymiais}
    driver_style: dict su vairuotojo tipiniais stiliaus požymiais
    (accel_rms, harsh_*, idle_share, rpm_offset, speed_factor).
    Grąžina kiekvieno maršruto prognozuojamus litrus ir rekomenduojamą."""
    res = {}
    for name, seg in routes.items():
        seg = seg.copy()
        flow = seg["speed_limit_kmh"] * (1 - 0.75 * seg["congestion"])
        seg["avg_speed_kmh"] = flow * driver_style["speed_factor"]
        seg["avg_rpm"] = 1500 + driver_style["rpm_offset"] + 9 * seg["avg_speed_kmh"]
        for k in ["accel_rms", "harsh_acc_per_km", "harsh_brake_per_km"]:
            seg[k] = driver_style[k]
        seg["idle_share"] = driver_style["idle_share"] * seg["congestion"] * 3
        l100 = model.predict(seg[FEATURE_COLS])
        litres = float(np.sum(l100 / 100.0))          # segmentai po 1 km
        minutes = float(np.sum(60.0 / seg["avg_speed_kmh"]))
        res[name] = dict(km=len(seg), litres=litres, minutes=minutes)
    best = min(res, key=lambda k: res[k]["litres"])
    return best, res
