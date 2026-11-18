"""
Sintetinių telematikos duomenų generatorius.

Kiekvienas įrašas = 1 km kelio atkarpa (segmentas) su maršruto, eismo,
oro ir vairavimo stiliaus požymiais bei degalų sąnaudomis (l/100 km).

Struktūra: vairuotojas -> kelionės -> segmentai.
Kiekvienas vairuotojas turi latentinį agresyvumą a ∈ [0, 1] ir stiliaus
klasę (eco / normalus / agresyvus), bei savo automobilio efektyvumo
koeficientą (nestebimas modeliui – imituoja skirtingus automobilius).

Paleidimas:  python src/generate_data.py --out data/segments.csv
"""
import argparse
import numpy as np
import pandas as pd

from physics import fuel_l_per_100km, FEATURE_COLS

STYLE_NAMES = ["eco", "normalus", "agresyvus"]
STYLE_CENTER = [0.20, 0.50, 0.80]


def simulate(n_drivers=60, trips_per_driver=12, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    trip_id = 0
    for drv in range(n_drivers):
        style = drv % 3
        a = float(np.clip(rng.normal(STYLE_CENTER[style], 0.12), 0.0, 1.0))
        veh_factor = float(rng.normal(1.0, 0.06))
        idle_habit = float(np.clip(rng.normal(0.4 + 0.5 * a, 0.15), 0.05, 1.2))

        for _ in range(trips_per_driver):
            trip_id += 1
            # --- kelionės lygio sąlygos (oras, paros metas, krovinys) ---
            month = rng.integers(1, 13)
            temp = rng.normal(7 + 11 * -np.cos((month - 1) / 12 * 2 * np.pi), 4.5)
            if temp < 0 and rng.random() < 0.35:
                precip = 2          # sniegas
            elif rng.random() < 0.28:
                precip = 1          # lietus / šlapia danga
            else:
                precip = 0
            wind = rng.normal(0, 3.5)
            rush = rng.random() < 0.35
            load = float(rng.choice([0, 75, 150, 300], p=[0.45, 0.30, 0.17, 0.08]))
            ac_trip = (temp > 22 and rng.random() < 0.8) or rng.random() < 0.08

            n_seg = int(rng.integers(15, 40))
            # maršruto profilis: miestas -> priemiestis -> magistralė ir atgal
            route_mix = rng.dirichlet([2.0, 1.2, 1.5])
            road_types = rng.choice(3, size=n_seg, p=route_mix)
            road_types.sort()
            if rng.random() < 0.5:
                road_types = road_types[::-1]

            for s in range(n_seg):
                rt = int(road_types[s])
                limit = [50, 70, 110 if rng.random() < 0.6 else 130][rt]
                grade = rng.normal(0, [0.6, 0.9, 0.7][rt])
                cong_base = [0.35, 0.18, 0.10][rt] + (0.30 if rush else 0.0)
                congestion = float(np.clip(rng.normal(cong_base, 0.15), 0, 1))
                headwind = wind * rng.uniform(-1, 1)

                flow = limit * (1 - 0.75 * congestion)
                weather_slow = [1.0, 0.95, 0.80][precip]
                v = flow * weather_slow * (0.90 + 0.22 * a + rng.normal(0, 0.05))
                v = float(np.clip(v, 6, limit * 1.25))

                stops = rng.poisson([1.6, 0.6, 0.05][rt] * (0.5 + 2.2 * congestion))
                accel_rms = max(0.12, 0.22 + 0.55 * a + 0.35 * congestion * (rt == 0)
                                + rng.normal(0, 0.07))
                harsh_acc = rng.poisson((0.05 + 1.8 * a ** 2) * [1.4, 1.0, 0.5][rt])
                harsh_brk = rng.poisson((0.05 + 1.6 * a ** 2) * [1.4, 1.0, 0.5][rt]
                                        + 0.4 * congestion)
                idle_share = float(np.clip(
                    (0.04 + 0.30 * congestion) * [1.6, 0.8, 0.2][rt] * idle_habit
                    + rng.normal(0, 0.02), 0, 0.6))
                rpm = 1500 + 1100 * a + 9 * v + rng.normal(0, 150)
                ac_on = int(ac_trip)

                rows.append(dict(
                    driver_id=drv, trip_id=trip_id, seg_idx=s,
                    style=STYLE_NAMES[style], style_id=style, aggressiveness=a,
                    road_type=["miestas", "priemiestis", "magistralė"][rt],
                    road_city=int(rt == 0), road_suburban=int(rt == 1),
                    road_highway=int(rt == 2),
                    speed_limit_kmh=limit, grade_pct=grade, congestion=congestion,
                    temp_c=temp, headwind_ms=headwind, precip=precip,
                    load_kg=load, avg_speed_kmh=v, accel_rms=accel_rms,
                    harsh_acc_per_km=harsh_acc, harsh_brake_per_km=harsh_brk,
                    stops_per_km=stops, idle_share=idle_share, avg_rpm=rpm,
                    ac_on=ac_on, vehicle_factor=veh_factor,
                ))

    df = pd.DataFrame(rows)
    clean = fuel_l_per_100km(df[FEATURE_COLS], vehicle_factor=df["vehicle_factor"])
    noise = rng.lognormal(0, 0.06, size=len(df))       # matavimo triukšmas ~6 %
    df["fuel_l_100km"] = clean * noise
    # realistiškumo dėlei – 0,3 % jutiklio klaidų (outlierių), kurias valysime
    bad = rng.random(len(df)) < 0.003
    df.loc[bad, "fuel_l_100km"] *= rng.uniform(2.5, 4.0, bad.sum())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/segments.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    d = simulate(seed=args.seed)
    d.to_csv(args.out, index=False)
    print(f"Sugeneruota {len(d)} segmentų, {d.trip_id.nunique()} kelionių, "
          f"{d.driver_id.nunique()} vairuotojų -> {args.out}")
    print(d.groupby("road_type")["fuel_l_100km"].describe().round(2))
