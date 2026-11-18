"""
Realaus laiko demonstracija komandinėje eilutėje.
Imituoja kelionę: kas 1 km gaunamas naujas telematikos langas,
sistema per kelias ms grąžina patarimus.

Paleidimas:  python src/demo_realtime.py --trip 97 --delay 0.3
"""
import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from recommender import recommend

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", type=int, default=None)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    model = joblib.load(ROOT / "models" / "fuel_hgb_mono.joblib")
    df = pd.read_csv(ROOT / "data" / "segments.csv")
    tid = args.trip or int(df[df["style"] == "agresyvus"].trip_id.iloc[0])
    trip = df[df.trip_id == tid].sort_values("seg_idx")
    print(f"Kelionė #{tid}: {len(trip)} km, vairuotojas #{trip.driver_id.iloc[0]}\n")
    for _, r in trip.iterrows():
        recs, ms = recommend(model, r)
        head = (f"[{int(r.seg_idx)+1:>2} km | {r.road_type:<11} | "
                f"{r.avg_speed_kmh:5.0f} km/h | {r.fuel_l_100km:5.1f} l/100 km | {ms:4.1f} ms]")
        if recs:
            for i, x in enumerate(recs):
                print(head if i == 0 else " " * len(head),
                      f"-> {x['text']} (≈ −{x['saving_l100']:.2f} l/100 km)")
        else:
            print(head, "-> vairuojate taupiai 👍")
        time.sleep(args.delay)
