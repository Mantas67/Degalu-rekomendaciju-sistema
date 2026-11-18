"""
Paprasta grafinė sąsaja (Streamlit).
Paleidimas:  streamlit run src/app_streamlit.py
Vartotojas slankikliais nustato dabartinę situaciją ir mato AI patarimus.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from recommender import recommend

ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(ROOT / "models" / "fuel_hgb_mono.joblib")

st.title("⛽ Degalų taupymo AI patarėjas")
st.caption("Informacinė sistema – tik pataria, automobilio nevaldo.")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Maršrutas ir eismas")
    road = st.selectbox("Kelio tipas", ["miestas", "priemiestis", "magistralė"])
    limit = st.selectbox("Leistinas greitis, km/h", [50, 70, 90, 110, 130],
                         index={"miestas": 0, "priemiestis": 1, "magistralė": 3}[road])
    grade = st.slider("Nuolydis, %", -4.0, 4.0, 0.0, 0.5)
    congestion = st.slider("Spūstis (0–1)", 0.0, 1.0, 0.3, 0.05)
    stops = st.slider("Sustojimai per km", 0, 8, 1)
with c2:
    st.subheader("Oras ir krovinys")
    temp = st.slider("Temperatūra, °C", -20, 35, 8)
    precip = st.radio("Krituliai", ["nėra", "lietus", "sniegas"], horizontal=True)
    wind = st.slider("Priešpriešinis vėjas, m/s", -10, 10, 0)
    load = st.slider("Krovinys, kg", 0, 400, 75, 25)
    ac = st.checkbox("Kondicionierius įjungtas", value=False)
with c3:
    st.subheader("Vairavimo stilius")
    speed = st.slider("Vidutinis greitis, km/h", 5, 160, 45)
    accel = st.slider("Greitėjimo intensyvumas (RMS), m/s²", 0.1, 1.2, 0.5, 0.05)
    harsh_a = st.slider("Staigūs greitėjimai / km", 0, 6, 1)
    harsh_b = st.slider("Staigūs stabdymai / km", 0, 6, 1)
    idle = st.slider("Tuščiosios eigos dalis", 0.0, 0.6, 0.08, 0.01)
    rpm = st.slider("Vidutinės apsukos, aps./min", 1200, 4500, 2500, 50)

x = pd.Series(dict(
    road_city=int(road == "miestas"), road_suburban=int(road == "priemiestis"),
    road_highway=int(road == "magistralė"), speed_limit_kmh=limit, grade_pct=grade,
    congestion=congestion, temp_c=temp, headwind_ms=wind,
    precip=["nėra", "lietus", "sniegas"].index(precip), load_kg=load,
    avg_speed_kmh=speed, accel_rms=accel, harsh_acc_per_km=harsh_a,
    harsh_brake_per_km=harsh_b, stops_per_km=stops, idle_share=idle,
    avg_rpm=rpm, ac_on=int(ac)))

recs, ms = recommend(model, x)
cur = float(model.predict(pd.DataFrame([x]))[0])
st.metric("Prognozuojamos sąnaudos", f"{cur:.1f} l/100 km", help=f"Skaičiavimas: {ms:.1f} ms")
if recs:
    for r in recs:
        st.success(f"{r['text']}  \n**Sutaupytumėte ≈ {r['saving_l100']:.2f} l/100 km**")
else:
    st.info("Vairuojate taupiai – papildomų patarimų nėra. 👍")
