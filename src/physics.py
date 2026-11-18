"""
Fizikinis degalų sąnaudų modelis (benzininis C klasės lengvasis automobilis).

Šis modulis naudojamas dviem tikslais:
  1) sintetiniams duomenims generuoti (generate_data.py);
  2) kaip „orakulas“ rekomendacijų patikrai (evaluate.py): pritaikius
     rekomenduotą veiksmą, galima apskaičiuoti TIKRĄ (fizikinį) sutaupymą
     ir palyginti jį su AI modelio prognozuotu sutaupymu.

Modelis remiasi ratų traukos energijos balansu kilometrui:
  E = E_riedėjimo + E_aerodinaminė + E_nuolydžio + E_kinetinė (PKE)
ir variklio naudingumo koeficientu, kuris priklauso nuo apsukų ir temperatūros.
"""
import numpy as np

G = 9.81            # m/s²
RHO_AIR = 1.2       # kg/m³
CDA = 0.60          # m² (Cd*A, kompaktinis sedanas)
CRR_DRY = 0.011     # riedėjimo pasipriešinimo koef. sausame kelyje
LHV_J_PER_L = 32e6  # benzino energija, J/l
ETA_DRIVETRAIN = 0.90
ETA_ENGINE_MAX = 0.30
IDLE_L_PER_H = 0.75
AC_L_PER_H = 0.35
BASE_MASS = 1400.0  # kg (automobilis + vairuotojas)

FEATURE_COLS = [
    # maršrutas
    "road_city", "road_suburban", "road_highway", "speed_limit_kmh", "grade_pct",
    # eismas
    "congestion",
    # oras
    "temp_c", "headwind_ms", "precip",
    # transporto priemonė / krovinys
    "load_kg",
    # vairavimo stilius (valdomi vairuotojo)
    "avg_speed_kmh", "accel_rms", "harsh_acc_per_km", "harsh_brake_per_km",
    "stops_per_km", "idle_share", "avg_rpm", "ac_on",
]


def pke_per_m(avg_speed_kmh, stops_per_km, accel_rms, harsh_acc_per_km):
    """Teigiamoji kinetinė energija (PKE, m/s²) – kiek kartų per kilometrą
    automobilis „įsibėgėja“. Didėja dėl sustojimų, agresyvaus greitėjimo."""
    v = np.asarray(avg_speed_kmh) / 3.6
    return (np.asarray(stops_per_km) * v ** 2 / 1000.0
            + 0.10 * np.asarray(accel_rms)
            + 0.045 * np.asarray(harsh_acc_per_km))


def fuel_l_per_100km(df, vehicle_factor=1.0):
    """Deterministinės degalų sąnaudos (l/100 km) pagal požymių lentelę."""
    v = np.maximum(df["avg_speed_kmh"].to_numpy() / 3.6, 1.0)
    m = BASE_MASS + df["load_kg"].to_numpy()
    precip = df["precip"].to_numpy()
    crr = CRR_DRY * np.where(precip == 1, 1.15, np.where(precip == 2, 1.35, 1.0))
    grade = df["grade_pct"].to_numpy() / 100.0
    wind = df["headwind_ms"].to_numpy()

    d = 1000.0  # 1 km
    e_roll = m * G * crr * d
    e_aero = 0.5 * RHO_AIR * CDA * np.maximum(v + wind, 0) ** 2 * d
    e_grade = m * G * grade * d
    pke = pke_per_m(df["avg_speed_kmh"], df["stops_per_km"],
                    df["accel_rms"], df["harsh_acc_per_km"])
    e_kin = m * pke * d
    e_wheel = np.maximum(e_roll + e_aero + e_grade + e_kin, 0.15 * e_roll)

    rpm = df["avg_rpm"].to_numpy()
    temp = df["temp_c"].to_numpy()
    eta = ETA_ENGINE_MAX * (1 - 0.00012 * np.maximum(rpm - 1800, 0)) \
        * (1 - 0.006 * np.maximum(15 - temp, 0))
    fuel_motion = e_wheel / (eta * ETA_DRIVETRAIN * LHV_J_PER_L)  # l/km

    moving_time_s = d / v
    idle = np.clip(df["idle_share"].to_numpy(), 0, 0.9)
    idle_time_s = idle / (1 - idle) * moving_time_s
    fuel_idle = IDLE_L_PER_H * idle_time_s / 3600.0
    fuel_ac = AC_L_PER_H * df["ac_on"].to_numpy() * (moving_time_s + idle_time_s) / 3600.0

    return (fuel_motion + fuel_idle + fuel_ac) * 100.0 * vehicle_factor
