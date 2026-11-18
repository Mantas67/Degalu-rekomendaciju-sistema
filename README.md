# Degalų sąnaudų optimizavimo rekomendacijų sistema (tema Nr. 30)

AI sistema, kuri pagal **maršrutą, eismo spūstis, orą ir vairavimo stilių** prognozuoja
automobilio degalų sąnaudas ir **realiu laiku** (≈5 ms) pataria, kaip jas sumažinti.
Sistema yra **informacinė** – ji tik pataria ir niekada nevaldo automobilio.

## Struktūra
```
src/physics.py          fizikinis sąnaudų modelis (duomenų generavimas + rekomendacijų patikra)
src/generate_data.py    sintetinių telematikos duomenų generatorius (19 607 segmentai)
src/recommender.py      rekomendacijų variklis (what-if + saugos filtras) ir maršruto parinkimas
src/train_evaluate.py   visi eksperimentai: Baseline vs Proposed, XAI, latencija, grafikai
src/demo_realtime.py    realaus laiko demonstracija komandinėje eilutėje
src/app_streamlit.py    grafinė sąsaja (Streamlit)
data/ models/ outputs/ figures/   sugeneruojami paleidus skriptus
```

## Paleidimas
```bash
pip install -r requirements.txt
python src/generate_data.py --out data/segments.csv   # ~2 s
python src/train_evaluate.py                          # ~1,5 min (CPU), sukuria figures/ ir outputs/metrics.json
python src/demo_realtime.py --trip 97 --delay 0.3     # realaus laiko demo
streamlit run src/app_streamlit.py                    # grafinė sąsaja
```
Arba su Docker: `docker build -t fuel-ai . && docker run -p 8501:8501 fuel-ai`

Visi atsitiktinumai fiksuoti (`seed=42`), todėl rezultatai atkartojami.
Nenaudojamas joks komercinis / uždaro kodo API, internetas paleidimui nereikalingas.

## Pagrindiniai rezultatai (testavimo aibė – 12 nematytų vairuotojų)
| Užduotis | Baseline | Proposed |
|---|---|---|
| Sąnaudų regresija, R² | 0,707 (LinearRegression) | **0,905** (HGB + monotoniškumas) |
| Sąnaudų regresija, RMSE l/100 km | 2,12 | **1,20** |
| Stiliaus klasifikacija, F1 (segmentas) | 0,69 (taisyklė) | **0,80** (HGB) |
| Stiliaus klasifikacija, F1 (kelionė) | 0,78 | **0,85** |
| Fiziškai klaidingi patarimai („padidins sąnaudas“) | 20,7 % | **0,2 %** |
| Rekomendacijos latencija (p95) | – | **≈6 ms** (< 200 ms) |

⚠️ Duomenys yra **sintetiniai** (fizikinis modelis). Norint naudoti realius duomenis
(pvz. VED – Vehicle Energy Dataset, savo OBD-II įrašus), pakanka pakeisti `data/segments.csv`
tais pačiais stulpeliais – modelių kodas nesikeičia.
