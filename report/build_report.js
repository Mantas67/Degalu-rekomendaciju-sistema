// Ataskaitos generatorius: skaičiai imami iš outputs/metrics.json ir outputs/failures.json
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat,
  PageBreak, Footer, PageNumber, TableOfContents,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "outputs/metrics.json"), "utf8"));
const F = JSON.parse(fs.readFileSync(path.join(ROOT, "outputs/failures.json"), "utf8"));

// ---------- pagalbinės funkcijos
const n = (x, d = 2) => Number(x).toFixed(d).replace(".", ",");
const pct = (x, d = 1) => n(x * 100, d) + " %";
const FONT = "Arial";
const W = 9026; // A4 turinio plotis DXA (1" paraštės)

function runs(text, opts = {}) {
  // **paryškinta** ir _kursyvas_ žymėjimas
  const out = [];
  const re = /(\*\*[^*]+\*\*|~~[^~]+~~)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else out.push(new TextRun({ text: t.slice(2, -2), italics: true, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), spacing: { after: 120, line: 288 },
  alignment: o.align || AlignmentType.JUSTIFIED, ...o.para });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true });
const H1n = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const B = (t, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(t),
  spacing: { after: 60, line: 276 } });
const N = (t, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t),
  spacing: { after: 60, line: 276 } });
const CODE = (t) => new Paragraph({ children: [new TextRun({ text: t, font: "Consolas", size: 18 })],
  spacing: { after: 0 }, shading: { type: ShadingType.CLEAR, fill: "F3F3F3" } });
const CAP = (t) => new Paragraph({ children: runs(t, { italics: true, size: 18, color: "444444" }),
  alignment: AlignmentType.CENTER, spacing: { after: 200 } });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function IMG(file, widthPx = 600) {
  const f = path.join(ROOT, "figures", file);
  const s = pngSize(f);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(f),
      transformation: { width: widthPx, height: Math.round(widthPx * s.h / s.w) } })] });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
const borders = { top: border, bottom: border, left: border, right: border };
function TBL(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head, hl) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    shading: head ? { type: ShadingType.CLEAR, fill: "1F4E79" } :
      hl ? { type: ShadingType.CLEAR, fill: "E2EFDA" } : undefined,
    children: [new Paragraph({ children: runs(String(t), { size: 18, bold: head, color: head ? "FFFFFF" : undefined }),
      alignment: i === 0 ? AlignmentType.LEFT : (opts.alignLeft ? AlignmentType.LEFT : AlignmentType.CENTER) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false,
        opts.highlight && opts.highlight.includes(ri))) }))],
  });
}
const SP = () => new Paragraph({ children: [], spacing: { after: 80 } });

// ---------- skaičiai
const R = M.regression, RB = R["Baseline – LinearRegression"], RF = R["HGB be monotoniškumo"],
  RP = R["Proposed – HGB + monotoniškumas"], RM = R["Vidurkio prognozė"];
const TL = M.trip_level;
const C = M.classification;
const CB = C["Baseline – accel_rms taisyklė (segmentas)"], CP = C["Proposed – HGB (segmentas)"];
const CBt = C["Baseline – accel_rms taisyklė (kelionė)"], CPt = C["Proposed – HGB (kelionė)"];
const rep = C.report_segment;
const REC = M.recommendations;
const QB = REC["Baseline – LinearRegression"], QF = REC["HGB be monotoniškumo"], QP = REC["Proposed – HGB + monotoniškumas"];
const L = M.latency_ms, D = M.demo_trip, RC = M.route_choice;
const PI = M.permutation_importance;
const actLt = { greitejimas: "Švelnesnis greitėjimas", apsukos: "Ankstesnis pavarų perjungimas",
  tuscioji_eiga: "Tuščiosios eigos mažinimas", greitis: "Greičio mažinimas", kondicionierius: "Kondicionieriaus išjungimas" };
const acts = ["greitejimas", "greitis", "kondicionierius", "apsukos", "tuscioji_eiga"];
const rpmPick = F.pick_counts.apsukos, rpmCorrect = F.crosstab.apsukos.apsukos;
const routeNames = Object.keys(RC.routes);

// ---------- turinys
const body = [];

// I. TITULINIS LAPAS
body.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400, after: 80 },
    children: [new TextRun({ text: "LIETUVOS INŽINERIJOS KOLEGIJA", bold: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1400 },
    children: [new TextRun({ text: "Dirbtinio intelekto metodai – individualus semestro projektas", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "TEMA Nr. 30 (VI. Telematika, apdraudimas ir flotės valdymas)", size: 22, color: "1F4E79", bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
    children: [new TextRun({ text: "DEGALŲ SĄNAUDŲ OPTIMIZAVIMO REKOMENDACIJŲ SISTEMA", bold: true, size: 36 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1600 },
    children: [new TextRun({ text: "Maršruto, eismo spūsčių, oro sąlygų ir vairavimo stiliaus analize grįstas realaus laiko patarėjas, mažinantis automobilio degalų sąnaudas", italics: true, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 60 }, children: [new TextRun({ text: "Autorius: Mantas Šarakojis", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 60 }, children: [new TextRun({ text: "Grupė: [įrašykite]", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 60 }, children: [new TextRun({ text: "Dėstytojas(-a): [įrašykite]", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 1200 }, children: [new TextRun({ text: "GitHub: [įrašykite nuorodą]", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Kaunas, 2026", size: 22 })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Darbo pasidalijimo lentelė", bold: true, size: 24 })], spacing: { after: 120 } }),
  TBL(["Narys", "Atsakomybė", "Skyriai"], [
    ["Mantas Šarakojis", "Visa darbo dalis (individualus darbas): duomenų generatorius, modeliai, rekomendacijų variklis, eksperimentai, sąsaja, ataskaita", "I–X"],
  ], [2400, 5226, 1400]),
  SP(),
  new Paragraph({ children: [new TextRun({ text: "Turinys", bold: true, size: 24 })], spacing: { before: 240, after: 120 } }),
  new TableOfContents("Turinys", { hyperlink: true, headingStyleRange: "1-2" }),
  P("~~(Atidarius dokumentą Word programoje: dešiniu pelės klavišu ant turinio → „Atnaujinti lauką“, jei turinys neatsinaujina automatiškai.)~~", { run: { size: 18 } }),
);

// II. SANTRAUKA
body.push(H1("II. Santrauka"),
  P("**Problema.** Tas pats automobilis, važiuodamas tuo pačiu maršrutu, gali sudeginti labai skirtingą degalų kiekį – priklausomai nuo eismo spūsčių, oro sąlygų ir, svarbiausia, nuo vairuotojo elgesio (greičio, greitėjimo, pavarų perjungimo, tuščiosios eigos). Vairuotojas paprastai nežino, kuris jo veiksmas konkrečiu momentu kainuoja daugiausia degalų."),
  P(`**Metodas.** Sukurta informacinė (tik patarianti) AI sistema, kuri kas 1 km iš 18 požymių (maršrutas, eismas, oras, krovinys, vairavimo stilius) prognozuoja degalų sąnaudas ir „what-if“ analize įvertina, kuris vairuotojo valdomas veiksmas labiausiai sumažintų sąnaudas. Patarimai praeina saugos filtrą (niekada nesiūlomas greitis virš leistino, greičio mažinimas spūstyje ar sniege ir pan.). Kadangi realių duomenų rinkinio parsisiųsti nebuvo galimybės, sukurtas fizikiniu energijos balansu grįstas sintetinių duomenų generatorius (${M.data.raw_segments.toLocaleString("lt-LT")} kelio segmentų, ${M.data.trips} kelionių, ${M.data.drivers} vairuotojų). Lyginti Baseline (LinearRegression; accel_rms taisyklė) ir Proposed (HistGradientBoosting su monotoniškumo apribojimais) modeliai.`),
  P(`**Rezultatai.** Su ${M.split.test_drivers} niekada nematytų vairuotojų duomenimis Proposed modelis prognozuoja sąnaudas su R² = ${n(RP.R2, 3)} (Baseline – ${n(RB.R2, 3)}), RMSE sumažėja nuo ${n(RB.RMSE)} iki ${n(RP.RMSE)} l/100 km. Vairavimo stiliaus klasifikatoriaus makro F1 – ${n(CP.f1_macro)} (segmento lygiu) ir ${n(CPt.f1_macro)} (kelionės lygiu). Svarbiausia – rekomendacijų patikra su fizikiniu „orakulu“: tiesinis modelis ${pct(QB.nonphysical_share)} atvejų pateikia fiziškai klaidingą įvertį („veiksmas padidins sąnaudas“), modelis be apribojimų – ${pct(QF.nonphysical_share)}, o Proposed modelis – tik ${pct(QP.nonphysical_share)}. Patarimas apskaičiuojamas per ${n(L.median, 1)} ms (p95 – ${n(L.p95, 1)} ms), t. y. gerokai greičiau nei reikalaujamos 200 ms. Demonstracinėje agresyvaus vairuotojo kelionėje, jei būtų laikomasi patarimų, sąnaudos sumažėtų ${n(D.saving_pct, 1)} % (${n(D.litres_actual)} → ${n(D.litres_if_followed)} l).`),
  P("**Svarbi išlyga.** Visi skaičiai gauti su sintetiniais duomenimis ir turi būti patikrinti su realiais OBD-II įrašais (VIII skyrius). Darbe atvirai aprašytos nesėkmės: modelis pervertina pavarų perjungimo naudą ir nepakankamai įvertina tuščiosios eigos poveikį – tai koreliacijos ir priežastingumo painiojimo pasekmė."),
);

// III. PROBLEMOS ANALIZĖ
body.push(H1("III. Problemos analizė"),
  H2("3.1 Kodėl tai aktualu"),
  P("Kelių transportas yra vienas didžiausių CO₂ emisijų šaltinių, o degalai – didelė automobilio išlaikymo ir ypač krovinių vežimo įmonių išlaidų dalis. Eko-vairavimo literatūros apžvalgose [1–3] nurodoma, kad vien vairavimo įpročių keitimas (tolygesnis greitėjimas, ankstesnis pavarų perjungimas, mažesnis greitis magistralėje, trumpesnė tuščioji eiga) leidžia sutaupyti kelis–keliolika procentų degalų be jokių investicijų į automobilį. Problema ta, kad bendri patarimai („važiuok ramiau“) greitai pamirštami, o vairuotojas nežino, kuris veiksmas konkrečioje situacijoje duoda didžiausią naudą: miesto spūstyje svarbiausia gali būti tuščioji eiga ir staigūs greitėjimai, o magistralėje – greitis ir aerodinaminis pasipriešinimas."),
  P("Todėl reikalinga sistema, kuri **realiu laiku** įvertina konkretų kontekstą (maršrutą, eismą, orą) ir vairuotojo elgesį bei pasiūlo **vieną ar du konkrečius, kiekybiškai pagrįstus** veiksmus („sumažinkite greitį iki ~110 km/h – sutaupysite ≈1 l/100 km“)."),
  H2("3.2 Vartotojai ir suinteresuotosios šalys"),
  B("**Privatūs vairuotojai** – nori mažesnių išlaidų degalams, bet nenori būti blaškomi."),
  B("**Krovinių vežimo ir flotės valdymo įmonės** – degalai yra viena didžiausių sąnaudų eilučių; reikia objektyvių vairuotojų įpročių ataskaitų ir mokymų."),
  B("**Navigacijos / telematikos sprendimų kūrėjai** – galėtų integruoti modelį į esamas programėles."),
  B("**Visuomenė ir aplinka** – mažesnės CO₂ ir kitų teršalų emisijos."),
  H2("3.3 Funkciniai ir nefunkciniai reikalavimai"),
  TBL(["Nr.", "Reikalavimas", "Kaip įgyvendinta"], [
    ["F1", "Prognozuoti degalų sąnaudas (l/100 km) pagal maršrutą, eismą, orą ir vairavimo stilių", "HGB regresijos modelis, 18 požymių (VI sk.)"],
    ["F2", "Atpažinti vairavimo stilių (eco / normalus / agresyvus)", "HGB klasifikatorius (VII.2)"],
    ["F3", "Realiu laiku pateikti 1–2 konkrečius patarimus su sutaupymo įverčiu", "What-if rekomendacijų variklis (V.4)"],
    ["F4", "Prieš kelionę palyginti alternatyvius maršrutus pagal degalus ir laiką", "recommend_route() (VII.6)"],
    ["NF1", "Patarimo latencija < 200 ms (metodikos reikalavimas)", `Išmatuota p95 = ${n(L.p95, 1)} ms`],
    ["NF2", "Saugumas: sistema tik pataria, nesiūlo neteisėtų ar pavojingų veiksmų", "Saugos filtras, informacinis režimas"],
    ["NF3", "Atkartojamumas, be uždaro kodo API, veikia be interneto", "seed=42, requirements.txt, Dockerfile"],
    ["NF4", "Paaiškinamumas (XAI)", "Permutacinė svarba, PDP, kiekybinis kiekvieno patarimo įvertis"],
  ], [700, 4426, 3900], { alignLeft: true }),
  H2("3.4 Privalomas pradinis projekto klasifikavimas"),
  TBL(["Kriterijus", "Pasirinktas atsakymas", "Poveikis metodikai"], [
    ["Duomenų pobūdis", "Skaitiniai signalai (telematika: OBD-II/CAN, GPS, orų ir eismo duomenys)", "Lentelių ML bibliotekos: pandas, scikit-learn; NLP/CV įrankiai nereikalingi"],
    ["Duomenų šaltinis", "Sukurti sintetiniai (fizikinis modelis); numatyti atviri rinkiniai realiam taikymui", "Sintetinių duomenų validumas pagrįstas IV.3 skyriuje; asmens duomenų nerenkama"],
    ["Reikalavimas realiam laikui", "TAIP (inference < 200 ms)", "Lengvas medžių ansamblis (HGB), visi kandidatai vertinami vienu predict() kvietimu; jokių sunkių transformerių"],
    ["Aparatūros poreikis", "Realiam diegimui – OBD-II adapteris (Bluetooth ELM327) ir telefonas; prototipas – grynai programinis", "Pridėtas techninės integracijos aprašas (V.5)"],
    ["Sprendimo kritiškumas", "Informacinis (pataria)", "AI niekada nevaldo automobilio (nei greičio, nei pavarų); galutinį sprendimą priima vairuotojas"],
  ], [2100, 3500, 3426], { alignLeft: true }),
  H2("3.5 Etika, sauga ir privatumas"),
  B("**Eismo saugumas pirmiau už taupymą.** Saugos filtras niekada nesiūlo viršyti leistino greičio, nesiūlo mažinti greičio spūstyje (kai srautas sulėtėjęs) ir sniego sąlygomis (greitį tada sprendžia vairuotojas), nesiūlo važiuoti magistralėje lėčiau nei 85 % leistino greičio (per lėtas važiavimas taip pat pavojingas), nesiūlo išjungti kondicionieriaus, kai lauke > 24 °C (vairuotojo budrumas karštyje)."),
  B("**Dėmesio neblaškymas.** Rodoma ne daugiau nei 2 patarimai per kilometrą, patarimai nerodomi, jei sutaupymas < 0,15 l/100 km. Realiame diegime patarimai turėtų būti pateikiami balsu ir tik tiesiose atkarpose, ne manevro metu."),
  B("**Privatumas (BDAR).** GPS trajektorijos ir vairavimo stilius yra asmens duomenys. Prototipe naudojami tik sintetiniai duomenys. Realioje sistemoje: skaičiavimas vietoje telefone (be duomenų siuntimo į serverį), saugomi tik 1 km agreguoti požymiai be tikslių koordinačių, aiškus vairuotojo sutikimas."),
  B("**Piktnaudžiavimo rizika.** Stiliaus klasifikatorius („agresyvus“) gali būti panaudotas prieš vairuotoją (draudimo kainai, darbdavio baudoms). Sistema skirta savišvietai; flotės ataskaitose rekomenduojama rodyti tik suvestinius taupymo rodiklius, o ne „agresyvaus vairuotojo“ etiketę."),
  B("**Nediskriminavimas.** Modelis nenaudoja jokių demografinių požymių (amžiaus, lyties) – tik fizikinius ir elgsenos parametrus."),
  B("**Modelio rizikos atskleidimas.** VII ir VIII skyriuose atvirai parodyta, kur modelio patarimai yra netikslūs (pervertinama pavarų perjungimo nauda), kad vartotojas nepasitikėtų sutaupymo skaičiais aklai."),
  H2("3.6 Vartotojų patirtis (UX)"),
  P("Metodikoje rekomenduojama atlikti 2–3 neformalius pokalbius su potencialiais vartotojais. **Šią dalį reikia užpildyti pačiam po pokalbių** (pvz., su pažįstamais vairuotojais ar krovinių vežėjais), įrašant tikras jų įžvalgas. Siūlomi klausimai: (1) Ar norėtumėte patarimų balsu ar ekrane? (2) Kiek patarimų per kelionę jums nebūtų erzinantis? (3) Ar labiau motyvuotų litrai, eurai ar CO₂? (4) Ar sutiktumėte, kad jūsų vairavimo duomenys būtų matomi darbdaviui?"),
  P("~~[Čia įrašykite 2–3 pokalbių santraukas ir citatas.]~~"),
);

// IV. DUOMENŲ RINKIMAS
body.push(H1("IV. Duomenų rinkimas"),
  H2("4.1 Duomenų šaltiniai"),
  TBL(["Duomenų poreikis", "Rekomenduojamas atviras šaltinis", "Panaudota šiame darbe"], [
    ["Automobilio telematika (greitis, apsukos, degalų srautas)", "VED – Vehicle Energy Dataset [4]; Kaggle OBD-II rinkiniai; savo įrašai per OBD-II adapterį", "Sintetinis atitikmuo (fizikinis modelis, 4.3)"],
    ["Maršrutas, nuolydis, leistinas greitis", "OpenStreetMap, SRTM aukščio modelis", "Sintetinis: kelio tipas, leistinas greitis, nuolydis"],
    ["Eismo spūstys", "TomTom / HERE Traffic API (nemokami planai), istoriniai srautai", "Sintetinis spūsties indeksas 0–1, piko valandos"],
    ["Oras", "OpenWeatherMap API (metodikoje nurodytas šaltinis)", "Sintetinis: temperatūra pagal mėnesį, krituliai, vėjas"],
  ], [2500, 3500, 3026], { alignLeft: true }),
  H2("4.2 Apimtis ir struktūra"),
  P(`Duomenų vienetas – **1 km kelio segmentas** (realiu laiku tai atitinka slenkantį 1 km telematikos langą). Hierarchija: vairuotojas → kelionė → segmentai. Iš viso sugeneruota ${M.data.raw_segments.toLocaleString("lt-LT")} segmentų, ${M.data.trips} kelionės, ${M.data.drivers} vairuotojų (po 20 kiekvieno stiliaus). Kiekvienas vairuotojas turi latentinį agresyvumą a ∈ [0; 1] ir savo automobilio efektyvumo koeficientą (±6 %), kurio modelis nemato – tai imituoja skirtingus automobilius ir yra realistiškas „nepaaiškinamas“ triukšmas.`),
  TBL(["Grupė", "Požymiai (18)", "Ar valdo vairuotojas?"], [
    ["Maršrutas", "kelio tipas (miestas / priemiestis / magistralė, one-hot), leistinas greitis, nuolydis %", "Ne (tik prieš kelionę – maršruto pasirinkimas)"],
    ["Eismas", "spūsties indeksas 0–1", "Ne"],
    ["Oras", "temperatūra °C, priešpriešinis vėjas m/s, krituliai (nėra / lietus / sniegas)", "Ne"],
    ["Krovinys", "papildoma masė, kg", "Iš dalies (prieš kelionę)"],
    ["Vairavimo stilius", "vid. greitis, greitėjimo RMS, staigūs greitėjimai/km, staigūs stabdymai/km, sustojimai/km, tuščiosios eigos dalis, vid. apsukos, kondicionierius", "Taip (išskyrus sustojimus)"],
    ["Tikslas", "degalų sąnaudos, l/100 km (regresija); stiliaus klasė (klasifikacija)", "–"],
  ], [1700, 5226, 2100], { alignLeft: true }),
  H2("4.3 SVARBU – sintetinių duomenų atskleidimas ir pagrindimas"),
  P("Darbo aplinkoje nebuvo prieigos prie interneto, todėl realių rinkinių (pvz., VED) parsisiųsti nepavyko. **Visi kiekybiniai duomenys yra sintetiniai** ir neturi būti interpretuojami kaip realių automobilių matavimai. Metodika reikalauja pagrįsti, kodėl sintetiniai duomenys yra validūs – pagrindimas:"),
  N("**Generuojama iš fizikos dėsnių, ne atsitiktinai.** Sąnaudos skaičiuojamos iš ratų traukos energijos balanso kilometrui: E = E_riedėjimo (m·g·C_rr) + E_aerodinaminė (½·ρ·C_dA·(v+vėjas)²) + E_nuolydžio (m·g·sinα) + E_kinetinė (m·PKE), padalijus iš variklio naudingumo koeficiento (≈30 %, mažėja didėjant apsukoms virš 1800 aps./min ir esant šaltam varikliui) ir benzino energijos (32 MJ/l). Pridedama tuščioji eiga (0,75 l/h) ir kondicionierius (0,35 l/h). Tokia struktūra naudojama ir moksliniuose emisijų modeliuose (pvz., CMEM [5]).", "num"),
  N("**Realistiškos skaitinės ribos.** Gautos medianos: miestas ≈ 8,6 l/100 km, priemiestis ≈ 7,0, magistralė ≈ 7,6; eco vairuotojų mediana ≈ 6,5, agresyvių ≈ 9,7 l/100 km – tai atitinka tipinius C klasės benzininio automobilio rodiklius.", "num"),
  N("**Pridėtas realistiškas „triukšmas“:** 6 % matavimo triukšmas, ±6 % automobilių skirtumai, 0,3 % jutiklių klaidų (outlierių), kurie valomi 4.4 skyriuje.", "num"),
  N("**Pagrindinis privalumas – žinoma „tiesa“.** Kadangi fizikinis modelis žinomas, galima patikrinti, ar AI patarimas tikrai sutaupo degalų (VII.3). Su realiais duomenimis to padaryti neįmanoma be brangių lauko bandymų.", "num"),
  P("Norint pritaikyti realiems duomenims, pakanka pakeisti failą data/segments.csv tais pačiais stulpeliais – modelių ir rekomendacijų kodas nesikeičia (VIII sk.)."),
  H2("4.4 Valymas ir vizualizacija"),
  P(`Trūkstamų reikšmių nėra (generuojant). Outlieriai šalinti IQR taisykle atskirai kiekvienam kelio tipui (sąnaudos > Q3 + 3·IQR arba ≤ 0,2 l/100 km): pašalinti **${M.data.removed_outliers} segmentai** (${n(M.data.removed_outliers / M.data.raw_segments * 100, 2)} %), liko ${M.data.segments.toLocaleString("lt-LT")}. Klasės subalansuotos (eco ${M.data.style_counts.eco}, normalus ${M.data.style_counts.normalus}, agresyvus ${M.data.style_counts.agresyvus} segmentai). 1 pav. parodytas sąnaudų pasiskirstymas ir koreliacijų matrica: stipriausiai su sąnaudomis koreliuoja sustojimai/km, tuščiosios eigos dalis ir greitėjimo intensyvumas; vairavimo stiliaus požymiai tarpusavyje stipriai koreliuoja (pvz., apsukos ir greitėjimas), o tai vėliau sukelia problemų (VIII sk.).`),
  IMG("fig1_duomenys.png", 600),
  CAP("1 pav. Degalų sąnaudų pasiskirstymas pagal kelio tipą ir požymių koreliacijų matrica"),
  H2("4.5 Mokymo ir testavimo aibės"),
  P(`Duomenys padalyti **pagal vairuotojus**, ne atsitiktinai: testavimo aibėje – ${M.split.test_drivers} vairuotojų (po 4 kiekvieno stiliaus), kurių modelis niekada nematė (${M.split.test_segments.toLocaleString("lt-LT")} segmentų); mokymo aibėje – ${M.split.train_drivers} vairuotojai (${M.split.train_segments.toLocaleString("lt-LT")} segmentai). Taip išvengiama duomenų nutekėjimo (to paties vairuotojo segmentai negali būti abiejose aibėse) ir vertinama, kaip sistema veiks naujam vartotojui.`),
);

// V. ARCHITEKTŪRA
body.push(H1("V. Sisteminė architektūra"),
  H2("5.1 Blokinė schema"),
  IMG("fig8_architektura.png", 620),
  CAP("2 pav. Sistemos blokinė schema: duomenų šaltiniai → požymiai → modeliai → rekomendacijos → vairuotojas"),
  H2("5.2 Duomenų srautas"),
  N("**generate_data.py** – sugeneruoja data/segments.csv (sintetiniai segmentai pagal physics.py).", "num2"),
  N("**train_evaluate.py** – išvalo duomenis, padalija pagal vairuotojus, parenka hiperparametrus (GroupKFold), apmoko Baseline ir Proposed modelius, atlieka visus eksperimentus, sukuria grafikus ir outputs/metrics.json.", "num2"),
  N("**recommender.py** – realaus laiko variklis: kandidatiniai veiksmai → saugos filtras → vienas predict() → 1–2 patarimai lietuvių kalba; taip pat maršruto parinkimas.", "num2"),
  N("**demo_realtime.py** (komandinė eilutė) ir **app_streamlit.py** (grafinė sąsaja) – demonstracijai.", "num2"),
  H2("5.3 Rekomendacijų variklio algoritmas"),
  P("Kiekvieną kilometrą sistema gauna požymių vektorių x ir sugeneruoja kandidatinius veiksmus, keičiančius **tik vairuotojo valdomus** požymius:"),
  TBL(["Veiksmas", "Kada svarstomas", "Kaip keičiami požymiai"], [
    ["Greičio mažinimas", "v > leistinas + 2 km/h arba v > 100 km/h", "v → max(min(v−10, leistinas), 0,85·leistinas); apsukos mažėja proporcingai"],
    ["Švelnesnis greitėjimas", "greitėjimo RMS > 0,45 m/s² arba ≥ 1 staigus greitėjimas/km", "RMS × 0,7; staigūs greitėjimai → 0; staigūs stabdymai −1"],
    ["Ankstesnis pavarų perjungimas", "vid. apsukos > 2400", "apsukos −500 (ne mažiau 1800)"],
    ["Tuščiosios eigos mažinimas", "tuščiosios eigos dalis > 8 %", "× 0,4 (variklio išjungimas / Start-Stop ilgiau stovint)"],
    ["Kondicionieriaus išjungimas", "įjungtas ir lauke ≤ 24 °C", "ac_on → 0"],
  ], [2300, 3200, 3526], { alignLeft: true }),
  P("Visi kandidatai (kartu su dabartine būsena) sudedami į vieną lentelę ir įvertinami **vienu** modelio predict() kvietimu; sutaupymas = prognozė(x) − prognozė(x′). Tada taikomas saugos filtras (3.5 sk.), atmetami patarimai su sutaupymu < 0,15 l/100 km ir pateikiami 2 geriausi. Kiekvienas patarimas turi kiekybinį įvertį („≈ −1,2 l/100 km“) – tai kartu ir lokalus sprendimo paaiškinimas (XAI)."),
  H2("5.4 Naudojamos bibliotekos"),
  P("numpy, pandas, scipy, scikit-learn, matplotlib, joblib; grafinei sąsajai – streamlit (requirements.txt, X sk.). Nereikia GPU, interneto ar jokio komercinio API – tai atitinka bendruosius metodikos reikalavimus."),
  H2("5.5 Techninė integracija (realiam diegimui)"),
  P("Pigiausias realus variantas: Bluetooth OBD-II adapteris (ELM327) → išmanusis telefonas (Python/Kotlin programėlė). Iš OBD-II PID gaunami greitis, apsukos, įsiurbiamo oro srautas (MAF – iš jo skaičiuojamas degalų srautas) ir variklio temperatūra; telefono GPS ir akselerometras suteikia trajektoriją, nuolydį ir greitėjimo RMS; orų ir eismo duomenys gaunami per API. Modelis (~1 MB) veikia telefone vietoje. Svarbu: OBD-II skaitymas yra tik skaitymas – sistema į automobilio valdymą nesikiša."),
);

// VI. MODELIO KŪRIMAS
body.push(H1("VI. Modelio kūrimas"),
  H2("6.1 Architektūros pasirinkimo pagrindimas"),
  P(`Pagal metodikos hierarchinę taisyklę: duomenų daug (> 5000 – ${M.split.train_segments.toLocaleString("lt-LT")} mokymo segmentai), bet duomenys **lenteliniai**, o ne vaizdai ar laiko eilutės, todėl CNN/LSTM netinka; reikalingas **realus laikas** – privalomas lengvas modelis; reikalingas **paaiškinamumas**. Todėl pasirinktas gradientinio stiprinimo medžių ansamblis – **HistGradientBoostingRegressor** (scikit-learn implementacija, analogiška LightGBM [6]; pati LightGBM / XGBoost biblioteka aplinkoje nebuvo įdiegta). Baseline – LinearRegression (tiesinė regresija), kaip rekomenduoja metodika.`),
  P("**Monotoniškumo apribojimai** – esminis projekto sprendimas. Fizikiškai žinoma, kad didesnis greitėjimo intensyvumas, daugiau staigių greitėjimų, sustojimų, didesnė tuščiosios eigos dalis, didesnės apsukos, didesnis krovinys, įkalnė, priešpriešinis vėjas ir įjungtas kondicionierius **niekada nesumažina** sąnaudų. Šie 9 požymiai modeliui nurodyti kaip monotoniškai didinantys prognozę. Greitis apribojimo negavo (sąnaudų priklausomybė nuo greičio nėra monotoniška)."),
  H2("6.2 Hiperparametrų parinkimas"),
  P(`Parinkta 4 dalių GroupKFold kryžmine patikra mokymo aibėje (vairuotojai nesikerta tarp dalių), tinklelis: learning_rate ∈ {0,05; 0,1}, max_leaf_nodes ∈ {15; 31; 63}, l2_regularization ∈ {0; 1} (12 derinių, visi rezultatai – outputs/cv_grid.csv).`),
  TBL(["Modelis", "Hiperparametrai"], [
    ["Baseline – LinearRegression", "StandardScaler + mažiausių kvadratų metodas, be reguliarizacijos"],
    ["Proposed – HGB + monotoniškumas", `learning_rate=${n(M.best_hparams.learning_rate)}, max_leaf_nodes=${M.best_hparams.max_leaf_nodes}, l2_regularization=${n(M.best_hparams.l2_regularization, 1)}, max_iter=400 su early_stopping, monotonic_cst (9 požymiai = +1), random_state=42`],
    ["Stiliaus klasifikatorius – HGB", "learning_rate=0,08, max_iter=300, early_stopping; požymiai: greičio ir leistino greičio santykis, greitėjimo RMS, staigūs greitėjimai/stabdymai, tuščioji eiga, apsukos, greitis, spūstis, kelio tipas"],
    ["Stiliaus Baseline – taisyklė", `accel_rms < ${n(C.thresholds_accel_rms[0], 3)} → eco; < ${n(C.thresholds_accel_rms[1], 3)} → normalus; kitaip agresyvus (mokymo aibės terciliai)`],
  ], [3000, 6026], { alignLeft: true }),
  H2("6.3 Nesėkmės ir projektiniai pakeitimai"),
  P("Metodikoje skatinama dokumentuoti nesėkmes. Kūrimo metu buvo trys svarbūs pakeitimai:"),
  N(`**Bandymas 1 – atsitiktinis vairuotojų padalijimas.** Pirmą kartą testavimo vairuotojai buvo parinkti atsitiktinai, ir į testavimo aibę pateko daugiausia eco vairuotojai (≈2600 iš ≈3900 segmentų). Klasifikatoriaus rezultatai atrodė geresni nei iš tikrųjų, nes eco klasę atpažinti lengviausia. **Sprendimas:** stratifikuotas padalijimas – po 4 kiekvieno stiliaus vairuotojus.`, "num3"),
  N(`**Bandymas 2 – HGB be monotoniškumo apribojimų.** Šis modelis turi geriausią prognozės tikslumą (RMSE ${n(RF.RMSE)} vs ${n(RP.RMSE)} l/100 km), bet tikrinant rekomendacijas paaiškėjo, kad ${pct(QF.nonphysical_share)} atvejų jis teigia, jog fiziškai visada naudingas veiksmas **padidins** sąnaudas (pvz., tuščiosios eigos mažinimui – net ${pct(QF.neg_by_action.tuscioji_eiga)} atvejų). Toks patarėjas klaidintų vairuotoją. **Sprendimas:** monotoniškumo apribojimai – nedidelė tikslumo kaina už fizikiškai nuoseklius patarimus.`, "num3"),
  N(`**Bandymas 3 – rekomendacijos iš tiesinio modelio.** LinearRegression tuščiosios eigos koeficientas gavosi **neigiamas** (dėl koreliacijos su lėtu važiavimu spūstyje), todėl ${pct(QB.neg_by_action.tuscioji_eiga, 0)} tuščiosios eigos patarimų jis įvertina kaip žalingus. Be to, tiesinis modelis nesugeba atvaizduoti kvadratinės aerodinaminio pasipriešinimo priklausomybės nuo greičio, todėl greičio mažinimo naudą nuvertina beveik perpus (VII.3).`, "num3"),
  H2("6.4 Paaiškinamumas (XAI)"),
  P(`Kadangi SHAP biblioteka darbo aplinkoje nebuvo įdiegta, naudoti scikit-learn įrankiai: **permutacinė požymių svarba** [7] (3 pav.) ir **dalinės priklausomybės grafikai** (PDP, 4 pav.). Svarbiausi požymiai: sustojimai/km (RMSE padidėja ${n(PI.stops_per_km)} l/100 km juos sumaišius), vidutinis greitis (${n(PI.avg_speed_kmh)}), nuolydis (${n(PI.grade_pct)}), staigūs greitėjimai (${n(PI.harsh_acc_per_km)}) ir apsukos (${n(PI.avg_rpm)}). Tai atitinka fizikinį modelį – sustojimai ir greitis lemia kinetinės energijos nuostolius. Staigūs stabdymai turi ≈0 svarbos – tai teisinga, nes fizikiniame modelyje stabdymas pats degalų nenaudoja (degalai išleidžiami ankstesniam greitėjimui).`),
  IMG("fig3_pozymiu_svarba.png", 470),
  CAP("3 pav. Permutacinė požymių svarba (Proposed modelis). Mėlyna – vairuotojo valdomi požymiai"),
  P("PDP grafikai rodo fizikiškai prasmingas priklausomybes: sąnaudos (per km) auga didėjant greičiui, apsukoms; šaltis (< 10–15 °C) didina sąnaudas. Įdomu, kad greitėjimo RMS kreivė beveik plokščia – jo informaciją modelis „pasiima“ iš koreliuojančių požymių (staigūs greitėjimai, apsukos). Tai tiesiogiai susiję su VIII skyriuje aprašytu ribotumu."),
  IMG("fig4_pdp.png", 620),
  CAP("4 pav. Dalinės priklausomybės grafikai: kaip keičiasi prognozė keičiant vieną požymį"),
);

// VII. EKSPERIMENTAI
body.push(H1("VII. Eksperimentai"),
  H2("7.1 A užduotis – degalų sąnaudų prognozė (Baseline vs Proposed)"),
  P(`Testavimo aibė: ${M.split.test_drivers} nematytų vairuotojų, ${M.split.test_segments.toLocaleString("lt-LT")} segmentų.`),
  TBL(["Modelis", "RMSE, l/100 km", "MAE, l/100 km", "R²", "MAPE"], [
    ["Vidurkio prognozė (trivialus)", n(RM.RMSE), n(RM.MAE), n(RM.R2, 3), n(RM.MAPE, 1) + " %"],
    ["Baseline – LinearRegression", n(RB.RMSE), n(RB.MAE), n(RB.R2, 3), n(RB.MAPE, 1) + " %"],
    ["HGB be monotoniškumo (ablacija)", n(RF.RMSE), n(RF.MAE), n(RF.R2, 3), n(RF.MAPE, 1) + " %"],
    ["**Proposed – HGB + monotoniškumas**", n(RP.RMSE), n(RP.MAE), n(RP.R2, 3), n(RP.MAPE, 1) + " %"],
  ], [3426, 1400, 1400, 1300, 1500], { highlight: [3] }),
  SP(),
  P(`Proposed modelis RMSE sumažina **${n((1 - RP.RMSE / RB.RMSE) * 100, 0)} %** lyginant su Baseline. Kelionės lygiu (vidutinės kelionės sąnaudos) Proposed R² = ${n(TL["Proposed – HGB + monotoniškumas"].R2, 3)}, MAPE = ${n(TL["Proposed – HGB + monotoniškumas"].MAPE, 1)} %, Baseline – R² = ${n(TL["Baseline – LinearRegression"].R2, 3)}, MAPE = ${n(TL["Baseline – LinearRegression"].MAPE, 1)} %. Kodėl Proposed geresnis: tiesinis modelis negali atvaizduoti netiesinių sąveikų (greičio kvadratas aerodinamikoje, sustojimų ir greičio sandauga kinetinėje energijoje, šalto variklio efektas), todėl sistemingai nuvertina dideles sąnaudas (5 pav. kairėje – „suplota“ taškų debesis).`),
  IMG("fig2_prognoze_vs_tikrove.png", 560),
  CAP("5 pav. Prognozuotos vs tikros sąnaudos testavimo aibėje"),
  H2("7.2 B užduotis – vairavimo stiliaus klasifikacija"),
  TBL(["Modelis", "Accuracy (segm.)", "F1 makro (segm.)", "Accuracy (kelionė)", "F1 makro (kelionė)"], [
    ["Baseline – accel_rms taisyklė", pct(CB.acc), n(CB.f1_macro), pct(CBt.acc), n(CBt.f1_macro)],
    ["**Proposed – HGB klasifikatorius**", pct(CP.acc), n(CP.f1_macro), pct(CPt.acc), n(CPt.f1_macro)],
  ], [3026, 1500, 1500, 1500, 1500], { highlight: [1] }),
  SP(),
  P(`Proposed modelis pasiekia metodikoje reikalaujamą **F1 ≥ 0,70** ir **Accuracy ≥ 80 %** (segmento lygiu F1 = ${n(CP.f1_macro)}, Accuracy = ${pct(CP.acc)}). Kelionės lygiu (vidutinės segmentų tikimybės) rezultatas geresnis – F1 = ${n(CPt.f1_macro)}, nes vieno kilometro triukšmas išsilygina. Pagal klases (segmentas): eco F1 = ${n(rep.eco["f1-score"])}, normalus F1 = ${n(rep.normalus["f1-score"])}, agresyvus F1 = ${n(rep.agresyvus["f1-score"])}. Sunkiausia atpažinti „normalų“ stilių – jis persidengia su abiem kraštutinumais (6 pav.), o tai natūralu, nes stilius yra tolydus, o ne trys griežtos klasės.`),
  IMG("fig5_konfuzijos_matrica.png", 520),
  CAP("6 pav. Normalizuotos konfuzijos matricos (segmento lygiu)"),
  H2("7.3 C – ar rekomendacijos tikrai sutaupo? (patikra su fizikiniu orakulu)"),
  P(`Tai svarbiausias eksperimentas: gera sąnaudų prognozė dar negarantuoja gerų patarimų. Iš testavimo aibės paimta ${REC.n_segments.toLocaleString("lt-LT")} segmentų su bent vienu leistinu veiksmu (iš viso ${REC.n_action_evals.toLocaleString("lt-LT")} veiksmo įvertinimų). Kiekvienam veiksmui palygintas modelio prognozuotas sutaupymas su **tikru** sutaupymu, apskaičiuotu fizikiniu modeliu. Visi penki veiksmai fiziškai visada sutaupo (tikras sutaupymas > 0 visais atvejais).`),
  TBL(["Rodiklis", "Baseline (LinReg)", "HGB be monot.", "Proposed (HGB + monot.)"], [
    ["Fiziškai klaidingi įverčiai („veiksmas padidins sąnaudas“) ↓", pct(QB.nonphysical_share), pct(QF.nonphysical_share), "**" + pct(QP.nonphysical_share) + "**"],
    ["Teisinga kryptis (modelis mato sutaupymą) ↑", pct(QB.sign_accuracy), pct(QF.sign_accuracy), "**" + pct(QP.sign_accuracy) + "**"],
    ["Modelis „nemato“ efekto (|Δ| ≤ 0,01) ↓", pct(QB.zero_effect_share), pct(QF.zero_effect_share), pct(QP.zero_effect_share)],
    ["Sutaupymo įverčio MAE, l/100 km ↓", n(QB.saving_MAE, 3), n(QF.saving_MAE, 3), n(QP.saving_MAE, 3)],
    ["Spearman ρ (rangų koreliacija) ↑", n(QB.spearman), n(QF.spearman), n(QP.spearman)],
    ["Pirmas patarimas – tikrai geriausias ↑", pct(QB.top1_hit), pct(QF.top1_hit), pct(QP.top1_hit)],
    ["Realus sutaupymas laikantis 1-o patarimo", `${n(QB.realized_saving_l100)} l/100 km (${n(QB.realized_saving_pct, 1)} %)`, `${n(QF.realized_saving_l100)} l/100 km (${n(QF.realized_saving_pct, 1)} %)`, `${n(QP.realized_saving_l100)} l/100 km (${n(QP.realized_saving_pct, 1)} %)`],
  ], [3626, 1800, 1800, 1800]),
  SP(),
  P(`Optimalus (orakulo) sutaupymas, jei visada būtų pasirinktas tikrai geriausias veiksmas, yra ${n(REC.oracle_best_possible_l100)} l/100 km; Proposed pasiekia ${n(QP.realized_saving_l100 / REC.oracle_best_possible_l100 * 100, 0)} % šio maksimumo.`),
  P(`**Interpretacija – sąžiningas palyginimas.** Proposed modelis yra **saugiausias**: fiziškai klaidingų patarimų beveik nėra (${pct(QP.nonphysical_share)} vs ${pct(QB.nonphysical_share)} Baseline). Tačiau modelis be apribojimų turi geresnį rangavimą (Spearman ${n(QF.spearman)} vs ${n(QP.spearman)}) ir dažniau pirmu pasiūlo tikrai geriausią veiksmą (${pct(QF.top1_hit)} vs ${pct(QP.top1_hit)}). Kitaip tariant, monotoniškumas „ištrina“ kai kuriuos klaidingus įverčius paversdamas juos nuliu (${pct(QP.zero_effect_share)} atvejų efektas nematomas – daugiausia tuščiosios eigos, ${pct(QP.zero_by_action.tuscioji_eiga, 0)}). Tokie patarimai tiesiog nerodomi (žemiau 0,15 l/100 km slenksčio) – tai **saugi** klaida (praleistas patarimas), o ne pavojinga (klaidinantis patarimas). Patarėjo sistemai, kuria vairuotojas turi pasitikėti, pasirinkome saugumą.`),
  IMG("fig6_rekomendaciju_patikra.png", 560),
  CAP("7 pav. Modelio prognozuotas vs tikras (fizikinis) sutaupymas. Raudona linija – nulis: taškai po ja yra fiziškai klaidingi įverčiai"),
  P("Vidutinis sutaupymas pagal veiksmų tipus (l/100 km) atskleidžia, kur modeliai klysta:"),
  TBL(["Veiksmas", "Kartų", "Tikras (fizika)", "Proposed prognozė", "Baseline prognozė"],
    acts.map(a => [actLt[a], REC.action_counts[a], n(REC.by_action_true_saving_mean[a]), n(REC.by_action_hgb_saving_mean[a]), n(REC.by_action_lr_saving_mean[a])]),
    [3226, 1000, 1600, 1600, 1600]),
  SP(),
  P(`Baseline greičio mažinimo naudą nuvertina beveik perpus (${n(REC.by_action_lr_saving_mean.greitis)} vs tikras ${n(REC.by_action_true_saving_mean.greitis)}), nes negali atvaizduoti v² priklausomybės, o tuščiosios eigos mažinimą laiko žalingu. Proposed greičio ir kondicionieriaus naudą įvertina tiksliai, tačiau **pervertina pavarų perjungimo naudą** (${n(REC.by_action_hgb_saving_mean.apsukos)} vs ${n(REC.by_action_true_saving_mean.apsukos)}) ir **nuvertina tuščiosios eigos** (${n(REC.by_action_hgb_saving_mean.tuscioji_eiga)} vs ${n(REC.by_action_true_saving_mean.tuscioji_eiga)}).`),
  P(`**Nesėkmės pavyzdys.** Iš ${F.n.toLocaleString("lt-LT")} segmentų su ≥ 2 veiksmais Proposed pirmu patarimu ${rpmPick} kartų pasirinko „perjunkite pavarą anksčiau“, bet tik ${rpmCorrect} kartų (${n(rpmCorrect / rpmPick * 100, 0)} %) tai buvo tikrai geriausias veiksmas – dažniausiai geriau būtų buvę patarti švelniau greitėti. Blogiausiu atveju modelis pavarų perjungimui prognozavo ${n(F.worst[2].pick_pred)} l/100 km sutaupymą, tikrasis buvo ${n(F.worst[2].pick_true)}, o geriausias veiksmas (švelnesnis greitėjimas) būtų sutaupęs ${n(F.worst[2].best_true)} l/100 km. Priežastis – apsukos ir agresyvus greitėjimas duomenyse stipriai koreliuoja (abu priklauso nuo vairuotojo agresyvumo), todėl modelis dalį greitėjimo „kaltės“ priskiria apsukoms (žr. VIII sk.). Vis dėlto vidutiniškai prarandama tik ${n(F.mean_regret, 2)} l/100 km, palyginti su optimaliu patarimu.`),
  H2("7.4 D – realaus laiko reikalavimas"),
  P(`Išmatuota 1000 rekomendacijų užklausų (visas ciklas: kandidatų generavimas, saugos filtras, predict(), rūšiavimas) įprastu CPU: **mediana ${n(L.median, 1)} ms, p95 ${n(L.p95, 1)} ms, maksimumas ${n(L.max, 1)} ms**. Reikalavimas < 200 ms įvykdytas su didele atsarga. Kadangi patarimas atnaujinamas kas 1 km (prie 50 km/h – kas ~72 s), tikrasis „vėlavimas“ priklauso nuo lango ilgio, o ne nuo modelio (VIII sk.).`),
  H2("7.5 E – realaus laiko demonstracija"),
  P(`8 pav. parodyta agresyvaus testavimo vairuotojo kelionė (#${D.trip_id}, ${D.km} km). Kiekvieną kilometrą sistema pateikė patarimus. Jei vairuotojas būtų jų laikęsis, fizikiniu modeliu apskaičiuotos sąnaudos sumažėtų nuo **${n(D.litres_actual)} l iki ${n(D.litres_if_followed)} l (−${n(D.saving_pct, 1)} %)**. Tai viršutinė riba – daroma prielaida, kad vairuotojas laikosi visų patarimų.`),
  IMG("fig7_realaus_laiko_demo.png", 620),
  CAP("8 pav. Realaus laiko demonstracija: faktinės sąnaudos ir sąnaudos laikantis AI patarimų"),
  P("Sistemos išvesties pavyzdys (demo_realtime.py):"),
  ...D.log_first.slice(0, 3).map(l => CODE(`[${l.km} km | ${l.kelias} | ${l.greitis} km/h | ${n(l.sanaudos, 1)} l/100 km | ${n(l.latency_ms, 1)} ms]  ${l.patarimai.replace(/\.(\d)/g, ",$1")}`)),
  SP(),
  H2("7.6 F – maršruto pasirinkimas prieš kelionę"),
  P("Normalaus stiliaus vairuotojui palyginti trys alternatyvūs maršrutai (žiemos diena, +5 °C). Modelio prognozė patikrinta fizikiniu modeliu:"),
  TBL(["Maršrutas", "Prognozė, l", "Fizika, l", "Laikas, min"],
    routeNames.map(k => [k + (k === RC.best ? " ✔" : ""), n(RC.routes[k].litres), n(RC.routes[k].litres_physics), n(RC.routes[k].minutes, 0)]),
    [4226, 1600, 1600, 1600], { highlight: [routeNames.indexOf(RC.best)] }),
  SP(),
  P(`Modelis teisingai surikiavo maršrutus (tokia pati tvarka kaip fizikinio modelio). Taupiausias – trumpas maršrutas per miestą, nors jame daugiau spūsčių; tačiau jis užtrunka ${n(RC.routes[routeNames[0]].minutes, 0)} min vs ${n(RC.routes[routeNames[1]].minutes, 0)} min aplinkkeliu. Todėl sistema turi rodyti **abu rodiklius** ir leisti vairuotojui pasirinkti – „taupiausias“ ne visada reiškia „geriausias“.`),
);

// VIII. RIBOTUMAI
body.push(H1("VIII. Ribotumai"),
  N("**Sintetiniai duomenys (svarbiausias ribotumas).** Modelis išmoko fizikinį modelį, kurį pats sukūriau, todėl aukšti rodikliai (R² ≈ 0,9) realiuose duomenyse greičiausiai bus žemesni: realūs jutikliai triukšmingesni, yra daug nematomų veiksnių (padangų slėgis, variklio būklė, kuro kokybė, pavarų dėžės tipas). Kitas žingsnis – surinkti savo OBD-II duomenis arba naudoti VED [4] ir pakartoti tą patį vertinimą.", "num4"),
  N("**Koreliacija ≠ priežastingumas.** Modelis išmoksta koreliacijas, o rekomendacijoms reikia priežastinių efektų. Kadangi apsukos ir agresyvus greitėjimas kinta kartu, modelis pervertina pavarų perjungimo naudą (VII.3). Su realiais duomenimis tai bus dar ryškiau. Galimi sprendimai: priežastiniai modeliai (pvz., double ML), fizikiniai apribojimai (hibridinis modelis „fizika + ML likutis“) arba kontroliuoti eksperimentai su vairuotojais.", "num4"),
  N(`**Tuščiosios eigos efektas nuvertinamas.** Proposed modelis ${pct(QP.zero_by_action.tuscioji_eiga, 0)} tuščiosios eigos atvejų sutaupymo „nemato“, nes šis požymis turi mažą bendrą svarbą. Sistema praleis dalį naudingų patarimų miesto spūstyse.`, "num4"),
  N("**1 km langas.** Patarimas atnaujinamas kas kilometrą, todėl staigiems įvykiams (pvz., vienam staigiam greitėjimui nuo šviesoforo) reakcija vėluoja. Realiame diegime reikėtų trumpesnio slenkančio lango (pvz., 200 m) ir įvykių detektoriaus.", "num4"),
  N("**Vienas automobilio tipas.** Fizikinis modelis aprašo benzininį C klasės automobilį. Dyzeliniams, hibridiniams ir elektriniams automobiliams (rekuperacija!) reikia kitų parametrų; sunkvežimiams – kitos masės ir aerodinamikos.", "num4"),
  N("**Vairuotojo paklusnumo prielaida.** Demonstracijos sutaupymas (8 pav.) skaičiuotas darant prielaidą, kad visų patarimų laikomasi. Realiai laikomasi tik dalies, o efektas laikui bėgant mažėja; tai patikrinti galima tik bandymais su žmonėmis.", "num4"),
  N("**Supaprastintas eismas ir oras.** Spūstis aprašyta vienu skaičiumi; nėra šviesoforų fazių, sankryžų geometrijos, realių orų API duomenų.", "num4"),
);

// IX. LITERATŪRA
body.push(H1("IX. Literatūra"),
  ...[
    "Barkenbus, J. N. (2010). Eco-driving: An overlooked climate change initiative. Energy Policy, 38(2), 762–769.",
    "Alam, M. S., & McNabola, A. (2014). A critical review and assessment of Eco-Driving policy & technology: Benefits & limitations. Transport Policy, 35, 42–49.",
    "Huang, Y., Ng, E. C. Y., Zhou, J. L., Surawski, N. C., Chan, E. F. C., & Hong, G. (2018). Eco-driving technology for sustainable road transport: A review. Renewable and Sustainable Energy Reviews, 93, 596–609.",
    "Oh, G., Leblanc, D. J., & Peng, H. (2022). Vehicle Energy Dataset (VED), a large-scale dataset for vehicle energy consumption research. IEEE Transactions on Intelligent Transportation Systems, 23(4), 3302–3312.",
    "Barth, M., An, F., Younglove, T., Scora, G., Levine, C., Ross, M., & Wenzel, T. (2000). Comprehensive Modal Emissions Model (CMEM), version 2.0 user's guide. University of California, Riverside.",
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30.",
    "Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. Journal of Machine Learning Research, 20(177), 1–81.",
    "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30.",
    "scikit-learn dokumentacija: HistGradientBoostingRegressor, monotonic constraints, permutation importance, partial dependence. https://scikit-learn.org/stable/",
  ].map(t => N(t, "lit")),
);

// X. PRIEDAI
body.push(H1("X. Priedai"),
  H2("10.1 Kodo saugykla"),
  P("GitHub nuoroda: **[įkelkite projekto aplanką į GitHub ir įrašykite nuorodą čia]**. Saugykloje yra README.md, requirements.txt ir Dockerfile (be jų metodikoje numatyta −5 balų bauda)."),
  H2("10.2 Projekto struktūra"),
  ...[
    "src/physics.py          fizikinis sąnaudų modelis",
    "src/generate_data.py    sintetinių duomenų generatorius",
    "src/recommender.py      rekomendacijų variklis + maršruto parinkimas",
    "src/train_evaluate.py   visi eksperimentai, grafikai, metrics.json",
    "src/analyze_failures.py rekomendacijų nesėkmių analizė",
    "src/demo_realtime.py    realaus laiko demo (komandinė eilutė)",
    "src/app_streamlit.py    grafinė sąsaja",
    "requirements.txt, Dockerfile, README.md",
  ].map(CODE),
  SP(),
  H2("10.3 Paleidimo instrukcija"),
  ...[
    "pip install -r requirements.txt",
    "python src/generate_data.py --out data/segments.csv",
    "python src/train_evaluate.py          # ~1,5 min, sukuria figures/ ir outputs/",
    "python src/analyze_failures.py",
    "python src/demo_realtime.py --trip 97 --delay 0.3",
    "streamlit run src/app_streamlit.py    # grafinė sąsaja naršyklėje",
  ].map(CODE),
  SP(),
  P("Arba per Docker: docker build -t fuel-ai . ir docker run -p 8501:8501 fuel-ai, tada naršyklėje atidaryti http://localhost:8501."),
  H2("10.4 requirements.txt"),
  ...fs.readFileSync(path.join(ROOT, "requirements.txt"), "utf8").trim().split("\n").map(CODE),
  SP(),
  H2("10.5 Pastaba apie gyvą demonstraciją"),
  P("Pristatymui rekomenduojama: (1) paleisti demo_realtime.py su --delay 0.3 – matosi, kaip kas kilometrą atsiranda patarimai; (2) Streamlit sąsajoje „gyvai“ pakeisti greitį iš 130 į 110 km/h arba įjungti sniegą ir parodyti, kaip keičiasi patarimai (sniege greičio patarimas dingsta dėl saugos filtro); (3) būti pasiruošus atsakyti, kodėl modelis pervertina pavarų perjungimo naudą (VII.3, VIII.2)."),
);

const doc = new Document({
  creator: "Mantas Šarakojis",
  title: "Degalų sąnaudų optimizavimo rekomendacijų sistema",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: "1F4E79" },
        paragraph: { spacing: { before: 240, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "2E75B6" },
        paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      ...["num", "num2", "num3", "num4", "lit"].map(r => ({ reference: r, levels: [{ level: 0,
        format: LevelFormat.DECIMAL, text: r === "lit" ? "[%1]" : "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 400 } } } }] })),
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(ROOT, "report", "Degalu_sanaudu_rekomendaciju_sistema_ataskaita.docx");
  fs.writeFileSync(out, buf);
  console.log("OK", out);
});
