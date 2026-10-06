import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED, N_IND = 42, 60
MONTHS = pd.period_range("2025-01", periods=12, freq="M").astype(str).tolist()

# name, category, hs, hazard, moisture%, calorific MJ/kg, pH, heavy_metals ppm, state, base INR/t, virgin kgCO2e/t, proc kgCO2e/t
MAT = [("fly ash", "mineral", "2621", "non-hazardous", 1, 1, 9, 40, "powder", 800, 900, 40),
       ("blast furnace slag", "mineral", "2618", "non-hazardous", 5, 0.5, 11, 60, "solid", 1200, 850, 60),
       ("steel slag", "mineral", "2619", "irritant", 4, 0.5, 12, 120, "solid", 900, 800, 50),
       ("FGD gypsum", "mineral", "2520", "non-hazardous", 10, 0, 7, 20, "powder", 700, 400, 30),
       ("rice husk", "biomass", "1213", "non-hazardous", 12, 14, 6.5, 5, "solid", 1500, 300, 20),
       ("bagasse", "biomass", "2303", "non-hazardous", 45, 9, 6, 3, "solid", 1100, 250, 25),
       ("waste paper", "fibre", "4707", "non-hazardous", 8, 13, 7, 10, "solid", 9000, 1100, 150),
       ("spent acid", "chemical", "2807", "toxic", 70, 0, 1.5, 300, "liquid", 2500, 1500, 200),
       ("used oil", "chemical", "2710", "toxic", 3, 38, 7, 250, "liquid", 18000, 2500, 400),
       ("textile sludge", "sludge", "3825", "irritant", 75, 4, 8, 180, "sludge", 300, 200, 80),
       ("plastic scrap", "polymer", "3915", "non-hazardous", 2, 35, 7, 30, "solid", 14000, 2300, 350),
       ("mill scale", "metal", "2619", "non-hazardous", 3, 0, 7, 90, "solid", 6000, 1900, 120)]

# sector -> {material: tonnes waste generated per tonne of main throughput}
PRODUCE = {"steel": {"blast furnace slag": .30, "steel slag": .15, "mill scale": .03},
           "thermal_power": {"fly ash": .30, "FGD gypsum": .04},
           "chemicals": {"spent acid": .08},
           "textile": {"textile sludge": .05},
           "sugar": {"bagasse": .28},
           "rice_mill": {"rice husk": .20},
           "auto_service": {"used oil": .50},
           "packaging": {"plastic scrap": .06, "waste paper": .10}}
# sector -> {material: (tonnes consumed per tonne throughput, application, is_fuel)}
CONSUME = {"steel": {"mill scale": (.02, "sinter plant feed", False)},
           "thermal_power": {"rice husk": (.03, "biomass co-firing", True), "bagasse": (.02, "biomass co-firing", True)},
           "cement": {"fly ash": (.28, "PPC blending", False), "blast furnace slag": (.15, "PSC blending", False),
                      "FGD gypsum": (.04, "set retarder", False), "plastic scrap": (.02, "kiln alt. fuel", True),
                      "used oil": (.005, "kiln alt. fuel", True), "textile sludge": (.01, "kiln co-processing", True),
                      "rice husk": (.02, "kiln alt. fuel", True)},
           "construction_materials": {"fly ash": (.35, "fly-ash bricks/concrete", False),
                                      "steel slag": (.25, "road aggregate", False),
                                      "blast furnace slag": (.15, "slag concrete", False),
                                      "FGD gypsum": (.06, "gypsum plaster/board", False)},
           "fertilizer": {"spent acid": (.03, "acid substitute in SSP", False)},
           "chemicals": {"mill scale": (.02, "iron oxide/ferrite feed", False)},
           "textile": {"rice husk": (.12, "boiler fuel", True), "bagasse": (.10, "boiler fuel", True)},
           "paper": {"waste paper": (1.1, "recycled pulp", False), "bagasse": (.30, "agro pulp", False),
                     "rice husk": (.10, "boiler fuel", True)},
           "oil_recycling": {"used oil": (.80, "re-refining", False)},
           "plastic_recycling": {"plastic scrap": (.90, "mechanical recycling", False)}}
SECTOR_COUNT = {"steel": 6, "thermal_power": 7, "cement": 6, "fertilizer": 3, "chemicals": 5, "textile": 5, "sugar": 5,
                "paper": 4, "rice_mill": 5, "construction_materials": 4, "auto_service": 3, "oil_recycling": 2,
                "packaging": 3, "plastic_recycling": 2}  # sums to 60
MEDIAN_CAP = {"steel": 1.5e6, "thermal_power": 4e6, "cement": 2.5e6, "fertilizer": 6e5, "chemicals": 2e5, "textile": 3e4,
              "sugar": 5e5, "paper": 1.5e5, "rice_mill": 1e5, "construction_materials": 2e5, "auto_service": 5e3,
              "oil_recycling": 5e4, "packaging": 3e4, "plastic_recycling": 2e4}
_S = {"sugar": [1.6, 1.6, 1.4, .8, .05, .05, .05, .05, .05, .1, 1.4, 1.6],  # crushing season Nov-Apr
      "rice_mill": [1.3, 1.1, 1, 1.1, 1, .8, .7, .7, .8, 1, 1.3, 1.4],
      "thermal_power": [.95, .95, 1, 1.05, 1.1, 1.05, .9, .9, .95, 1, 1, 1]}
SEASON = {k: np.array(v) / np.mean(v) for k, v in _S.items()}
# real industrial clusters: name, state, region, lat, lon, sectors hosted
CL = [("Jamshedpur", "Jharkhand", "East", 22.80, 86.20, {"steel", "chemicals", "auto_service", "construction_materials"}),
      ("Bokaro", "Jharkhand", "East", 23.67, 86.15, {"steel", "thermal_power"}),
      ("Rourkela-Jharsuguda", "Odisha", "East", 22.00, 84.50, {"steel", "thermal_power", "cement"}),
      ("Angul-Talcher", "Odisha", "East", 20.95, 85.10, {"steel", "thermal_power", "chemicals"}),
      ("Chennai-Ennore", "Tamil Nadu", "South", 13.20, 80.30, {"thermal_power", "chemicals", "fertilizer", "auto_service", "packaging", "oil_recycling", "plastic_recycling", "construction_materials"}),
      ("Tiruppur-Erode", "Tamil Nadu", "South", 11.20, 77.40, {"textile", "paper", "rice_mill"}),
      ("Salem-Ariyalur", "Tamil Nadu", "South", 11.50, 78.50, {"steel", "cement"}),
      ("Thoothukudi", "Tamil Nadu", "South", 8.80, 78.15, {"thermal_power", "chemicals"}),
      ("Thanjavur-Delta", "Tamil Nadu", "South", 10.80, 79.15, {"rice_mill", "sugar"}),
      ("Pune-Chakan", "Maharashtra", "West", 18.70, 73.80, {"auto_service", "packaging", "plastic_recycling", "construction_materials"}),
      ("Nagpur-Chandrapur", "Maharashtra", "West", 20.50, 79.20, {"thermal_power", "cement", "steel", "construction_materials"}),
      ("Kolhapur-Sangli", "Maharashtra", "West", 16.80, 74.40, {"sugar", "textile", "paper"}),
      ("Mumbai-Thane", "Maharashtra", "West", 19.10, 73.00, {"chemicals", "packaging", "plastic_recycling", "oil_recycling"}),
      ("Surat-Vapi", "Gujarat", "West", 20.80, 72.90, {"textile", "chemicals", "packaging", "fertilizer"}),
      ("Mundra-Kutch", "Gujarat", "West", 22.80, 69.70, {"thermal_power", "cement"}),
      ("Ballari-Hospet", "Karnataka", "South", 15.20, 76.40, {"steel", "thermal_power", "cement"}),
      ("Mandya-Mysuru", "Karnataka", "South", 12.50, 76.90, {"sugar", "rice_mill", "textile"}),
      ("Visakhapatnam", "Andhra Pradesh", "South", 17.70, 83.20, {"steel", "chemicals", "fertilizer", "thermal_power"}),
      ("Krishna-Godavari", "Andhra Pradesh", "South", 16.80, 81.00, {"rice_mill", "paper", "thermal_power", "sugar"})]
RULES = {"max_heavy_metals_ppm": ("heavy metal content", "ppm"), "max_moisture_pct": ("moisture content", "%"),
         "max_storage_days": ("storage duration", "days")}
DOC = {"toxic": "Hazardous Waste Rules 2016", "irritant": "Hazardous Waste Rules 2016", "non-hazardous": "Solid Waste Rules 2016"}
GRADE = {"A": (0.9, 1.0), "B": (1.0, 1.0), "C": (1.15, 1.3)}  # (moisture, heavy-metal) multiplier


def hav(a, b):
    p = np.pi / 180
    h = np.sin((b[0] - a[0]) * p / 2) ** 2 + np.cos(a[0] * p) * np.cos(b[0] * p) * np.sin((b[1] - a[1]) * p / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def build(seed=SEED, n_ind=N_IND):
    rng = np.random.default_rng(seed)
    mats = pd.DataFrame([(f"M{i+1:02d}",) + m[:9] for i, m in enumerate(MAT)],
                        columns=["material_id", "name", "category", "hs_code", "hazard_class", "moisture_pct",
                                 "calorific_value", "ph", "heavy_metals_ppm", "physical_state"])
    mi, mid = mats.set_index("material_id"), dict(zip(mats.name, mats.material_id))
    base = {f"M{i+1:02d}": m[9] for i, m in enumerate(MAT)}
    ef = pd.DataFrame([(f"M{i+1:02d}", m[10], m[11]) for i, m in enumerate(MAT)],
                      columns=["material_id", "virgin_kgco2e_per_tonne", "processing_kgco2e_per_tonne"])
    # industries: sector -> real cluster -> jittered location; role derived from what the sector actually does
    rows, n = [], 0
    for sec, c in SECTOR_COUNT.items():
        for _ in range(max(1, round(c * n_ind / 60))):
            n += 1
            cl = [x for x in CL if sec in x[5]]; cl = cl[rng.integers(len(cl))]
            p, q = bool(PRODUCE.get(sec)), bool(CONSUME.get(sec))
            rows.append((f"I{n:03d}", f"{sec}_{cl[0].split('-')[0]}_{n}", sec, round(cl[3] + rng.normal(0, .12), 4),
                         round(cl[4] + rng.normal(0, .12), 4), cl[1], int(rng.lognormal(np.log(MEDIAN_CAP[sec]), .35)),
                         "both" if p and q else "producer" if p else "consumer", cl[2], round(rng.uniform(.7, .95), 2)))
    ind = pd.DataFrame(rows, columns=["industry_id", "name", "sector", "lat", "lon", "state", "capacity_tpy", "role",
                                      "region", "utilization"])
    # production: capacity x utilization x sector yield x seasonality
    prod = []
    for r in ind.itertuples():
        for mn, y in PRODUCE.get(r.sector, {}).items():
            m, g = mid[mn], rng.choice(["A", "B", "C"], p=[.3, .5, .2])
            q0, sc = r.capacity_tpy * r.utilization * y / 12, round(base[mid[mn]] * rng.uniform(0.02, 0.08), 1)
            for k, mo in enumerate(MONTHS):
                prod.append((r.industry_id, m, mo, round(max(q0 * SEASON.get(r.sector, np.ones(12))[k] * rng.normal(1, .06), 0), 1), g, sc))
    prod = pd.DataFrame(prod, columns=["industry_id", "material_id", "month", "quantity_tonnes", "quality_grade",
                                       "storage_cost_per_tonne"])
    # demand: capacity x utilization x sector consumption rate; specs depend on application
    dem = []
    for r in ind.itertuples():
        for mn, (rate, app, fuel) in CONSUME.get(r.sector, {}).items():
            m, x = mid[mn], mi.loc[mid[mn]]
            dem.append((r.industry_id, m, round(r.capacity_tpy * r.utilization * rate, 1),
                        round(x.moisture_pct * rng.uniform(.9, 1.6) + 1, 1),
                        round(x.calorific_value * rng.uniform(.6, .9), 1) if fuel else 0.0,
                        round(x.heavy_metals_ppm * rng.uniform(1.0, 1.8) + 5, 1), round(base[m] * rng.uniform(.9, 1.5), 1),
                        round(base[m] * rng.uniform(.01, .1), 1), app))
    dem = pd.DataFrame(dem, columns=["industry_id", "material_id", "required_tonnes", "max_moisture_pct",
                                     "min_calorific_value", "max_heavy_metals_ppm", "max_price_per_tonne",
                                     "pretreatment_cost_per_tonne", "application"])
    # market prices (regional random walk)
    mp = []
    for m in mats.material_id:
        for rg in ["North", "South", "East", "West"]:
            p = base[m] * rng.uniform(.9, 1.1)
            for mo in MONTHS:
                p *= rng.normal(1.003, .03)
                mp.append((m, mo, rg, round(p, 1), round(p * rng.uniform(1.5, 3.0), 1)))
    mp = pd.DataFrame(mp, columns=["material_id", "month", "region", "price_per_tonne", "virgin_substitute_price"])
    # transport routes (all ordered pairs, road <600 km else rail)
    xy = ind.set_index("industry_id")[["lat", "lon"]]
    rt = []
    for a in xy.index:
        for b in xy.index:
            if a == b: continue
            d = round(float(hav(xy.loc[a].values, xy.loc[b].values)) * 1.3, 1)
            mode = "road" if d < 600 else "rail"
            c, co2, v = (rng.normal(4.5, .4), .062, 350) if mode == "road" else (rng.normal(1.8, .2), .022, 500)
            rt.append((a, b, d, mode, round(c, 2), round(d / v + 1, 1), co2))
    rt = pd.DataFrame(rt, columns=["origin_industry_id", "dest_industry_id", "distance_km", "mode",
                                   "cost_per_tonne_km", "transit_days", "co2_per_tonne_km"])
    # regulations
    rg_rows, k = [], 1
    for m in mats.material_id:
        x = mi.loc[m]
        for jur in ["National", "State"]:
            for rtp, thr in [("max_heavy_metals_ppm", x.heavy_metals_ppm * rng.uniform(1.2, 2.5) + 10),
                             ("max_moisture_pct", x.moisture_pct * rng.uniform(1.3, 2.0) + 3),
                             ("max_storage_days", rng.choice([30, 60, 90, 180]))]:
                if rng.random() < .6:
                    rg_rows.append((f"R{k:04d}", m, x.hazard_class, jur, rtp, round(float(thr), 1), RULES[rtp][1],
                                    str(pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.integers(0, 2900))))[:10],
                                    DOC[x.hazard_class], int(rng.integers(1, 4)))); k += 1
    regs = pd.DataFrame(rg_rows, columns=["rule_id", "material_id", "hazard_class", "jurisdiction", "rule_type",
                                          "threshold", "unit", "effective_date", "source_doc", "version"])
    # pair features: only producer/consumer pairs that share a material (so every pair is sector-sensible)
    sec = ind.set_index("industry_id").sector
    rti = rt.set_index(["origin_industry_id", "dest_industry_id"])
    pa = prod.groupby(["industry_id", "material_id"]).agg(q=("quantity_tonnes", "mean"), g=("quality_grade", "first")).reset_index()
    pf = pa.merge(dem, on="material_id", suffixes=("_p", "_c"))
    pf = pf[pf.industry_id_p != pf.industry_id_c].rename(columns={"industry_id_p": "producer_id", "industry_id_c": "consumer_id",
                                                                   "q": "avg_monthly_tonnes", "g": "quality_grade"})
    pf["producer_sector"], pf["consumer_sector"] = pf.producer_id.map(sec), pf.consumer_id.map(sec)
    pf["distance_km"] = [rti.distance_km[(a, b)] for a, b in zip(pf.producer_id, pf.consumer_id)]
    pf["moisture_pct"] = [mi.loc[m, "moisture_pct"] * GRADE[g][0] for m, g in zip(pf.material_id, pf.quality_grade)]
    pf["heavy_metals_ppm"] = [mi.loc[m, "heavy_metals_ppm"] * GRADE[g][1] for m, g in zip(pf.material_id, pf.quality_grade)]
    pf["calorific_value"] = pf.material_id.map(mi.calorific_value)
    pf["moisture_gap"] = pf.max_moisture_pct - pf.moisture_pct
    pf["calorific_gap"] = pf.calorific_value - pf.min_calorific_value
    pf["heavy_metals_gap"] = pf.max_heavy_metals_ppm - pf.heavy_metals_ppm
    pf["feasible"] = ((pf.moisture_gap >= 0) & (pf.calorific_gap >= 0) & (pf.heavy_metals_gap >= 0)).astype(int)
    ireg = ind.set_index("industry_id").region
    pf["same_region"] = (pf.producer_id.map(ireg) == pf.consumer_id.map(ireg)).astype(int)
    pp = mp.groupby("material_id")[["price_per_tonne", "virgin_substitute_price"]].mean()
    pf["market_price"], pf["virgin_price"] = pf.material_id.map(pp.price_per_tonne), pf.material_id.map(pp.virgin_substitute_price)
    pf["transport_cost_per_tonne"] = [rti.distance_km[(a, b)] * rti.cost_per_tonne_km[(a, b)] for a, b in zip(pf.producer_id, pf.consumer_id)]
    pf["margin_per_tonne"] = pf.virgin_price - pf.market_price - pf.transport_cost_per_tonne - pf.pretreatment_cost_per_tonne
    e = ef.set_index("material_id")
    pf["co2_saving_per_tonne"] = pf.material_id.map(e.virgin_kgco2e_per_tonne - e.processing_kgco2e_per_tonne)
    pf["n_rules"] = pf.material_id.map(regs.groupby("material_id").size()).fillna(0).astype(int)
    # trade likelihood: feasible, profitable, and near (low-value bulk goods travel shorter distances)
    scale = np.clip(pf.market_price * 0.6, 300, 3000)
    p_tx = np.where(pf.feasible == 1, .6, .05) * np.exp(-pf.distance_km / scale) * (1 + (pf.margin_per_tonne > 0) * .5)
    pf["label"] = (rng.random(len(pf)) < np.clip(p_tx, 0, .95)).astype(int)
    tx = []
    for r in pf[pf.label == 1].itertuples():
        for mo in rng.choice(MONTHS, rng.integers(4, 12), replace=False):
            q = min(r.avg_monthly_tonnes, r.required_tonnes / 12) * rng.uniform(.3, .8)
            tx.append((f"T{len(tx)+1:06d}", r.producer_id, r.consumer_id, r.material_id, mo, round(q, 1),
                       round(min(r.max_price_per_tonne, r.market_price * rng.uniform(.9, 1.05)), 1)))
    tx = pd.DataFrame(tx, columns=["transaction_id", "producer_id", "consumer_id", "material_id", "month",
                                   "quantity_tonnes", "agreed_price"])
    # graph
    nodes = pd.concat([ind[["industry_id", "name", "sector", "state"]].rename(columns={"industry_id": "node_id"}).assign(node_type="industry"),
                       mats[["material_id", "name", "category"]].rename(columns={"material_id": "node_id", "category": "sector"}).assign(node_type="material", state="")])
    edges = pd.concat([prod[["industry_id", "material_id"]].drop_duplicates().assign(relation="produces").set_axis(["src", "dst", "relation"], axis=1),
                       dem[["industry_id", "material_id"]].drop_duplicates().assign(relation="requires").set_axis(["src", "dst", "relation"], axis=1),
                       tx.groupby(["producer_id", "consumer_id"]).quantity_tonnes.sum().reset_index().assign(relation="supplies_to").set_axis(["src", "dst", "weight", "relation"], axis=1)])
    # market timeseries (month x material = RL state)
    tci = dict(zip(MONTHS, np.cumprod(rng.normal(1.002, .02, 12))))
    ts = []
    for m in mats.material_id:
        for mo in MONTHS:
            s = mp[(mp.material_id == m) & (mp.month == mo)]
            ts.append((mo, m, s.price_per_tonne.mean(), s.virgin_substitute_price.mean(),
                       prod[(prod.material_id == m) & (prod.month == mo)].quantity_tonnes.sum(),
                       dem[dem.material_id == m].required_tonnes.sum() / 12, round(tci[mo], 3),
                       int(((regs.material_id == m) & (regs.effective_date <= mo + "-28")).sum())))
    ts = pd.DataFrame(ts, columns=["month", "material_id", "avg_price", "avg_virgin_price", "supply_tonnes",
                                   "demand_tonnes", "transport_cost_index", "active_rules"])
    # regulation text with NER spans
    txt = []
    for r in regs.itertuples():
        name, (phrase, unit) = mi.loc[r.material_id, "name"], RULES[r.rule_type]
        ob = rng.choice(["shall not accept", "must not exceed", "is prohibited from receiving"])
        s = (f"Facilities {ob} {name} with {phrase} above {r.threshold:g} {unit}." if r.rule_type != "max_storage_days"
             else f"Facilities {ob} {name} stored beyond {r.threshold:g} {unit}.")
        spans = [{"text": t, "label": l, "start": s.find(t), "end": s.find(t) + len(t)}
                 for t, l in [(name, "SUBSTANCE"), (f"{r.threshold:g}", "THRESHOLD"), (unit, "UNIT"), (ob, "OBLIGATION")]]
        txt.append({"rule_id": r.rule_id, "text": s, "entities": spans})
    raw = dict(industries=ind, materials=mats, production=prod, demand=dem, market_prices=mp, transport_routes=rt,
               regulations=regs, emission_factors=ef, transactions=tx)
    return raw, dict(pair_features=pf, nodes=nodes, edges=edges, market_timeseries=ts), txt


def validate(raw):
    """Every flow must be allowed by the sector maps (no nonsense producers/consumers)."""
    ind, mats = raw["industries"].set_index("industry_id"), raw["materials"].set_index("material_id").name
    for df, mp in ((raw["production"], PRODUCE), (raw["demand"], CONSUME)):
        for i, m in zip(df.industry_id, df.material_id):
            assert mats[m] in mp.get(ind.sector[i], {}), f"invalid flow {ind.sector[i]} - {mats[m]}"
    tx = raw["transactions"]
    assert tx.producer_id.map(ind.sector).notna().all()
    return True


def main(root="data"):
    raw, proc, txt = build()
    validate(raw)
    for d in ("raw", "processed"):
        Path(root, d).mkdir(parents=True, exist_ok=True)
    for k, v in raw.items(): v.to_csv(Path(root, "raw", f"{k}.csv"), index=False)
    for k, v in proc.items(): v.to_csv(Path(root, "processed", f"{k}.csv"), index=False)
    with open(Path(root, "processed", "regulation_text.jsonl"), "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in txt)
    pf = proc["pair_features"]
    print({k: len(v) for k, v in {**raw, **proc}.items()}, "| positives:", int(pf.label.sum()), "/", len(pf))


if __name__ == "__main__":
    main()