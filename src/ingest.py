"""Real-World Dataset Ingestion & Validation Pipeline.

This module validates external/real-world industrial CSV datasets, verifies foreign
key integrity, computes pair features, and generates knowledge graph edges ready
for model training and inference.

Usage:
    python src/ingest.py --data_dir data/raw --out_dir data/processed
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_RAW_FILES = [
    "industries.csv",
    "materials.csv",
    "production.csv",
    "demand.csv",
    "transport_routes.csv",
    "emission_factors.csv",
    "market_prices.csv",
    "regulations.csv"
]

def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    p = np.pi / 180
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = np.sin(dlat / 2)**2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def validate_raw_schema(data_dir: Path) -> dict:
    """Validates presence and schema of all required raw CSV files."""
    errors = []
    data = {}
    for filename in REQUIRED_RAW_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            errors.append(f"Missing required file: {filename}")
            continue
        try:
            df = pd.read_csv(filepath)
            data[filename.replace(".csv", "")] = df
        except Exception as e:
            errors.append(f"Failed to read {filename}: {str(e)}")

    if errors:
        return {"valid": False, "errors": errors, "data": None}

    # Foreign key validations
    ind_ids = set(data["industries"]["industry_id"])
    mat_ids = set(data["materials"]["material_id"])

    # Check production references
    prod_invalid_ind = set(data["production"]["industry_id"]) - ind_ids
    prod_invalid_mat = set(data["production"]["material_id"]) - mat_ids
    if prod_invalid_ind:
        errors.append(f"production.csv has unknown industry_ids: {list(prod_invalid_ind)[:5]}")
    if prod_invalid_mat:
        errors.append(f"production.csv has unknown material_ids: {list(prod_invalid_mat)[:5]}")

    # Check demand references
    dem_invalid_ind = set(data["demand"]["industry_id"]) - ind_ids
    dem_invalid_mat = set(data["demand"]["material_id"]) - mat_ids
    if dem_invalid_ind:
        errors.append(f"demand.csv has unknown industry_ids: {list(dem_invalid_ind)[:5]}")
    if dem_invalid_mat:
        errors.append(f"demand.csv has unknown material_ids: {list(dem_invalid_mat)[:5]}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "data": data,
        "summary": {
            "industries_count": len(data["industries"]),
            "materials_count": len(data["materials"]),
            "production_records": len(data["production"]),
            "demand_records": len(data["demand"])
        }
    }


def process_real_data(data_dir: Path, out_dir: Path):
    """Processes validated raw data into model-ready pair features and graphs."""
    validation = validate_raw_schema(data_dir)
    if not validation["valid"]:
        print("Schema validation failed:")
        for err in validation["errors"]:
            print(f"  - {err}")
        sys.exit(1)

    print("Schema validation passed.")
    print("Dataset Summary:", validation["summary"])
    raw = validation["data"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ind = raw["industries"].set_index("industry_id")
    mats = raw["materials"].set_index("material_id")
    dem = raw["demand"]
    prod = raw["production"]
    rt = raw["transport_routes"].set_index(["origin_industry_id", "dest_industry_id"])
    ef = raw["emission_factors"].set_index("material_id")
    mp = raw["market_prices"]
    regs = raw["regulations"]

    # 1. Aggregate annual average production per facility-material
    pa = prod.groupby(["industry_id", "material_id"]).agg(
        avg_monthly_tonnes=("quantity_tonnes", "mean"),
        quality_grade=("quality_grade", "first")
    ).reset_index()

    # 2. Match potential pairs on material_id
    pf = pa.merge(dem, on="material_id", suffixes=("_p", "_c"))
    pf = pf[pf.industry_id_p != pf.industry_id_c].rename(
        columns={"industry_id_p": "producer_id", "industry_id_c": "consumer_id"}
    )

    if pf.empty:
        print("No overlapping material pairs found between producers and consumers.")
        return

    grade_mult = {"A": (0.9, 1.0), "B": (1.0, 1.0), "C": (1.15, 1.3)}
    sec = ind["sector"].to_dict()
    reg = ind["region"].to_dict()

    pf["producer_sector"] = pf.producer_id.map(sec)
    pf["consumer_sector"] = pf.consumer_id.map(sec)
    pf["same_region"] = (pf.producer_id.map(reg) == pf.consumer_id.map(reg)).astype(int)

    # Route distance & freight calculation
    def get_route_info(p_id, c_id):
        if (p_id, c_id) in rt.index:
            row = rt.loc[(p_id, c_id)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return float(row["distance_km"]), float(row["cost_per_tonne_km"])
        else:
            # Fallback to Haversine calculation
            p_lat, p_lon = ind.loc[p_id, "lat"], ind.loc[p_id, "lon"]
            c_lat, c_lon = ind.loc[c_id, "lat"], ind.loc[c_id, "lon"]
            dist = float(haversine_distance(p_lat, p_lon, c_lat, c_lon) * 1.3)
            cost_rate = 4.2 if dist < 600 else 2.1
            return round(dist, 1), cost_rate

    route_res = [get_route_info(p, c) for p, c in zip(pf.producer_id, pf.consumer_id)]
    pf["distance_km"] = [r[0] for r in route_res]
    pf["transport_cost_per_tonne"] = [r[0] * r[1] for r in route_res]

    # Material quality gaps
    pf["moisture_pct"] = [
        mats.loc[m, "moisture_pct"] * grade_mult.get(g, (1.0, 1.0))[0]
        for m, g in zip(pf.material_id, pf.quality_grade)
    ]
    pf["heavy_metals_ppm"] = [
        mats.loc[m, "heavy_metals_ppm"] * grade_mult.get(g, (1.0, 1.0))[1]
        for m, g in zip(pf.material_id, pf.quality_grade)
    ]
    pf["calorific_value"] = pf.material_id.map(mats["calorific_value"])

    pf["moisture_gap"] = pf.max_moisture_pct - pf.moisture_pct
    pf["calorific_gap"] = pf.calorific_value - pf.min_calorific_value
    pf["heavy_metals_gap"] = pf.max_heavy_metals_ppm - pf.heavy_metals_ppm
    pf["feasible"] = (
        (pf.moisture_gap >= 0) & (pf.calorific_gap >= 0) & (pf.heavy_metals_gap >= 0)
    ).astype(int)

    # Prices and margins
    price_map = mp.groupby("material_id")[["price_per_tonne", "virgin_substitute_price"]].mean()
    pf["market_price"] = pf.material_id.map(price_map["price_per_tonne"]).fillna(1000.0)
    pf["virgin_price"] = pf.material_id.map(price_map["virgin_substitute_price"]).fillna(2500.0)
    pf["margin_per_tonne"] = (
        pf.virgin_price - pf.market_price - pf.transport_cost_per_tonne - pf.pretreatment_cost_per_tonne
    )

    # CO2 savings
    if "virgin_kgco2e_per_tonne" in ef.columns and "processing_kgco2e_per_tonne" in ef.columns:
        co2_net = ef["virgin_kgco2e_per_tonne"] - ef["processing_kgco2e_per_tonne"]
        pf["co2_saving_per_tonne"] = pf.material_id.map(co2_net).fillna(500.0)
    else:
        pf["co2_saving_per_tonne"] = 500.0

    # Regulations count
    reg_counts = regs.groupby("material_id").size() if len(regs) > 0 else pd.Series(dtype=int)
    pf["n_rules"] = pf.material_id.map(reg_counts).fillna(0).astype(int)

    # Label initialization (if transactions exist, match against real trades)
    if "transactions" in raw and len(raw["transactions"]) > 0:
        real_trades = set(zip(raw["transactions"]["producer_id"], raw["transactions"]["consumer_id"], raw["transactions"]["material_id"]))
        pf["label"] = [
            1 if (p, c, m) in real_trades else 0
            for p, c, m in zip(pf.producer_id, pf.consumer_id, pf.material_id)
        ]
    else:
        # Default heuristic feasibility label
        pf["label"] = ((pf.feasible == 1) & (pf.margin_per_tonne > 0) & (pf.distance_km < 800)).astype(int)

    # Output pair_features.csv
    pf.to_csv(out_dir / "pair_features.csv", index=False)
    print(f"Exported pair_features.csv with {len(pf)} candidate pairs ({int(pf.label.sum())} positive trades).")

    # Nodes & Edges
    raw_ind = raw["industries"]
    raw_mats = raw["materials"]
    nodes = pd.concat([
        raw_ind[["industry_id", "name", "sector", "state"]].rename(columns={"industry_id": "node_id"}).assign(node_type="industry"),
        raw_mats[["material_id", "name", "category"]].rename(columns={"material_id": "node_id", "category": "sector"}).assign(node_type="material", state="")
    ])
    nodes.to_csv(out_dir / "nodes.csv", index=False)

    edges_list = [
        prod[["industry_id", "material_id"]].drop_duplicates().assign(relation="produces", weight=1.0).set_axis(["src", "dst", "relation", "weight"], axis=1),
        dem[["industry_id", "material_id"]].drop_duplicates().assign(relation="requires", weight=1.0).set_axis(["src", "dst", "relation", "weight"], axis=1)
    ]
    if "transactions" in raw and len(raw["transactions"]) > 0:
        tx_agg = raw["transactions"].groupby(["producer_id", "consumer_id"]).quantity_tonnes.sum().reset_index()
        tx_agg["relation"] = "supplies_to"
        edges_list.append(tx_agg.rename(columns={"producer_id": "src", "consumer_id": "dst", "quantity_tonnes": "weight"}))

    edges = pd.concat(edges_list)
    edges.to_csv(out_dir / "edges.csv", index=False)
    print(f"Exported {len(nodes)} nodes and {len(edges)} edges to {out_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and validate real raw datasets.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory with raw CSV files")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Directory for processed CSV files")
    args = parser.parse_args()
    process_real_data(Path(args.data_dir), Path(args.out_dir))

