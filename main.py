"""Industrial Symbiosis AI Platform - Pure Python CLI & Pipeline.

Run directly with:
    python main.py
"""
import sys
from pathlib import Path
import joblib
import pandas as pd

from src.dataset import MAT, PRODUCE, CONSUME
from src.recommender import load as load_pairs, recommend
from src.simulator import load_data, scenario as build_scenario, evaluate
from src.explain import MatchExplainer
from src.gnn import load_gnn_model, build_knowledge_graph
from src.regulatory_ner import RegulatoryEntityExtractor


def run_pipeline():
    print("=" * 75)
    print("INDUSTRIAL SYMBIOSIS AI PLATFORM (PYTHON PIPELINE)")
    print("=" * 75)

    # 1. Recommender Engine
    print("\n[1] RECOMMENDATION ENGINE")
    print("-" * 50)
    data = load_pairs("data/processed/pair_features.csv")
    rec_model = joblib.load("models/recommender.joblib")
    sample_producer = "I001"
    recs = recommend(producer_id=sample_producer, d=data, model=rec_model, top_k=3)
    print(f"Top 3 Symbiosis Matches for Producer '{sample_producer}':")
    print(recs[["producer_id", "consumer_id", "material_id", "consumer_sector", "score", "distance_km", "margin_per_tonne"]].to_string(index=False))

    # 2. Profitability Simulator
    print("\n[2] CIRCULAR ECONOMY PROFITABILITY SIMULATOR")
    print("-" * 50)
    top_match = recs.iloc[0]
    # Build the scenario from the matched pair's own data (market price, virgin price,
    # freight rate, pretreatment cost, price ceiling, feasibility) instead of hardcoded values.
    scenario = build_scenario(
        load_data("data"),
        top_match.producer_id,
        top_match.consumer_id,
        top_match.material_id,
        carbon_price=400.0,
    )
    sim_res = evaluate(scenario)
    print(f"Simulation Deal: {scenario.producer_id} -> {scenario.consumer_id} ({scenario.volume_tpy:,.0f} tonnes/year)")
    print(f"  * Annual Consumer Savings : INR {sim_res['annual_consumer_saving']:,.2f}")
    print(f"  * Annual Producer Gain    : INR {sim_res['annual_producer_gain']:,.2f}")
    print(f"  * Annual CO2 Avoided      : {sim_res['annual_co2_saved_t']:,.1f} tonnes CO2e")
    print(f"  * Viability Verdict       : {'VIABLE' if sim_res['viable'] else 'NON-VIABLE'}")

    # 3. Explainability Engine
    print("\n[3] MATCH EXPLAINABILITY & TRADE-OFF ATTRIBUTION")
    print("-" * 50)
    explainer = MatchExplainer("models/recommender.joblib")
    match_row = data[(data.producer_id == top_match.producer_id) & (data.consumer_id == top_match.consumer_id)].iloc[0].to_dict()
    exp = explainer.explain_pair(match_row)
    print(f"Viability Confidence Score: {exp['score'] * 100:.1f}%")
    print("Positive Drivers:")
    for s in exp["strengths"]:
        print(f"  [+] {s}")
    if exp["cautions"]:
        print("Cautions:")
        for c in exp["cautions"]:
            print(f"  [-] {c}")

    # 4. GNN Knowledge Graph Link Prediction
    print("\n[4] KNOWLEDGE GRAPH GNN LINK PREDICTION")
    print("-" * 50)
    gnn_model = load_gnn_model("models/gnn_link_predictor.joblib")
    src_node, dst_node = "I001", "I020"
    link_prob = gnn_model.predict_link_probability(src_node, dst_node)
    print(f"Predicted Zero-Shot Synergy Likelihood ({src_node} -> {dst_node}): {link_prob * 100:.2f}%")

    # 5. Multi-Objective RL / NSGA-II Pareto Frontier
    print("\n[5] MULTI-OBJECTIVE PARETO OPTIMAL TRADES (NSGA-II)")
    print("-" * 50)
    rl_data = joblib.load("models/rl_nsga2_pareto.joblib")
    top_pareto = rl_data["top_pareto_trades"]
    print("Top Pareto Rank-1 Deals (Balancing Profit vs CO2 vs Distance):")
    print(top_pareto[["producer_id", "consumer_id", "material_id", "pareto_rank", "margin_per_tonne", "co2_saving_per_tonne", "distance_km"]].head(4).to_string(index=False))

    # 6. Regulatory NER Extraction
    print("\n[6] REGULATORY CONSTRAINT EXTRACTION (NER)")
    print("-" * 50)
    extractor = RegulatoryEntityExtractor()
    sample_legal_text = "Plants shall not accept spent acid with heavy metal content above 120.0 ppm."
    extracted = extractor.parse_text(sample_legal_text)
    print(f"Input Legal Clause : \"{sample_legal_text}\"")
    print(f"Extracted Entities  : Substance='{extracted['substance']}', Property='{extracted['property']}', Threshold={extracted['threshold']} {extracted['unit']}, Obligation='{extracted['obligation']}'")

    print("\n" + "=" * 75)
    print("All AI engines successfully executed in pure Python.")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
