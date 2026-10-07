
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender import NUM, CAT, load, recommend


class MatchExplainer:
    """Computes transparent attribution and trade-off analysis for recommended pairs."""
    def __init__(self, model_path="models/recommender.joblib"):
        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps["columntransformer"]
        self.classifier = self.pipeline.named_steps["histgradientboostingclassifier"]

    def explain_pair(self, row_dict):
        """Returns feature breakdown and human-readable trade-off rationale for a single pair."""
        df_single = pd.DataFrame([row_dict])
        prob = float(self.pipeline.predict_proba(df_single[NUM + CAT])[0, 1])
        
        # Calculate key metric drivers
        distance = float(row_dict.get("distance_km", 0))
        margin = float(row_dict.get("margin_per_tonne", 0))
        co2_saving = float(row_dict.get("co2_saving_per_tonne", 0))
        feasible = bool(row_dict.get("feasible", 1))
        same_region = bool(row_dict.get("same_region", 0))
        
        strengths = []
        cautions = []
        
        if margin > 2000:
            strengths.append(f"High profit margin: INR {margin:,.0f}/t")
        elif margin > 0:
            strengths.append(f"Positive profit margin: INR {margin:,.0f}/t")
        else:
            cautions.append(f"Negative margin: INR {margin:,.0f}/t requires subsidy or higher gate fee")
            
        if co2_saving > 1000:
            strengths.append(f"Exceptional carbon abatement: {co2_saving:,.0f} kg CO2e/t avoided")
        elif co2_saving > 0:
            strengths.append(f"Net carbon reduction: {co2_saving:,.0f} kg CO2e/t avoided")
        else:
            cautions.append("High transport emissions offset material carbon benefits")
            
        if distance < 300:
            strengths.append(f"Close geographic proximity ({distance:.1f} km, low freight burden)")
        elif distance > 1000:
            cautions.append(f"Long haul route ({distance:.1f} km) increases transit risk and cost")
            
        if same_region:
            strengths.append("Inter-state regulatory friction minimized (same region)")
            
        if not feasible:
            cautions.append("Requires quality/pretreatment upgrade to meet buyer contamination specs")
            
        return {
            "score": round(prob, 4),
            "viable": feasible and margin > 0,
            "strengths": strengths,
            "cautions": cautions,
            "metrics": {
                "distance_km": distance,
                "margin_per_tonne_inr": margin,
                "co2_saving_kg_per_t": co2_saving,
                "supply_demand_ratio": float(row_dict.get("supply_demand_ratio", 1.0))
            }
        }


def explain_top_matches(producer_id, data_path="data/processed/pair_features.csv", top_k=3):
    d = load(data_path)
    model = joblib.load("models/recommender.joblib")
    recs = recommend(producer_id, d, model, top_k=top_k)
    explainer = MatchExplainer("models/recommender.joblib")
    
    explanations = []
    for _, row in recs.iterrows():
        full_row = d[(d.producer_id == row.producer_id) & 
                     (d.consumer_id == row.consumer_id) & 
                     (d.material_id == row.material_id)].iloc[0].to_dict()
        exp = explainer.explain_pair(full_row)
        explanations.append({
            "consumer_id": row.consumer_id,
            "material_id": row.material_id,
            "application": row.application,
            "explanation": exp
        })
    return explanations


if __name__ == "__main__":
    d = load()
    p_sample = d.producer_id.iloc[0]
    print(f"--- Explaining Recommendations for Producer {p_sample} ---")
    res = explain_top_matches(p_sample, top_k=2)
    for r in res:
        print(f"\nMatch: {r['consumer_id']} ({r['material_id']} for {r['application']})")
        print(f"Score: {r['explanation']['score']}")
        print("Strengths:", r['explanation']['strengths'])
        print("Cautions:", r['explanation']['cautions'])
