import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

NUM = ["avg_monthly_tonnes", "required_tonnes", "supply_demand_ratio", "distance_km", "moisture_gap", "calorific_gap",
       "heavy_metals_gap", "feasible", "same_region", "market_price", "virgin_price", "price_headroom",
       "transport_cost_per_tonne", "pretreatment_cost_per_tonne", "margin_per_tonne", "co2_saving_per_tonne", "n_rules"]
CAT = ["material_id", "producer_sector", "consumer_sector", "quality_grade"]
SHOW = ["producer_id", "consumer_id", "material_id", "consumer_sector", "application", "score", "distance_km",
        "margin_per_tonne", "co2_saving_per_tonne", "feasible"]


def load(path="data/processed/pair_features.csv"):
    d = pd.read_csv(path)
    d["supply_demand_ratio"] = d.avg_monthly_tonnes * 12 / d.required_tonnes
    d["price_headroom"] = d.max_price_per_tonne - d.market_price
    return d


def make_model():
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")
    return make_pipeline(pre, HistGradientBoostingClassifier(max_depth=4, learning_rate=.08, max_iter=200, random_state=42))


def hit_at_k(d, score, k=3):
    """Share of producers (with >=1 true buyer) whose top-k contains a true buyer."""
    d = d.assign(s=score); hits = []
    for _, g in d.groupby("producer_id"):
        if g.label.sum(): hits.append(g.nlargest(k, "s").label.max())
    return float(np.mean(hits)) if hits else float("nan")


def train(data="data/processed/pair_features.csv", out="models/recommender.joblib"):
    d = load(data); X, y = d[NUM + CAT], d.label
    tr, te = next(GroupShuffleSplit(1, test_size=.25, random_state=42).split(X, y, d.producer_id))  # unseen producers
    m = make_model().fit(X.iloc[tr], y.iloc[tr]); p = m.predict_proba(X.iloc[te])[:, 1]
    rep = {"auc": roc_auc_score(y.iloc[te], p), "avg_precision": average_precision_score(y.iloc[te], p),
           "hit@3": hit_at_k(d.iloc[te], p), "n_train": len(tr), "n_test": len(te)}
    m = make_model().fit(X, y)  # refit on all data for deployment
    Path(out).parent.mkdir(exist_ok=True); joblib.dump(m, out)
    return rep


def recommend(producer_id, d, model, material_id=None, top_k=5):
    g = d[d.producer_id == producer_id]
    if material_id: g = g[g.material_id == material_id]
    g = g.assign(score=model.predict_proba(g[NUM + CAT])[:, 1])
    return g.nlargest(top_k, "score")[SHOW].round(3)


if __name__ == "__main__":
    print({k: round(v, 3) for k, v in train().items()})
    d, m = load(), joblib.load("models/recommender.joblib")
    print(recommend(d.producer_id.iloc[0], d, m).to_string(index=False))