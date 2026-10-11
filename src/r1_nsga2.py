"""Objective 6: Multi-Objective Deep Reinforcement Learning & NSGA-II Engine.

Balances economic yield (INR profit) against environmental impact (CO2 savings)
and logistical risk (distance/transport). Computes the Pareto frontier of optimal
industrial symbiosis trading policies under dynamic conditions.

Implementation note: this module does NOT train a reinforcement-learning agent.
It ranks all candidate pairs with NSGA-II-style fast non-dominated sorting and then
orders each front by a fixed-weight composite score. The "dynamic conditions"
(prices, freight, regulations) are not simulated here; re-run the optimizer on
updated pair features to refresh the ranking.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def fast_non_dominated_sort(objectives):
    """Computes Pareto fronts from a 2D array of objectives (all objectives assumed to be maximized)."""
    N = objectives.shape[0]
    domination_counts = np.zeros(N, dtype=int)
    dominated_solutions = [[] for _ in range(N)]
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            p_better_or_equal = np.all(objectives[p] >= objectives[q])
            p_strictly_better = np.any(objectives[p] > objectives[q])
            
            if p_better_or_equal and p_strictly_better:
                dominated_solutions[p].append(q)
            elif np.all(objectives[q] >= objectives[p]) and np.any(objectives[q] > objectives[p]):
                domination_counts[p] += 1
                
        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if len(f) > 0]


class MultiObjectivePolicyOptimizer:
    """Multi-Objective Policy Optimizer combining NSGA-II selection and adaptive policy scoring."""
    def __init__(self, weights=(0.45, 0.40, 0.15)):
        self.weights = np.array(weights) / np.sum(weights)
        self.pareto_fronts = []
        self.best_trades = None

    def optimize(self, pair_df):
        df = pair_df.copy()
        max_dist = df.distance_km.max() + 50.0
        
        obj1 = df.margin_per_tonne.values
        obj2 = df.co2_saving_per_tonne.values
        obj3 = max_dist - df.distance_km.values
        
        objectives = np.column_stack([obj1, obj2, obj3])
        fronts = fast_non_dominated_sort(objectives)
        
        self.pareto_fronts = fronts
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)
        denom = np.maximum(max_vals - min_vals, 1e-6)
        norm_objs = (objectives - min_vals) / denom
        
        composite_scores = norm_objs @ self.weights
        df["pareto_rank"] = 999
        for rank, indices in enumerate(fronts):
            df.iloc[indices, df.columns.get_loc("pareto_rank")] = rank + 1
            
        df["rl_composite_score"] = composite_scores
        self.best_trades = df.sort_values(by=["pareto_rank", "rl_composite_score"], ascending=[True, False])
        return self.best_trades


def train_and_evaluate(data_path="data/processed/pair_features.csv", out="models/rl_nsga2_pareto.joblib"):
    df = pd.read_csv(data_path)
    optimizer = MultiObjectivePolicyOptimizer(weights=(0.45, 0.40, 0.15))
    results = optimizer.optimize(df)
    
    Path(out).parent.mkdir(exist_ok=True)
    # Save clean primitive dict for 100% portable unpickling
    joblib.dump({
        "weights": optimizer.weights.tolist(),
        "pareto_fronts": optimizer.pareto_fronts,
        "top_pareto_trades": results.head(50)
    }, out)
    
    p1_count = len(optimizer.pareto_fronts[0]) if len(optimizer.pareto_fronts) > 0 else 0
    return {
        "total_evaluated_pairs": len(df),
        "pareto_front_rank1_count": p1_count,
        "num_fronts": len(optimizer.pareto_fronts)
    }


def load_rl_model(path="models/rl_nsga2_pareto.joblib"):
    return joblib.load(path)


if __name__ == "__main__":
    metrics = train_and_evaluate()
    print("Multi-Objective Optimization Evaluation:", metrics)
