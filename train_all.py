"""One-click pipeline to generate/ingest data and train all models.

Usage:
    # Scale synthetic data to 150 facilities and train:
    python train_all.py --scale 150

    # Train directly on existing or real ingested data:
    python train_all.py --skip_data

    # Ingest custom real data directory and train:
    python train_all.py --real_data_dir path/to/real_raw_csvs
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.dataset import main as generate_dataset
from src.ingest import process_real_data
from src.recommender import train as train_recommender
from src.gnn import train as train_gnn
from src.r1_nsga2 import train_and_evaluate as train_rl
from src.regulatory_ner import train_and_save as train_ner


def run_pipeline(scale: int = None, real_data_dir: str = None, skip_data: bool = False):
    print("=" * 70)
    print("INDUSTRIAL SYMBIOSIS AI PLATFORM — TRAINING PIPELINE")
    print("=" * 70)

    # 1. Data Stage
    if skip_data:
        print("\n[Step 1/5] Skipping data generation (using current data/processed).")
    elif real_data_dir:
        print(f"\n[Step 1/5] Ingesting real data from '{real_data_dir}'...")
        process_real_data(Path(real_data_dir), BASE_DIR / "data" / "processed")
    else:
        n_facilities = scale if scale else 60
        print(f"\n[Step 1/5] Generating dataset with {n_facilities} industrial facilities...")
        generate_dataset(root=str(BASE_DIR / "data"), n_ind=n_facilities)

    # 2. Recommender Model
    print("\n[Step 2/5] Training Gradient-Boosting Match Recommender...")
    rec_metrics = train_recommender(
        data=str(BASE_DIR / "data" / "processed" / "pair_features.csv"),
        out=str(BASE_DIR / "models" / "recommender.joblib")
    )
    print(f"  [+] Recommender AUC: {rec_metrics['auc']:.3f} | Avg Precision: {rec_metrics['avg_precision']:.3f} | Hit@3: {rec_metrics['hit@3']:.3f}")

    # 3. Knowledge Graph GNN Link Predictor
    print("\n[Step 3/5] Training GNN Link Predictor...")
    gnn_metrics = train_gnn(
        nodes_path=str(BASE_DIR / "data" / "processed" / "nodes.csv"),
        edges_path=str(BASE_DIR / "data" / "processed" / "edges.csv"),
        out=str(BASE_DIR / "models" / "gnn_link_predictor.joblib")
    )
    print(f"  [+] GNN Graph: {gnn_metrics['n_nodes']} nodes | Link ROC-AUC: {gnn_metrics['roc_auc']:.3f} | Avg Precision: {gnn_metrics['avg_precision']:.3f}")

    # 4. Multi-Objective Pareto Optimizer (NSGA-II)
    print("\n[Step 4/5] Computing Multi-Objective Pareto Frontier...")
    rl_metrics = train_rl(
        data_path=str(BASE_DIR / "data" / "processed" / "pair_features.csv"),
        out=str(BASE_DIR / "models" / "rl_nsga2_pareto.joblib")
    )
    print(f"  [+] Pareto Fronts: {rl_metrics['num_fronts']} fronts | Rank-1 Optimal Deals: {rl_metrics['pareto_front_rank1_count']}")

    # 5. Regulatory NER Engine
    print("\n[Step 5/5] Extracting Regulatory Constraints & Directive Rules...")
    ner_df = train_ner(
        data_path=str(BASE_DIR / "data" / "processed" / "regulation_text.jsonl"),
        out=str(BASE_DIR / "models" / "regulatory_ner.joblib")
    )
    print(f"  [+] Regulatory Rules Extracted: {len(ner_df)}")

    print("\n" + "=" * 70)
    print("ALL MODELS SUCCESSFULLY RETRAINED AND DEPLOYED!")
    print("=" * 70)
    print("Run 'streamlit run app.py' to launch the updated dashboard.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end data generation and model retraining.")
    parser.add_argument("--scale", type=int, default=None, help="Scale synthetic facilities (e.g. 100, 200, 500)")
    parser.add_argument("--real_data_dir", type=str, default=None, help="Directory containing real raw CSVs to ingest")
    parser.add_argument("--skip_data", action="store_true", help="Skip data generation and retrain on existing data")
    args = parser.parse_args()

    run_pipeline(scale=args.scale, real_data_dir=args.real_data_dir, skip_data=args.skip_data)
