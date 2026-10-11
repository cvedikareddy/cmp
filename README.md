# Industrial Symbiosis AI Platform

Key Objectives:
1. AI-Driven Industrial Symbiosis Recommendation Engine - Trains on production, material-value, and transportation data to recommend the most viable and sustainable producer–consumer matches by evaluating material properties, demand, regulations, and profitability.
2. Circular Economy Profitability Simulator - Lets users simulate alternative collaboration scenarios across transport costs, requirements, and ROI to explore the marketable value and trade sensitivity of a producer's by-products.
3. Multi-Source Hybrid Dataset - Integrates transportation, logistics, market price, regulatory, and material composition data into a unified foundation for training and evaluating industrial connections.
4. Relation Matching and Trade-Off Analysis - Transparently explains why each match was recommended, detailing compatibility, influencing factors, and comparative economic/environmental benefits so industries can make informed decisions.
5. Industrial Symbiosis Knowledge Graph (GNN) - Uses graph-based link prediction and zero-shot learning to autonomously discover unseen waste-supply relationships between industries with no prior transaction history.
6. Multi-Objective Deep Reinforcement Learning Engine - Uses NSGA-II-style policy networks to continuously balance economic yield against carbon footprint, adapting recommendations in real time as costs, prices, and regulations shift.
7. Regulatory Constraint Extraction (LLM/NER) - Continuously crawls and parses environmental law documents via a fine-tuned NER pipeline to keep the recommendation engine's active regulatory constraints automatically up to date.

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py                    # terminal demo of all components
streamlit run app.py              # interactive dashboard
```

Run everything from the repository root (paths are relative). To regenerate data
and retrain the models:

```bash
python src/dataset.py
python src/recommender.py
python src/gnn.py
python src/r1_nsga2.py
python src/regulatory_ner.py
```

## Project structure

| Path | Purpose |
|------|---------|
| `src/dataset.py` | Generates the synthetic raw and processed datasets |
| `src/recommender.py` | Gradient-boosting match recommender (`recommend`) |
| `src/simulator.py` | Profitability economics, sensitivity, Monte Carlo |
| `src/explain.py` | Plain-language strengths and cautions per match |
| `src/gnn.py` | Knowledge graph and graph-embedding link predictor |
| `src/r1_nsga2.py` | Pareto ranking over profit, CO2 and distance |
| `src/regulatory_ner.py` | Rule-based constraint extraction from legal text |
| `main.py` / `app.py` | CLI pipeline / Streamlit dashboard |

## Implementation status

The objectives above describe the target. The current implementation is:

- **Data:** fully synthetic (seeded), so scores reflect the generator's rules.
- **Recommender:** gradient-boosting classifier on pair features (implemented).
- **Simulator:** closed-form economics plus sensitivity and Monte Carlo (implemented).
- **GNN (objective 5):** fixed graph-embedding propagation with a trained logistic-regression head; no learned GNN weights, and metrics are affected by label leakage (see `src/gnn.py`).
- **Multi-objective RL (objective 6):** NSGA-II-style non-dominated sorting with a weighted score; no RL agent yet.
- **Regulatory NER (objective 7):** regex and keyword extraction; no fine-tuned model or crawler yet.
