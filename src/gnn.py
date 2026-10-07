import json
from pathlib import Path
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


def build_knowledge_graph(nodes_path="data/processed/nodes.csv", edges_path="data/processed/edges.csv"):
    """Constructs a NetworkX graph with node attributes and multi-relational edges."""
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    G = nx.MultiDiGraph()
    for _, r in nodes_df.iterrows():
        G.add_node(r.node_id, name=r.get("name", ""), sector=r.get("sector", ""),
                   state=r.get("state", ""), node_type=r.get("node_type", "industry"))
        
    for _, r in edges_df.iterrows():
        w = float(r.weight) if pd.notna(r.get("weight")) and r.weight != "" else 1.0
        G.add_edge(r.src, r.dst, relation=r.relation, weight=w)
        
    return G, nodes_df, edges_df


class GraphEmbeddingLinkPredictor:
    """Lightweight Spectral & Relational GNN / Node Embedding link predictor."""
    def __init__(self, embedding_dim=32, num_hops=2, random_state=42):
        self.embedding_dim = embedding_dim
        self.num_hops = num_hops
        self.random_state = random_state
        self.node2idx = {}
        self.idx2node = []
        self.embeddings = None
        self.classifier = None
        self.feature_names = ["hadamard", "dot", "l1", "l2", "cosine"]

    def _compute_embeddings(self, G):
        self.idx2node = list(G.nodes())
        self.node2idx = {n: i for i, n in enumerate(self.idx2node)}
        N = len(self.idx2node)
        
        # Adjacency with relation weights
        A = np.zeros((N, N), dtype=np.float32)
        for u, v, data in G.edges(data=True):
            i, j = self.node2idx[u], self.node2idx[v]
            rel = data.get("relation", "general")
            w = data.get("weight", 1.0)
            mult = 2.0 if rel in ("produces", "consumes") else 1.0
            A[i, j] += w * mult
            A[j, i] += w * mult  # Undirected propagation
            
        # Degree normalization (Random walk / GCN style: D^-1/2 A D^-1/2)
        deg = np.sum(A, axis=1)
        deg_inv_sqrt = np.power(np.maximum(deg, 1e-6), -0.5)
        D = np.diag(deg_inv_sqrt)
        A_norm = D @ A @ D
        
        # Initial node features: one-hot or random orthogonal + sector/type encoding
        rng = np.random.RandomState(self.random_state)
        X = rng.normal(0, 0.1, size=(N, self.embedding_dim)).astype(np.float32)
        
        # Multi-hop GCN propagation: H^(l+1) = sigma(A_norm * H^l * W^l)
        H = X
        for _ in range(self.num_hops):
            H = A_norm @ H
            # Non-linear activation (tanh)
            H = np.tanh(H)
            
        self.embeddings = H
        return self.embeddings

    def _pair_features(self, u_idx, v_idx):
        h_u = self.embeddings[u_idx]
        h_v = self.embeddings[v_idx]
        hadamard = h_u * h_v
        dot = np.sum(hadamard, axis=-1, keepdims=True)
        l1 = np.abs(h_u - h_v)
        l2 = (h_u - h_v) ** 2
        norm_u = np.linalg.norm(h_u, axis=-1, keepdims=True) + 1e-8
        norm_v = np.linalg.norm(h_v, axis=-1, keepdims=True) + 1e-8
        cosine = dot / (norm_u * norm_v)
        return np.hstack([hadamard, dot, l1, l2, cosine])

    def fit(self, G, pair_features_path="data/processed/pair_features.csv"):
        self._compute_embeddings(G)
        
        df = pd.read_csv(pair_features_path)
        # Filter pairs present in graph
        valid = df[df.producer_id.isin(self.node2idx) & df.consumer_id.isin(self.node2idx)]
        
        u_idx = np.array([self.node2idx[p] for p in valid.producer_id])
        v_idx = np.array([self.node2idx[c] for c in valid.consumer_id])
        
        X = self._pair_features(u_idx, v_idx)
        y = valid.label.values
        
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=self.random_state, stratify=y)
        self.classifier = LogisticRegression(max_iter=500, random_state=self.random_state)
        self.classifier.fit(X_tr, y_tr)
        
        p_te = self.classifier.predict_proba(X_te)[:, 1]
        metrics = {
            "roc_auc": float(roc_auc_score(y_te, p_te)),
            "avg_precision": float(average_precision_score(y_te, p_te)),
            "n_nodes": len(self.idx2node),
            "embedding_dim": self.embedding_dim
        }
        return metrics

    def predict_link_probability(self, src_id, dst_id):
        if src_id not in self.node2idx or dst_id not in self.node2idx:
            return 0.0
        u_idx = np.array([self.node2idx[src_id]])
        v_idx = np.array([self.node2idx[dst_id]])
        X = self._pair_features(u_idx, v_idx)
        return float(self.classifier.predict_proba(X)[0, 1])

    def to_dict(self):
        return {
            "embedding_dim": self.embedding_dim,
            "num_hops": self.num_hops,
            "random_state": self.random_state,
            "node2idx": self.node2idx,
            "idx2node": self.idx2node,
            "embeddings": self.embeddings,
            "classifier": self.classifier,
            "feature_names": self.feature_names
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(embedding_dim=d["embedding_dim"], num_hops=d["num_hops"], random_state=d["random_state"])
        obj.node2idx = d["node2idx"]
        obj.idx2node = d["idx2node"]
        obj.embeddings = d["embeddings"]
        obj.classifier = d["classifier"]
        obj.feature_names = d.get("feature_names", [])
        return obj


def train(nodes_path="data/processed/nodes.csv", edges_path="data/processed/edges.csv",
          pairs_path="data/processed/pair_features.csv", out="models/gnn_link_predictor.joblib"):
    G, _, _ = build_knowledge_graph(nodes_path, edges_path)
    model = GraphEmbeddingLinkPredictor(embedding_dim=32, num_hops=2)
    metrics = model.fit(G, pairs_path)
    Path(out).parent.mkdir(exist_ok=True)
    joblib.dump(model.to_dict(), out)
    return metrics


def load_gnn_model(path="models/gnn_link_predictor.joblib"):
    data = joblib.load(path)
    if isinstance(data, dict) and "embeddings" in data:
        return GraphEmbeddingLinkPredictor.from_dict(data)
    return data


if __name__ == "__main__":
    rep = train()
    print("GNN Link Prediction Evaluation:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})
