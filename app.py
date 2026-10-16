"""Industrial Symbiosis Decision Platform.

An intuitive circular economy decision platform featuring:
1. Industrial exchange recommendations
2. Circular profitability and scenario simulation
3. Match rationale and trade-off attribution
4. Industrial knowledge graph exploration
5. Multi-objective Pareto optimization
6. Regulatory compliance search and clause parsing
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import joblib

from src.recommender import load as load_dataset, recommend, NUM, CAT
from src.simulator import Scenario, evaluate, sensitivity, monte_carlo, fair_price, sweep, load_data as load_sim_data, scenario as build_sim_scenario
from src.explain import MatchExplainer
from src.gnn import load_gnn_model
from src.regulatory_ner import RegulatoryEntityExtractor


# -------------------------------------------------------------
# Page Configuration & Modern Theme
# -------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Symbiosis Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
    /* Clean, modern sans-serif typography across the dashboard */
    html, body, .stApp, .stMarkdown, p, label, input, select, textarea {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* Base canvas */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1440px;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        font-weight: 700;
        color: #0F2942 !important;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.1rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* Header Box */
    .header-box {
        background: linear-gradient(135deg, #0F2942 0%, #1E3A5F 100%);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(15, 41, 66, 0.12);
        color: #FFFFFF;
    }
    
    .header-box h1 {
        color: #FFFFFF !important;
        margin-top: 0;
    }
    
    .header-tagline {
        color: #CBD5E1;
        font-size: 1.02rem;
        margin-top: 4px;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.06);
        border-color: #CBD5E1;
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F2942;
        line-height: 1.2;
    }
    .metric-subtitle {
        font-size: 0.82rem;
        color: #059669;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Badges & Status */
    .badge-success {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #A7F3D0;
        letter-spacing: 0.03em;
    }
    .badge-warning {
        background-color: #FFFBEB;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #FDE68A;
        letter-spacing: 0.03em;
    }
    .badge-danger {
        background-color: #FEF2F2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #FECACA;
        letter-spacing: 0.03em;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #334155;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #E2E8F0;
    }

    /* Section Cards */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
    }
    
    .filter-panel {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    /* Entity Badges */
    .entity-substance { background-color: #EFF6FF; color: #1E40AF; padding: 4px 10px; border-radius: 6px; font-weight: 600; border: 1px solid #BFDBFE; font-size: 0.9rem; }
    .entity-threshold { background-color: #FEF3C7; color: #92400E; padding: 4px 10px; border-radius: 6px; font-weight: 700; border: 1px solid #FDE68A; font-size: 0.9rem; }
    .entity-unit { background-color: #F3E8FF; color: #6B21A8; padding: 4px 8px; border-radius: 6px; font-weight: 600; border: 1px solid #E9D5FF; font-size: 0.85rem; }
    .entity-obligation { background-color: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 6px; font-weight: 700; border: 1px solid #FECACA; font-size: 0.85rem; }

    /* Consistent spacing and restrained interaction states */
    .main .block-container {
        padding-left: clamp(1rem, 4vw, 3.5rem) !important;
        padding-right: clamp(1rem, 4vw, 3.5rem) !important;
    }
    h2, h3, h4, h5 {
        margin-top: 1.25rem !important;
        margin-bottom: 0.45rem !important;
    }
    [data-testid="stCaptionContainer"] {
        color: #64748B;
        line-height: 1.55;
        margin-bottom: 1rem;
    }
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.65rem;
        border-bottom: 1px solid #E2E8F0;
    }
    [data-testid="stTabs"] button[role="tab"] {
        flex: 1 1 0;
        min-width: 0;
        min-height: 3rem;
        padding: 0.6rem 0.45rem;
        color: #475569;
        font-weight: 600;
        line-height: 1.3;
        white-space: normal;
        border-radius: 8px 8px 0 0;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #0F2942;
        background: #EFF6FF;
        border-bottom-color: #2563EB;
    }
    [data-testid="stTabs"] [data-testid="stIconMaterial"],
    [data-testid="stTabs"] .material-symbols-rounded {
        font-family: "Material Symbols Rounded" !important;
        font-weight: normal;
        font-style: normal;
        font-size: 1.5rem;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        font-feature-settings: "liga";
        -webkit-font-feature-settings: "liga";
        -webkit-font-smoothing: antialiased;
    }
    [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
        border-radius: 8px;
        font-weight: 600;
        transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
    }
    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-1px);
    }
    [data-testid="stSelectbox"], [data-testid="stTextInput"],
    [data-testid="stNumberInput"], [data-testid="stSlider"] {
        margin-bottom: 0.35rem;
    }
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
        border-radius: 10px;
    }
    .metric-card, .section-card, .filter-panel {
        border-radius: 10px;
    }
    .header-box {
        padding: clamp(1.4rem, 3vw, 2rem) clamp(1.4rem, 4vw, 2.5rem);
        margin-bottom: 1.5rem;
    }
    .header-tagline {
        max-width: 760px;
        line-height: 1.55;
    }
    .status-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.45rem 0;
        color: #334155;
        font-size: 0.9rem;
    }
    .status-dot {
        width: 0.5rem;
        height: 0.5rem;
        flex: 0 0 0.5rem;
        border-radius: 50%;
    }
    .status-dot-ready { background: #059669; }
    .status-dot-missing { background: #DC2626; }
    .status-label {
        margin-left: auto;
        font-weight: 600;
    }
    @media (max-width: 700px) {
        .main .block-container { padding-top: 1rem !important; }
        [data-testid="stTabs"] [role="tablist"] { gap: 0.25rem; }
        [data-testid="stTabs"] button[role="tab"] {
            flex: 0 0 auto;
            min-height: 2.7rem;
            padding: 0.5rem 0.65rem;
            font-size: 0.85rem;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------
# Data & Model Caching
# -------------------------------------------------------------
@st.cache_data
def load_all_data():
    csv_path = BASE_DIR / "data" / "processed" / "pair_features.csv"
    nodes_path = BASE_DIR / "data" / "processed" / "nodes.csv"
    edges_path = BASE_DIR / "data" / "processed" / "edges.csv"
    
    pairs_df = load_dataset(str(csv_path)) if csv_path.exists() else pd.DataFrame()
    nodes_df = pd.read_csv(nodes_path) if nodes_path.exists() else pd.DataFrame()
    edges_df = pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame()
    sim_data = load_sim_data(str(BASE_DIR / "data"))
    return pairs_df, nodes_df, edges_df, sim_data

@st.cache_resource
def load_models():
    rec = joblib.load(BASE_DIR / "models" / "recommender.joblib") if (BASE_DIR / "models" / "recommender.joblib").exists() else None
    gnn = load_gnn_model(str(BASE_DIR / "models" / "gnn_link_predictor.joblib")) if (BASE_DIR / "models" / "gnn_link_predictor.joblib").exists() else None
    rl = joblib.load(BASE_DIR / "models" / "rl_nsga2_pareto.joblib") if (BASE_DIR / "models" / "rl_nsga2_pareto.joblib").exists() else None
    ner = joblib.load(BASE_DIR / "models" / "regulatory_ner.joblib") if (BASE_DIR / "models" / "regulatory_ner.joblib").exists() else None
    return rec, gnn, rl, ner


@st.cache_data
def rank_pareto_candidates(candidates):
    """Assign local Pareto fronts within the currently selected candidate set."""
    ranked = candidates.copy()
    objective_columns = ["margin_per_tonne", "co2_saving_per_tonne", "distance_km"]
    values = ranked[objective_columns].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1)
    ranked = ranked.loc[valid].copy()
    values = values.loc[valid]

    if ranked.empty:
        ranked["pareto_rank"] = pd.Series(dtype="int64")
        return ranked

    margin = values["margin_per_tonne"].to_numpy()
    carbon = values["co2_saving_per_tonne"].to_numpy()
    distance = values["distance_km"].to_numpy()
    objectives = np.column_stack([margin, carbon, distance.max() - distance])

    dominates = (
        (objectives[:, None, :] >= objectives[None, :, :]).all(axis=2)
        & (objectives[:, None, :] > objectives[None, :, :]).any(axis=2)
    )
    domination_counts = dominates.sum(axis=0)
    current_front = np.flatnonzero(domination_counts == 0)
    ranks = np.zeros(len(ranked), dtype=np.int64)
    rank = 1

    while current_front.size:
        ranks[current_front] = rank
        domination_counts -= dominates[current_front].sum(axis=0)
        current_front = np.flatnonzero((domination_counts == 0) & (ranks == 0))
        rank += 1

    ranked["pareto_rank"] = ranks
    return ranked


pairs_df, nodes_df, edges_df, sim_data = load_all_data()
rec_model, gnn_model, rl_data, ner_data = load_models()

# Entity dictionary mapping for clean UI display
industry_names = {}
industry_sectors = {}
material_names = {}

if len(nodes_df) > 0:
    for _, r in nodes_df[nodes_df.node_type == "industry"].iterrows():
        raw_name = str(r.get("name", r.node_id))
        parts = raw_name.split("_")
        if len(parts) >= 2:
            sector_clean = parts[0].replace("-", " ").title()
            city_clean = parts[1].replace("-", " ").title()
            label = f"{city_clean} {sector_clean} ({r.node_id})"
        else:
            label = f"{raw_name.title()} ({r.node_id})"
        industry_names[r.node_id] = label
        industry_sectors[r.node_id] = str(r.get("sector", "unknown")).replace("-", " ").title()

    for _, r in nodes_df[nodes_df.node_type == "material"].iterrows():
        mat_clean = str(r.get("name", r.node_id)).replace("-", " ").title()
        material_names[r.node_id] = f"{mat_clean} ({r.node_id})"


# -------------------------------------------------------------
# Sidebar Status & Information
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <div>
            <div style="font-size: 1.15rem; font-weight: 750; color: #0F2942; line-height: 1.35;">Industrial Symbiosis</div>
            <div style="font-size: 0.82rem; color: #64748B;">Exchange planning and analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Tools for assessing by-product exchanges, commercial value, and regulatory constraints.")
    
    st.markdown("---")
    st.markdown("#### **System Architecture**")
    
    def status_indicator(is_active, name):
        state = "Ready" if is_active else "Missing"
        status_class = "ready" if is_active else "missing"
        return (
            f'<div class="status-row">'
            f'<span class="status-dot status-dot-{status_class}"></span>'
            f'<span>{name}</span>'
            f'<span class="status-label" style="color: {"#059669" if is_active else "#DC2626"};">{state}</span>'
            f'</div>'
        )
        
    st.markdown(status_indicator(rec_model is not None, "Match Recommender"), unsafe_allow_html=True)
    st.markdown(status_indicator(len(pairs_df) > 0, "Hybrid Dataset"), unsafe_allow_html=True)
    st.markdown(status_indicator(gnn_model is not None, "Facility link model"), unsafe_allow_html=True)
    st.markdown(status_indicator(rl_data is not None, "Trade-off ranking data"), unsafe_allow_html=True)
    st.markdown(status_indicator(ner_data is not None, "Regulatory rules"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### **Platform Scope**")
    st.markdown(f"""
    - **Facilities Indexed:** {len(industry_names)} plants
    - **By-Products Cataloged:** {len(material_names)} materials
    - **Evaluated Pairs:** {len(pairs_df):,} candidate trades
    - **Regulatory Directives:** {len(ner_data.get('extracted_rules', [])) if ner_data else 0} rules
    """)
    
    st.markdown("---")
    st.info("Start with exchange matches, then review their economics, trade-offs, and facility links.")


# -------------------------------------------------------------
# Top Header Banner
# -------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; color: #FFFFFF;">Industrial Symbiosis Decision Platform</h1>
            <div class="header-tagline">Assess by-product exchange opportunities, compare commercial and environmental outcomes, and review regulatory constraints.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Navigation Tabs
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Exchange Matches",
    "Profitability",
    "Match Rationale",
    "Facility Network",
    "Trade-off Ranking",
    "Regulatory Rules"
])


# =============================================================
# TAB 1: MATCH RECOMMENDER
# =============================================================
with tab1:
    st.markdown("### Industrial By-Product Exchange Matches")
    st.caption("Matches waste producers with optimal consumer facilities based on material properties, logistic distance, and economic surplus.")
    
    if rec_model is not None and len(pairs_df) > 0:
        producers = sorted(pairs_df.producer_id.unique().tolist())
        
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1.6, 1.4, 1.2, 0.8])
        
        with col_ctrl1:
            selected_producer = st.selectbox(
                "Select producer facility",
                producers,
                index=0,
                format_func=lambda x: industry_names.get(x, x),
                help="Select an industrial plant producing by-products."
            )
            
        with col_ctrl2:
            avail_materials = sorted(pairs_df[pairs_df.producer_id == selected_producer].material_id.unique().tolist())
            mat_options = ["ALL"] + avail_materials
            material_filter = st.selectbox(
                "Filter by by-product",
                mat_options,
                format_func=lambda x: "All By-Products" if x == "ALL" else material_names.get(x, x),
                help="Filter matches to a specific material or view all available exchanges."
            )
            
        with col_ctrl3:
            all_sectors = ["ALL"] + sorted(pairs_df.consumer_sector.dropna().unique().tolist())
            sector_filter = st.selectbox(
                "Consumer sector",
                all_sectors,
                format_func=lambda x: "All Sectors" if x == "ALL" else str(x).replace("-", " ").title()
            )
            
        with col_ctrl4:
            top_k = st.slider("Max Results", min_value=3, max_value=15, value=6)
        st.markdown('</div>', unsafe_allow_html=True)
            
        mat_arg = None if material_filter == "ALL" else material_filter
        recs = recommend(selected_producer, pairs_df, rec_model, material_id=mat_arg, top_k=top_k * 2)
        
        if sector_filter != "ALL":
            recs = recs[recs.consumer_sector == sector_filter]
        recs = recs.head(top_k)
        
        if len(recs) > 0:
            best_match = recs.iloc[0]
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Highest Confidence</div>
                    <div class="metric-value">{best_match.score * 100:.1f}%</div>
                    <div class="metric-subtitle">Partner: {industry_names.get(best_match.consumer_id, best_match.consumer_id).split(' (')[0]}</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Top Unit Margin</div>
                    <div class="metric-value">₹{recs.margin_per_tonne.max():,.0f}<span style="font-size: 0.85rem; font-weight: normal; color: #64748B;"> /t</span></div>
                    <div class="metric-subtitle">Average: ₹{recs.margin_per_tonne.mean():,.0f}/t</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Peak Carbon Abatement</div>
                    <div class="metric-value">{recs.co2_saving_per_tonne.max():,.0f}<span style="font-size: 0.85rem; font-weight: normal; color: #64748B;"> kg/t</span></div>
                    <div class="metric-subtitle">Avoided vs Virgin Material</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Transit Distance</div>
                    <div class="metric-value">{recs.distance_km.mean():.1f}<span style="font-size: 0.85rem; font-weight: normal; color: #64748B;"> km</span></div>
                    <div class="metric-subtitle">Nearest: {recs.distance_km.min():.1f} km</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            c_table, c_chart = st.columns([1.35, 1.15])
            
            with c_table:
                st.markdown("##### Recommended Exchange Matches")
                styled_recs = recs.copy()
                styled_recs["consumer_name"] = styled_recs["consumer_id"].map(lambda x: industry_names.get(x, x))
                styled_recs["material_name"] = styled_recs["material_id"].map(lambda x: material_names.get(x, x))
                styled_recs["score_pct"] = (styled_recs["score"] * 100).round(1).astype(str) + "%"
                styled_recs["margin_str"] = "₹" + styled_recs["margin_per_tonne"].round(0).astype(int).astype(str)
                styled_recs["dist_str"] = styled_recs["distance_km"].round(1).astype(str) + " km"
                styled_recs["co2_str"] = styled_recs["co2_saving_per_tonne"].round(0).astype(int).astype(str) + " kg/t"
                
                st.dataframe(
                    styled_recs[["consumer_name", "material_name", "consumer_sector", "application", "score_pct", "margin_str", "dist_str", "co2_str"]],
                    column_config={
                        "consumer_name": "Consumer Facility",
                        "material_name": "By-Product Material",
                        "consumer_sector": "Sector",
                        "application": "Application",
                        "score_pct": "Score",
                        "margin_str": "Unit Margin",
                        "dist_str": "Distance",
                        "co2_str": "CO2 Avoided"
                    },
                    width="stretch",
                    hide_index=True
                )

            with c_chart:
                st.markdown("##### Trade Viability vs Unit Margin")
                plot_df = recs.copy()
                plot_df["Consumer Name"] = plot_df["consumer_id"].map(lambda x: industry_names.get(x, x))
                plot_df["Material Name"] = plot_df["material_id"].map(lambda x: material_names.get(x, x))
                plot_df["Confidence Score"] = (plot_df["score"] * 100).round(1).astype(str) + "%"
                
                fig_rec = px.scatter(
                    plot_df,
                    x="distance_km",
                    y="margin_per_tonne",
                    size="score",
                    color="consumer_sector",
                    hover_name="Consumer Name",
                    hover_data={"distance_km": True, "margin_per_tonne": ":,.0f", "Confidence Score": True, "Material Name": True, "application": True},
                    labels={"distance_km": "Transit Distance (km)", "margin_per_tonne": "Unit Margin (₹/tonne)", "consumer_sector": "Sector"},
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_rec.update_layout(
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig_rec.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
                fig_rec.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
                st.plotly_chart(fig_rec, width="stretch")
        else:
            st.info("No exchange matches meet these filters. Try viewing all by-products or all sectors.")
    else:
        st.warning("The match model or data is unavailable. Check that the required model and data files are present.")


# =============================================================
# TAB 2: PROFITABILITY SIMULATOR
# =============================================================
with tab2:
    st.markdown("### Circular Economy Profitability & Scenario Simulator")
    st.caption("Simulate deal profitability, contract pricing, freight costs, and capital payback with sensitivity analysis and Monte Carlo risk assessment.")
    
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    sim_mode = st.radio(
        "Scenario source",
        ["Preset from real match", "Custom scenario"],
        horizontal=True,
        help="Load a candidate pair from the dataset or configure custom parameters."
    )
    if sim_mode == "Preset from real match" and pairs_df.empty:
        st.warning("No candidate exchanges are available. Add the processed pair data or choose a custom scenario.")
        def_v, def_p, def_vp, def_dist, def_fr, def_pre, def_co2 = 15000.0, 1400.0, 3000.0, 160.0, 4.2, 250.0, 850.0
    elif sim_mode == "Preset from real match":
        candidate_pairs = pairs_df.copy()
        producer_ids = sorted(candidate_pairs.producer_id.dropna().unique().tolist())
        producer_col, receiver_col, material_col = st.columns(3)
        with producer_col:
            selected_producer_id = st.selectbox(
                "Producer facility",
                producer_ids,
                key="sim_producer",
                format_func=lambda x: industry_names.get(x, x),
            )

        producer_pairs = candidate_pairs[candidate_pairs.producer_id == selected_producer_id]
        consumer_ids = sorted(producer_pairs.consumer_id.dropna().unique().tolist())
        with receiver_col:
            selected_consumer_id = st.selectbox(
                "Receiving facility",
                consumer_ids,
                key="sim_consumer",
                format_func=lambda x: industry_names.get(x, x),
            )

        receiver_pairs = producer_pairs[producer_pairs.consumer_id == selected_consumer_id]
        material_ids = sorted(receiver_pairs.material_id.dropna().unique().tolist())
        with material_col:
            selected_material_id = st.selectbox(
                "By-product",
                material_ids,
                key="sim_material",
                format_func=lambda x: material_names.get(x, x),
            )

        preset_row = receiver_pairs[receiver_pairs.material_id == selected_material_id].iloc[0]
        st.caption(
            f"Supplier: {industry_names.get(preset_row.producer_id, preset_row.producer_id)}  ·  "
            f"Receiver: {industry_names.get(preset_row.consumer_id, preset_row.consumer_id)}  ·  "
            f"By-product: {material_names.get(preset_row.material_id, preset_row.material_id)}"
        )
        st.caption(f"{len(candidate_pairs):,} candidate exchanges are available across all producer facilities.")
        def_v = float(min(preset_row.avg_monthly_tonnes * 12, preset_row.required_tonnes) * 0.5)
        def_p = float(preset_row.market_price)
        def_vp = float(preset_row.virgin_price)
        def_dist = float(preset_row.distance_km)
        def_fr = float(preset_row.transport_cost_per_tonne / max(preset_row.distance_km, 1.0))
        def_pre = float(preset_row.pretreatment_cost_per_tonne)
        def_co2 = float(preset_row.co2_saving_per_tonne)
    else:
        def_v, def_p, def_vp, def_dist, def_fr, def_pre, def_co2 = 15000.0, 1400.0, 3000.0, 160.0, 4.2, 250.0, 850.0
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("##### Material & Contract Pricing")
        v_traded = st.number_input("Annual Traded Volume (tpy)", min_value=100.0, max_value=1000000.0, value=float(def_v), step=500.0)
        p_sale = st.number_input("Negotiated By-Product Price (₹/t)", min_value=0.0, max_value=100000.0, value=float(def_p), step=50.0)
        p_virgin = st.number_input("Virgin Material Benchmark (₹/t)", min_value=100.0, max_value=100000.0, value=float(def_vp), step=100.0)
        
    with col_s2:
        st.markdown("##### Logistics & Pretreatment")
        d_km = st.number_input("Transit Distance (km)", min_value=1.0, max_value=5000.0, value=float(def_dist), step=10.0)
        f_rate = st.number_input("Freight Rate (₹/tonne-km)", min_value=0.5, max_value=50.0, value=float(def_fr), step=0.2)
        c_pre = st.number_input("Consumer Pretreatment Cost (₹/t)", min_value=0.0, max_value=20000.0, value=float(def_pre), step=25.0)
        
    with col_s3:
        st.markdown("##### Environmental & Capital")
        co2_gross = st.number_input("Gross CO2 Avoided (kg CO2e/t)", min_value=0.0, max_value=10000.0, value=float(def_co2), step=50.0)
        c_capex = st.number_input("Consumer Setup CAPEX (₹/tpy)", min_value=0.0, max_value=20000.0, value=120.0, step=10.0)
        c_carbon_price = st.number_input("Carbon Credit Benchmark (₹/tCO2e)", min_value=0.0, max_value=10000.0, value=500.0, step=50.0)
    st.markdown('</div>', unsafe_allow_html=True)

    sim_scenario = Scenario(
        producer_id="PROD_FACILITY", consumer_id="CONS_FACILITY", material_id="BY_PRODUCT",
        volume_tpy=v_traded, price=p_sale, virgin_price=p_virgin,
        distance_km=d_km, freight_rate=f_rate, pretreatment_cost=c_pre,
        co2_saving=co2_gross, capex_per_tpy=c_capex, carbon_price=c_carbon_price
    )
    sim_eval = evaluate(sim_scenario)
    
    st.write("")
    res1, res2, res3, res4, res5 = st.columns(5)
    with res1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Consumer Savings</div>
            <div class="metric-value">₹{sim_eval['annual_consumer_saving']:,.0f}</div>
            <div class="metric-subtitle">₹{sim_eval['consumer_saving_per_t']:,.1f}/tonne</div>
        </div>
        """, unsafe_allow_html=True)
    with res2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Producer Gain</div>
            <div class="metric-value">₹{sim_eval['annual_producer_gain']:,.0f}</div>
            <div class="metric-subtitle">₹{sim_eval['producer_gain_per_t']:,.1f}/tonne</div>
        </div>
        """, unsafe_allow_html=True)
    with res3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Net Joint Surplus</div>
            <div class="metric-value">₹{(sim_eval['annual_consumer_saving'] + sim_eval['annual_producer_gain']):,.0f}</div>
            <div class="metric-subtitle">Total Value Generated</div>
        </div>
        """, unsafe_allow_html=True)
    with res4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Net CO2 Saved</div>
            <div class="metric-value">{sim_eval['annual_co2_saved_t']:,.1f}<span style="font-size: 0.85rem; font-weight: normal;"> tCO2e</span></div>
            <div class="metric-subtitle">After freight deduction</div>
        </div>
        """, unsafe_allow_html=True)
    with res5:
        verdict = '<span class="badge-success">VIABLE</span>' if sim_eval['viable'] else '<span class="badge-danger">NON-VIABLE</span>'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Trade Feasibility</div>
            <div class="metric-value" style="font-size: 1.15rem; padding-top: 4px;">{verdict}</div>
            <div class="metric-subtitle">Max Transit: {sim_eval['max_viable_distance_km']:,.0f} km</div>
        </div>
        """, unsafe_allow_html=True)

    # Fair Bargaining Price Calculator
    fair_p = fair_price(sim_scenario, producer_share=0.5)
    st.info(f"**Suggested balanced price:** ₹{fair_p:,.2f}/t splits net surplus evenly between producer and consumer while keeping both parties profitable.")

    st.write("")
    sim_tab1, sim_tab2, sim_tab3 = st.tabs(["Sensitivity", "Price and surplus", "Risk simulation"])
    
    with sim_tab1:
        st.markdown("##### Parameter Sensitivity Tornado (NPV Swing by ±20% Parameter Shift)")
        sens_df = sensitivity(sim_scenario, pct=0.20, metric="joint_npv")
        fig_sens = px.bar(
            sens_df,
            x="swing",
            y="param",
            orientation="h",
            labels={"swing": "NPV Swing (₹)", "param": "Trade Parameter"},
            color="swing",
            color_continuous_scale="Tealgrn"
        )
        fig_sens.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False
        )
        fig_sens.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_sens, width="stretch")
        
    with sim_tab2:
        st.markdown("##### Price Sweep: Surplus Allocation across By-Product Pricing Range")
        p_range = np.linspace(max(0, p_sale * 0.3), p_virgin * 1.05, 30)
        sweep_df = sweep(sim_scenario, "price", p_range)
        
        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(x=sweep_df.price, y=sweep_df.annual_consumer_saving, mode="lines+markers", name="Consumer Savings (₹)", line=dict(color="#2B6CB0", width=2.5)))
        fig_sw.add_trace(go.Scatter(x=sweep_df.price, y=sweep_df.annual_producer_gain, mode="lines+markers", name="Producer Gain (₹)", line=dict(color="#059669", width=2.5)))
        fig_sw.add_vline(x=p_sale, line_dash="dash", line_color="#E53E3E", annotation_text="Selected Price", annotation_position="top left")
        fig_sw.add_vline(x=fair_p, line_dash="dot", line_color="#805AD5", annotation_text="Nash Fair Price", annotation_position="bottom right")
        
        fig_sw.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_sw.update_xaxes(title="Waste Sale Price (₹/t)", showgrid=True, gridcolor="#F1F5F9")
        fig_sw.update_yaxes(title="Annual Economic Value (₹)", showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_sw, width="stretch")
        
    with sim_tab3:
        st.markdown("##### Probabilistic Monte Carlo Simulation (1,000 Iterations with Market Volatility)")
        mc_summary, mc_df = monte_carlo(sim_scenario, n=1000)
        
        mc_c1, mc_c2, mc_c3 = st.columns(3)
        with mc_c1:
            st.metric("Probability of Viability", f"{mc_summary['p_viable'] * 100:.1f}%")
        with mc_c2:
            st.metric("Expected Joint 5-Yr NPV", f"₹{mc_summary['expected_joint_npv']:,.0f}")
        with mc_c3:
            st.metric("5% Value at Risk (VaR)", f"₹{mc_summary['value_at_risk_5pct']:,.0f}")
            
        fig_mc = px.histogram(
            mc_df,
            x="joint_npv",
            nbins=35,
            color="viable",
            color_discrete_map={True: "#059669", False: "#E53E3E"},
            labels={"joint_npv": "5-Year Joint NPV (₹)", "viable": "Deal Viable", "count": "Simulations"},
            title="Distribution of Simulated Joint Net Present Value"
        )
        fig_mc.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=20, r=20, t=40, b=20))
        fig_mc.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_mc, width="stretch")


# =============================================================
# TAB 3: EXPLAINABILITY & TRADE-OFFS
# =============================================================
with tab3:
    st.markdown("### Match Explainability & Factor Attribution")
    st.caption("Inspect the underlying rationale, positive drivers, and risk factors influencing any recommended symbiotic pairing.")
    
    if rec_model is not None and len(pairs_df) > 0:
        explainer = MatchExplainer(str(BASE_DIR / "models" / "recommender.joblib"))
        
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            p_sel = st.selectbox(
                "Select Producer Facility",
                sorted(pairs_df.producer_id.unique()),
                key="exp_p",
                format_func=lambda x: industry_names.get(x, x)
            )
        with exp_col2:
            avail_consumers = sorted(pairs_df[pairs_df.producer_id == p_sel].consumer_id.unique())
            c_sel = st.selectbox(
                "Select Consumer Facility",
                avail_consumers,
                key="exp_c",
                format_func=lambda x: industry_names.get(x, x)
            )
        with exp_col3:
            avail_pairs = pairs_df[(pairs_df.producer_id == p_sel) & (pairs_df.consumer_id == c_sel)]
            if len(avail_pairs) > 0:
                mat_list = sorted(avail_pairs.material_id.unique())
                m_sel = st.selectbox(
                    "Select By-Product",
                    mat_list,
                    format_func=lambda x: material_names.get(x, x)
                )
            else:
                m_sel = None
        st.markdown('</div>', unsafe_allow_html=True)
            
        matched_rows = pairs_df[(pairs_df.producer_id == p_sel) & (pairs_df.consumer_id == c_sel)]
        if m_sel:
            matched_rows = matched_rows[matched_rows.material_id == m_sel]
            
        if len(matched_rows) > 0:
            pair_record = matched_rows.iloc[0].to_dict()
            exp_res = explainer.explain_pair(pair_record)
            
            p_name = industry_names.get(p_sel, p_sel)
            c_name = industry_names.get(c_sel, c_sel)
            m_name = material_names.get(pair_record.get('material_id'), pair_record.get('material_id'))
            
            st.write("")
            c_score, c_detail = st.columns([1, 1.8])
            
            with c_score:
                st.markdown(f"""
                <div class="section-card" style="text-align: center;">
                    <div style="font-size: 0.82rem; color: #64748B; font-weight: 700; letter-spacing: 0.06em;">MATCH VIABILITY CONFIDENCE</div>
                    <div style="font-size: 3.2rem; font-weight: 800; color: #0F2942; margin: 8px 0;">{exp_res['score'] * 100:.1f}%</div>
                    <div>{'<span class="badge-success">HIGH SYNERGY FIT</span>' if exp_res['score'] > 0.7 else '<span class="badge-warning">MODERATE FIT</span>'}</div>
                    <hr style="margin: 18px 0; border: none; border-top: 1px solid #E2E8F0;">
                    <div style="text-align: left; font-size: 0.9rem; color: #334155; line-height: 1.8;">
                        <div>• <b>Material:</b> {m_name}</div>
                        <div>• <b>Consumer Sector:</b> {str(pair_record.get('consumer_sector', '')).replace('-', ' ').title()}</div>
                        <div>• <b>Application:</b> {pair_record.get('application')}</div>
                        <div>• <b>Transit Distance:</b> {pair_record.get('distance_km'):.1f} km</div>
                        <div>• <b>Unit Profit Margin:</b> ₹{pair_record.get('margin_per_tonne'):,.0f}/t</div>
                        <div>• <b>Carbon Abated:</b> {pair_record.get('co2_saving_per_tonne'):,.0f} kg CO2e/t</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_detail:
                st.markdown("##### Key Match Factors")
                
                st.markdown("###### Positive factors")
                if exp_res["strengths"]:
                    for strength in exp_res["strengths"]:
                        st.markdown(f"""
                        <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-left: 4px solid #16A34A; padding: 10px 14px; margin-bottom: 8px; border-radius: 6px; font-size: 0.92rem; color: #166534;">
                            <b>[POSITIVE]</b> {strength}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No outstanding positive drivers identified.")
                    
                st.markdown("###### Risks and constraints")
                if exp_res["cautions"]:
                    for caution in exp_res["cautions"]:
                        st.markdown(f"""
                        <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #D97706; padding: 10px 14px; margin-bottom: 8px; border-radius: 6px; font-size: 0.92rem; color: #92400E;">
                            <b>[CAUTION]</b> {caution}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #94A3B8; padding: 10px 14px; border-radius: 6px; font-size: 0.92rem; color: #475569;">
                        No prohibitive contamination, spec mismatch, or distance frictions detected.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Please select a valid producer and consumer pairing.")
    else:
        st.warning("Recommender model or data not found.")


# =============================================================
# TAB 4: FACILITY NETWORK
# =============================================================
with tab4:
    st.markdown("### Facility and Material Network")
    st.caption("Compare two facilities to see their by-products, material requirements, and recorded or potential exchange relationships.")
    
    if len(nodes_df) > 0 and len(edges_df) > 0:
        facility_ids = sorted(nodes_df.loc[nodes_df.node_type == "industry", "node_id"].tolist())
        if facility_ids:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("##### Compare two facilities")
            facility_col_a, facility_col_b, facility_col_potential = st.columns([1.4, 1.4, 1.1])
            with facility_col_a:
                facility_a = st.selectbox(
                    "Facility A",
                    facility_ids,
                    index=0,
                    key="network_facility_a",
                    format_func=lambda x: industry_names.get(x, x)
                )
            with facility_col_b:
                facility_b = st.selectbox(
                    "Facility B",
                    facility_ids,
                    index=min(1, len(facility_ids) - 1),
                    key="network_facility_b",
                    format_func=lambda x: industry_names.get(x, x)
                )

            if facility_a == facility_b:
                st.info("Choose two different facilities to compare their relationships.")
            else:
                facility_a_name = industry_names.get(facility_a, facility_a)
                facility_b_name = industry_names.get(facility_b, facility_b)

                def related_materials(facility_id, relation):
                    return set(
                        edges_df.loc[
                            (edges_df.src == facility_id) & (edges_df.relation == relation),
                            "dst"
                        ].tolist()
                    )

                a_produces = related_materials(facility_a, "produces")
                a_requires = related_materials(facility_a, "requires")
                b_produces = related_materials(facility_b, "produces")
                b_requires = related_materials(facility_b, "requires")
                materials_in_scope = sorted(a_produces | a_requires | b_produces | b_requires)

                recorded_links = edges_df[
                    (edges_df.relation == "supplies_to")
                    & (
                        ((edges_df.src == facility_a) & (edges_df.dst == facility_b))
                        | ((edges_df.src == facility_b) & (edges_df.dst == facility_a))
                    )
                ]

                def recorded_volume(source_id, target_id):
                    link_rows = recorded_links[
                        (recorded_links.src == source_id) & (recorded_links.dst == target_id)
                    ]
                    if link_rows.empty or "weight" not in link_rows.columns:
                        return 0.0
                    return float(pd.to_numeric(link_rows.weight, errors="coerce").fillna(0).sum())

                link_a_to_b = recorded_volume(facility_a, facility_b)
                link_b_to_a = recorded_volume(facility_b, facility_a)
                a_to_b_materials = a_produces & b_requires
                b_to_a_materials = b_produces & a_requires
                has_pair_data = all(
                    column in pairs_df.columns
                    for column in ("producer_id", "consumer_id", "material_id")
                )

                summary_a, summary_b, summary_c = st.columns(3)
                summary_a.metric("Shared materials: A to B", len(a_to_b_materials))
                summary_b.metric("Shared materials: B to A", len(b_to_a_materials))
                summary_c.metric(
                    "Recorded exchange volume",
                    f"{link_a_to_b + link_b_to_a:,.0f} t",
                    help=f"A to B: {link_a_to_b:,.0f} t; B to A: {link_b_to_a:,.0f} t"
                )

                relation_rows = []
                for material_id in materials_in_scope:
                    directions = []
                    assessments = []
                    for source_id, target_id, is_possible, source_label, target_label in (
                        (facility_a, facility_b, material_id in a_to_b_materials, "A", "B"),
                        (facility_b, facility_a, material_id in b_to_a_materials, "B", "A"),
                    ):
                        if not is_possible:
                            continue
                        directions.append(f"{source_label} to {target_label}")
                        candidate_rows = (
                            pairs_df[
                                (pairs_df.producer_id == source_id)
                                & (pairs_df.consumer_id == target_id)
                                & (pairs_df.material_id == material_id)
                            ]
                            if has_pair_data
                            else pd.DataFrame()
                        )
                        if candidate_rows.empty:
                            assessments.append(f"{source_label} to {target_label}: no match record")
                        elif "feasible" in candidate_rows.columns:
                            is_feasible = (
                                pd.to_numeric(candidate_rows.feasible, errors="coerce")
                                .fillna(0)
                                .max()
                                >= 1
                            )
                            assessments.append(
                                f"{source_label} to {target_label}: "
                                f"{'feasible' if is_feasible else 'review required'}"
                            )
                        else:
                            assessments.append(f"{source_label} to {target_label}: match recorded")

                    relation_rows.append({
                        "Material": material_names.get(material_id, material_id),
                        "A produces": "Yes" if material_id in a_produces else "—",
                        "A requires": "Yes" if material_id in a_requires else "—",
                        "B produces": "Yes" if material_id in b_produces else "—",
                        "B requires": "Yes" if material_id in b_requires else "—",
                        "Potential exchange": ", ".join(directions) if directions else "No shared requirement",
                        "Match assessment": "; ".join(assessments) if assessments else "—",
                    })

                st.caption(
                    f"Facility A: {facility_a_name}  ·  Facility B: {facility_b_name}"
                )
                if relation_rows:
                    st.dataframe(
                        pd.DataFrame(relation_rows),
                        width="stretch",
                        hide_index=True
                    )
                else:
                    st.info("No by-product or material requirement records were found for these facilities.")

                with facility_col_potential:
                    if gnn_model is not None:
                        try:
                            probability = gnn_model.predict_link_probability(facility_a, facility_b)
                        except Exception as exc:
                            st.error(f"Could not estimate link potential: {exc}")
                        else:
                            st.metric("Estimated link potential", f"{probability * 100:.1f}%")
                    else:
                        st.caption("Link potential estimate is unavailable.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        g_ctrl1, g_ctrl2, g_ctrl3 = st.columns([1.5, 1.5, 1.5])
        
        with g_ctrl1:
            all_sectors = ["All Sectors"] + sorted([s for s in nodes_df[nodes_df.node_type == "industry"].sector.dropna().unique()])
            filter_sector = st.selectbox("Filter by Industrial Sector", all_sectors)
            
        with g_ctrl2:
            relation_labels = {
                "All Relations": None,
                "By-products produced": "produces",
                "Materials required": "requires",
                "Recorded supplies": "supplies_to",
            }
            filter_rel_label = st.selectbox("Filter relationship", list(relation_labels))
            filter_rel = relation_labels[filter_rel_label]
            
        with g_ctrl3:
            max_nodes = st.slider("Display Subgraph Density (Nodes)", min_value=15, max_value=min(72, len(nodes_df)), value=35)
        st.markdown('</div>', unsafe_allow_html=True)

        G = nx.DiGraph()
        sub_nodes = nodes_df.copy()
        if filter_sector != "All Sectors":
            sub_nodes = sub_nodes[(sub_nodes.sector == filter_sector) | (sub_nodes.node_type == "material")]
        sub_nodes = sub_nodes.head(max_nodes)
        sub_node_ids = set(sub_nodes.node_id)
        
        for _, r in sub_nodes.iterrows():
            G.add_node(r.node_id, name=r.get("name", r.node_id), sector=r.get("sector", "material"), node_type=r.get("node_type", "industry"))
            
        sub_edges = edges_df[edges_df.src.isin(sub_node_ids) & edges_df.dst.isin(sub_node_ids)].copy()
        if filter_rel is not None:
            sub_edges = sub_edges[sub_edges.relation == filter_rel]
            
        for _, r in sub_edges.iterrows():
            G.add_edge(r.src, r.dst, relation=r.relation)

        pos = nx.spring_layout(G, seed=42, k=0.45, iterations=50)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color="#CBD5E1"),
            hoverinfo="none",
            mode="lines"
        )

        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_texts = []
        node_symbols = []

        sector_color_map = {
            "steel": "#EF4444", "thermal_power": "#F97316", "cement": "#14B8A6",
            "chemicals": "#8B5CF6", "sugar": "#EAB308", "textile": "#3B82F6",
            "paper": "#22C55E", "rice_mill": "#06B6D4", "auto_service": "#64748B",
            "fertilizer": "#FB923C", "construction_materials": "#A855F7",
            "oil_recycling": "#475569", "plastic_recycling": "#10B981", "material": "#1E3A5F"
        }

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            data = G.nodes[node]
            ntype = data.get("node_type", "industry")
            sector = data.get("sector", "material")
            
            if ntype == "material":
                node_colors.append("#1E3A5F")
                node_sizes.append(18)
                node_symbols.append("diamond")
                node_texts.append(f"Material: {material_names.get(node, node)}")
            else:
                c = sector_color_map.get(sector, "#64748B")
                node_colors.append(c)
                node_sizes.append(14)
                node_symbols.append("circle")
                node_texts.append(f"Facility: {industry_names.get(node, node)}<br>Sector: {sector.title()}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[material_names.get(n, n).split(' (')[0] if G.nodes[n].get("node_type") == "material" else "" for n in G.nodes()],
            textposition="top center",
            textfont=dict(size=10, color="#1E293B"),
            hovertext=node_texts,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                symbol=node_symbols,
                line=dict(width=1.5, color="#FFFFFF")
            )
        )

        fig_kg = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f"Knowledge Graph Subnetwork ({len(G.nodes())} Active Nodes, {len(G.edges())} Directed Links)",
                    font=dict(size=15, color="#0F2942")
                ),
                showlegend=False,
                hovermode="closest",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(b=20, l=20, r=20, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        st.plotly_chart(fig_kg, width="stretch")
        
    else:
        st.warning("Graph data files not found.")


# =============================================================
# TAB 5: TRADE-OFF RANKING
# =============================================================
with tab5:
    st.markdown("### Industry-specific Trade-off Analysis")
    st.caption(
        "Compare candidate exchanges for a selected industry, facility, consumer sector, and material. "
        "Rankings are recalculated for the selected candidate set."
    )

    trade_candidates = pairs_df.copy()
    if trade_candidates.empty and rl_data is not None:
        trade_candidates = rl_data.get("top_pareto_trades", pd.DataFrame()).copy()

    required_trade_columns = {
        "producer_id", "consumer_id", "material_id", "producer_sector",
        "consumer_sector", "margin_per_tonne", "co2_saving_per_tonne",
        "distance_km",
    }
    if not trade_candidates.empty and required_trade_columns.issubset(trade_candidates.columns):
        st.markdown("#### Define the candidate set")
        filter_cols = st.columns(4)

        producer_sectors = ["All industries"] + sorted(
            trade_candidates.producer_sector.dropna().astype(str).unique().tolist()
        )
        with filter_cols[0]:
            selected_producer_sector = st.selectbox(
                "Producer industry",
                producer_sectors,
                key="pareto_producer_sector",
            )

        scoped_candidates = trade_candidates.copy()
        if selected_producer_sector != "All industries":
            scoped_candidates = scoped_candidates[
                scoped_candidates.producer_sector.astype(str) == selected_producer_sector
            ]

        producer_options = ["All facilities"] + sorted(
            scoped_candidates.producer_id.dropna().unique().tolist()
        )
        with filter_cols[1]:
            selected_producer = st.selectbox(
                "Producer facility",
                producer_options,
                key="pareto_producer",
                format_func=lambda x: "All facilities" if x == "All facilities" else industry_names.get(x, x),
            )
        if selected_producer != "All facilities":
            scoped_candidates = scoped_candidates[
                scoped_candidates.producer_id == selected_producer
            ]

        consumer_sectors = ["All sectors"] + sorted(
            scoped_candidates.consumer_sector.dropna().astype(str).unique().tolist()
        )
        with filter_cols[2]:
            selected_consumer_sector = st.selectbox(
                "Consumer industry",
                consumer_sectors,
                key="pareto_consumer_sector",
            )
        if selected_consumer_sector != "All sectors":
            scoped_candidates = scoped_candidates[
                scoped_candidates.consumer_sector.astype(str) == selected_consumer_sector
            ]

        material_options = ["All materials"] + sorted(
            scoped_candidates.material_id.dropna().unique().tolist()
        )
        with filter_cols[3]:
            selected_material = st.selectbox(
                "By-product material",
                material_options,
                key="pareto_material",
                format_func=lambda x: (
                    "All materials"
                    if x == "All materials"
                    else material_names.get(x, x)
                ),
            )
        if selected_material != "All materials":
            scoped_candidates = scoped_candidates[
                scoped_candidates.material_id == selected_material
            ]

        if "feasible" in scoped_candidates.columns:
            only_feasible = st.checkbox(
                "Show only exchanges marked feasible",
                value=True,
                help="Feasibility is the dataset's material-specification check; it is not a guarantee of a commercial deal.",
            )
            if only_feasible:
                feasible_values = pd.to_numeric(scoped_candidates.feasible, errors="coerce").fillna(0)
                scoped_candidates = scoped_candidates[feasible_values >= 1]

        st.markdown("#### Set priorities")
        st.caption(
            "Weights express relative importance. Each measure is normalized within the selected candidate set; "
            "all three weights are rescaled to sum to 100%."
        )
        weight_cols = st.columns(3)
        with weight_cols[0]:
            w_profit = st.slider("Unit margin priority", 0.0, 1.0, 0.45, 0.05, key="pareto_profit_weight")
        with weight_cols[1]:
            w_co2 = st.slider("Emissions reduction priority", 0.0, 1.0, 0.40, 0.05, key="pareto_carbon_weight")
        with weight_cols[2]:
            w_dist = st.slider("Shorter distance priority", 0.0, 1.0, 0.15, 0.05, key="pareto_distance_weight")
        rank_filter = st.selectbox(
            "Show Pareto fronts",
            ["All fronts", "Front 1 only", "Fronts 1 to 3"],
            key="pareto_rank_filter",
        )

        st.markdown("#### How to read the analysis")
        with st.expander("What the scores, ranks, and charts mean"):
            st.markdown(
                """
                - **Each point or row is one candidate producer–consumer–material exchange.**
                - **Unit margin** and **emissions reduction** are better when higher; **distance** is better when lower.
                - **Pareto front 1** contains options that are not beaten on all three measures by another option in this selected set. Later fronts are progressively dominated alternatives.
                - **Priority score** combines the three normalized measures using your weights. It orders options within a front; it is not a probability or predicted deal outcome.
                - In the charts, the horizontal and vertical axes show the named measures, point size represents average monthly supply, and color indicates the Pareto front. The 3D view adds transport distance.

                This is a Pareto-ranking analysis, not a reinforcement-learning policy or a forecast. Results use the project’s candidate-pair data and are sensitive to the selected scope and weights.
                """
            )

        if scoped_candidates.empty:
            st.info("No candidate exchanges match this selection. Broaden the filters or include exchanges not marked feasible.")
        else:
            ranked_candidates = rank_pareto_candidates(scoped_candidates)
            objective_values = ranked_candidates[
                ["margin_per_tonne", "co2_saving_per_tonne", "distance_km"]
            ].apply(pd.to_numeric, errors="coerce")
            objective_values["shorter_distance"] = (
                objective_values["distance_km"].max() - objective_values["distance_km"]
            )
            score_columns = ["margin_per_tonne", "co2_saving_per_tonne", "shorter_distance"]
            score_values = objective_values[score_columns].to_numpy(dtype=float)
            score_min = score_values.min(axis=0)
            score_range = score_values.max(axis=0) - score_min
            normalized_values = (score_values - score_min) / np.where(score_range == 0, 1.0, score_range)

            weights = np.array([w_profit, w_co2, w_dist], dtype=float)
            if weights.sum() == 0:
                weights = np.ones(3, dtype=float) / 3
                st.info("All priorities are zero, so equal priority is used for the score.")
            else:
                weights /= weights.sum()
            ranked_candidates["priority_score"] = normalized_values @ weights
            ranked_candidates["Producer Name"] = ranked_candidates["producer_id"].map(
                lambda x: industry_names.get(x, x)
            )
            ranked_candidates["Consumer Name"] = ranked_candidates["consumer_id"].map(
                lambda x: industry_names.get(x, x)
            )
            ranked_candidates["Material Name"] = ranked_candidates["material_id"].map(
                lambda x: material_names.get(x, x)
            )
            ranked_candidates = ranked_candidates.sort_values(
                ["pareto_rank", "priority_score"], ascending=[True, False]
            )

            if rank_filter == "Front 1 only":
                plot_pareto = ranked_candidates[ranked_candidates.pareto_rank == 1].copy()
            elif rank_filter == "Fronts 1 to 3":
                plot_pareto = ranked_candidates[ranked_candidates.pareto_rank <= 3].copy()
            else:
                plot_pareto = ranked_candidates.copy()

            metric_cols = st.columns(3)
            metric_cols[0].metric("Candidates in scope", f"{len(ranked_candidates):,}")
            metric_cols[1].metric(
                "Non-dominated options",
                f"{int((ranked_candidates.pareto_rank == 1).sum()):,}",
            )
            metric_cols[2].metric(
                "Priority mix",
                f"Margin {weights[0]:.0%} · Emissions {weights[1]:.0%} · Distance {weights[2]:.0%}",
            )

            if plot_pareto.empty:
                st.info("No options are available for the selected Pareto fronts.")
            else:
                col_rl1, col_rl2 = st.columns([1.2, 1])
                with col_rl1:
                    st.markdown("##### Margin, emissions, and distance")
                    st.caption("Each point is an exchange; point size indicates average monthly supply.")
                    fig_3d = px.scatter_3d(
                        plot_pareto,
                        x="margin_per_tonne",
                        y="co2_saving_per_tonne",
                        z="distance_km",
                        color="pareto_rank",
                        size="avg_monthly_tonnes" if "avg_monthly_tonnes" in plot_pareto else None,
                        hover_name="Producer Name",
                        hover_data={
                            "Consumer Name": True,
                            "Material Name": True,
                            "margin_per_tonne": ":,.0f",
                            "co2_saving_per_tonne": ":,.0f",
                            "priority_score": ":.3f",
                        },
                        labels={
                            "margin_per_tonne": "Unit margin (₹/t)",
                            "co2_saving_per_tonne": "Emissions reduction (kg CO₂e/t)",
                            "distance_km": "Distance (km)",
                            "pareto_rank": "Pareto front",
                        },
                        color_continuous_scale="Blues_r",
                    )
                    fig_3d.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10),
                        scene=dict(
                            xaxis_title="Unit margin (₹/t)",
                            yaxis_title="Emissions reduction (kg CO₂e/t)",
                            zaxis_title="Distance (km)",
                            bgcolor="#FFFFFF",
                        ),
                    )
                    st.plotly_chart(fig_3d, width="stretch")

                with col_rl2:
                    st.markdown("##### Unit margin vs emissions reduction")
                    st.caption("Use hover details to compare the facilities, material, distance, and priority score.")
                    fig_2d = px.scatter(
                        plot_pareto,
                        x="margin_per_tonne",
                        y="co2_saving_per_tonne",
                        color="pareto_rank",
                        size="avg_monthly_tonnes" if "avg_monthly_tonnes" in plot_pareto else None,
                        hover_name="Producer Name",
                        hover_data={
                            "Consumer Name": True,
                            "Material Name": True,
                            "distance_km": ":.1f",
                            "priority_score": ":.3f",
                        },
                        labels={
                            "margin_per_tonne": "Unit margin (₹/t)",
                            "co2_saving_per_tonne": "Emissions reduction (kg CO₂e/t)",
                            "pareto_rank": "Pareto front",
                        },
                        color_continuous_scale="Blues_r",
                    )
                    fig_2d.update_layout(
                        plot_bgcolor="#FFFFFF",
                        paper_bgcolor="#FFFFFF",
                        margin=dict(l=10, r=10, t=20, b=10),
                    )
                    fig_2d.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
                    fig_2d.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
                    st.plotly_chart(fig_2d, width="stretch")

                st.markdown("##### Ranked exchange options")
                table = plot_pareto.copy()
                table["margin_fmt"] = "₹" + table["margin_per_tonne"].round(0).astype(int).astype(str)
                table["co2_fmt"] = table["co2_saving_per_tonne"].round(0).astype(int).astype(str) + " kg CO₂e/t"
                table["dist_fmt"] = table["distance_km"].round(1).astype(str) + " km"
                table["priority_fmt"] = (table["priority_score"] * 100).round(1).astype(str) + "%"
                st.dataframe(
                    table[
                        [
                            "Producer Name", "Consumer Name", "Material Name", "pareto_rank",
                            "priority_fmt", "margin_fmt", "co2_fmt", "dist_fmt",
                        ]
                    ],
                    column_config={
                        "Producer Name": "Supplier",
                        "Consumer Name": "Receiver",
                        "Material Name": "By-product",
                        "pareto_rank": "Pareto front",
                        "priority_fmt": "Priority score",
                        "margin_fmt": "Unit margin",
                        "co2_fmt": "Emissions reduction",
                        "dist_fmt": "Distance",
                    },
                    width="stretch",
                    hide_index=True,
                )
    else:
        st.warning("Candidate exchange data is unavailable. Check that the processed pair data is present.")


# =============================================================
# TAB 6: REGULATORY RULES
# =============================================================
with tab6:
    st.markdown("### Regulatory Constraint Extraction & Legal Compliance Parser")
    st.caption("Extracts environmental directives, permissible concentration thresholds, and legal storage limits from statutory gazettes.")
    
    if ner_data is not None:
        rules_df = ner_data.get("extracted_rules", pd.DataFrame())
        
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        c_search, c_filter = st.columns([2, 1])
        with c_search:
            search_sub = st.text_input("Search rules by substance or property", "")
        with c_filter:
            filter_ob = st.selectbox("Obligation Directive Type", ["All Obligations"] + list(rules_df.obligation.dropna().unique()))
        st.markdown('</div>', unsafe_allow_html=True)
            
        filt_df = rules_df.copy()
        if search_sub:
            filt_df = filt_df[filt_df.substance.str.contains(search_sub, case=False, na=False) | filt_df.property.str.contains(search_sub, case=False, na=False)]
        if filter_ob != "All Obligations":
            filt_df = filt_df[filt_df.obligation == filter_ob]
            
        st.dataframe(
            filt_df[["rule_id", "substance", "property", "threshold", "unit", "obligation", "raw_text"]],
            column_config={
                "rule_id": "Rule ID",
                "substance": "Regulated Substance",
                "property": "Constrained Property",
                "threshold": "Permissible Threshold",
                "unit": "Unit",
                "obligation": "Directive Type",
                "raw_text": "Statutory Gazette Excerpt"
            },
            width="stretch",
            hide_index=True
        )
        
        st.write("")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Environmental Clause Parser")
        st.caption("Input legal text from an environmental notice or select a sample clause below to extract regulatory entities.")
        
        sample_clauses = {
            "Blast Furnace Slag Directive": "Facilities shall not accept blast furnace slag with heavy metal content above 140 ppm or moisture above 8.5 %.",
            "Spent Sulfuric Acid Regulation": "Plants shall not accept spent acid with heavy metal content above 120.0 ppm.",
            "Fly Ash Utilization Order": "Thermal power plants must supply fly ash with unburnt carbon content not exceeding 4.0 % to cement blenders.",
            "Phosphogypsum Storage Limit": "Storage of phosphogypsum shall not exceed 15000 tonnes without impermeable leachate lining."
        }
        
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        selected_text = "Facilities shall not accept blast furnace slag with heavy metal content above 140 ppm or moisture above 8.5 %."
        
        if col_btn1.button("Slag Directive"):
            selected_text = sample_clauses["Blast Furnace Slag Directive"]
        if col_btn2.button("Spent Acid Rule"):
            selected_text = sample_clauses["Spent Sulfuric Acid Regulation"]
        if col_btn3.button("Fly Ash Order"):
            selected_text = sample_clauses["Fly Ash Utilization Order"]
        if col_btn4.button("Gypsum Limit"):
            selected_text = sample_clauses["Phosphogypsum Storage Limit"]
            
        test_text = st.text_area(
            "Legal or Environmental Directive Text",
            value=selected_text,
            height=80
        )
        
        if st.button("Extract Entities & Directives", type="primary"):
            extractor = RegulatoryEntityExtractor()
            parsed = extractor.parse_text(test_text)
            
            thresh_display = f"{parsed['threshold']}" if parsed['threshold'] is not None else "Not explicitly quantified"
            unit_display = f" {parsed['unit']}" if parsed['unit'] else ""
            
            st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 10px; padding: 18px 22px; margin-top: 14px;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #0F2942; margin-bottom: 12px;">Extracted regulatory details</div>
                <div style="margin-bottom: 8px;">• <b>Identified Substance:</b> <span class="entity-substance">{parsed['substance'].upper()}</span></div>
                <div style="margin-bottom: 8px;">• <b>Constrained Property:</b> <code>{parsed['property']}</code></div>
                <div style="margin-bottom: 8px;">• <b>Permissible Threshold:</b> <span class="entity-threshold">{thresh_display}</span> <span class="entity-unit">{unit_display}</span></div>
                <div>• <b>Regulatory Obligation:</b> <span class="entity-obligation">{parsed['obligation']}</span></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Regulatory rules are unavailable. Check that the required model data is present.")
