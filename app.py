"""Industrial Symbiosis AI Platform.

A formal, high-contrast, human-readable circular economy decision platform featuring:
- AI-Driven Recommendation Engine
- Circular Profitability & Sensitivity Simulator
- Explainable AI & Trade-Off Attribution
- Interactive Knowledge Graph (GNN) with Node-Edge Visualization
- Multi-Objective RL & NSGA-II Pareto Optimization
- Regulatory Constraint Extraction (NER) & Legal Text Parser
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
from src.simulator import Scenario, evaluate, sensitivity, monte_carlo, fair_price, sweep
from src.explain import MatchExplainer
from src.gnn import load_gnn_model, build_knowledge_graph
from src.regulatory_ner import RegulatoryEntityExtractor


# -------------------------------------------------------------
# Page Configuration & Elegant Typography Theme
# -------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Symbiosis AI Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
    /* Classic Times New Roman serif styling */
    html, body, [class*="css"], .stMarkdown, p, div, span, label, input, select, button {
        font-family: 'Times New Roman', Times, Georgia, serif !important;
    }

    /* Remove Streamlit default red highlight bar & top decoration */
    [data-baseweb="tab-highlight"],
    [data-testid="stDecoration"],
    div[data-testid="stToolbar"] {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        height: 0px !important;
    }

    /* Light, calm, spacious palette */
    .stApp {
        background-color: #FAFAFA;
        color: #1A202C;
    }

    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Times New Roman', Times, Georgia, serif !important;
        font-weight: 600;
        color: #1A365D !important;
        letter-spacing: -0.01em;
    }
    
    h1 {
        font-size: 2.3rem !important;
        margin-bottom: 0.3rem !important;
    }

    /* Header Box */
    .header-box {
        background: linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%);
        border: 1.5px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px 32px;
        margin-bottom: 28px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }
    
    .header-tagline {
        color: #4A5568;
        font-size: 1.05rem;
        margin-top: 4px;
        font-style: italic;
    }

    /* Metric Cards - Rounded & Spacious */
    .metric-card {
        background-color: #FFFFFF;
        border: 1.5px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1A365D;
    }
    .metric-subtitle {
        font-size: 0.82rem;
        color: #2E7D32;
        margin-top: 6px;
    }

    /* Badges & Status */
    .badge-success {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        border: 1.5px solid #C8E6C9;
        text-transform: uppercase;
    }
    .badge-warning {
        background-color: #FFF3E0;
        color: #B78103;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        border: 1.5px solid #FFE0B2;
        text-transform: uppercase;
    }
    .badge-danger {
        background-color: #FFEBEE;
        color: #B71C1C;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        border: 1.5px solid #FFCDD2;
        text-transform: uppercase;
    }
    
    /* Entity Tags */
    .entity-substance { background-color: #E3F2FD; color: #0D47A1; padding: 4px 10px; border-radius: 8px; font-weight: 600; border: 1.5px solid #BBDEFB; }
    .entity-threshold { background-color: #FFF8E1; color: #FF8F00; padding: 4px 10px; border-radius: 8px; font-weight: 600; border: 1.5px solid #FFE082; }
    .entity-unit { background-color: #EDE7F6; color: #4A148C; padding: 4px 10px; border-radius: 8px; font-weight: 600; border: 1.5px solid #D1C4E9; }
    .entity-obligation { background-color: #FFEBEE; color: #B71C1C; padding: 4px 10px; border-radius: 8px; font-weight: 600; border: 1.5px solid #FFCDD2; }

    /* Force dark text universally on all general elements */
    body, .stApp, p, span, label, div, input, textarea, select,
    .stMarkdown, .stMarkdown p, [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p,
    div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label, div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    
    [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        margin-bottom: 3px !important;
    }

    /* Input Fields & Dropdowns - Sleek, Compact & Standard */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 6px !important;
        min-height: 34px !important;
        height: 34px !important;
        padding: 0px 8px !important;
        font-size: 0.88rem !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #1A365D !important;
    }
    div[data-baseweb="select"] * {
        font-size: 0.88rem !important;
    }
    
    /* Strictly neutralize any selectbox internal buttons / arrows */
    div[data-baseweb="select"] button,
    div[data-baseweb="select"] [role="button"],
    div[data-baseweb="select"] svg {
        border: none !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: unset !important;
        width: auto !important;
        height: auto !important;
        transform: none !important;
    }
    
    div[data-baseweb="input"] > div, div[data-baseweb="base-input"], input, textarea {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 6px !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        min-height: 34px !important;
        height: 34px !important;
        padding: 2px 8px !important;
    }
    textarea {
        height: auto !important;
        min-height: 80px !important;
    }
    div[data-baseweb="input"] > div:focus-within, textarea:focus {
        border-color: #1A365D !important;
        box-shadow: 0 0 0 2px rgba(26, 54, 93, 0.15) !important;
    }

    /* --------------------------------------------------------- */
    /* MODULE SELECTION TABS - STRICTLY SCOPED TO TAB-LIST       */
    /* --------------------------------------------------------- */
    [data-testid="stTabs"] [data-baseweb="tab-list"],
    div[data-testid="stTabs"] > div:first-child {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 14px !important;
        border-bottom: none !important;
        margin: 10px 0 28px 0 !important;
        padding: 4px 0 !important;
    }

    /* UNSELECTED TABS: ONLY targets buttons inside tab-list */
    [data-testid="stTabs"] [data-baseweb="tab-list"] button,
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border: 2px solid #1A365D !important;
        border-radius: 26px !important;
        padding: 9px 24px !important;
        margin: 0 !important;
        height: auto !important;
        min-height: 44px !important;
        box-shadow: 0 2px 6px rgba(26, 54, 93, 0.08) !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        outline: none !important;
    }

    /* Text inside unselected tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] button p,
    [data-testid="stTabs"] [data-baseweb="tab-list"] button span,
    [data-testid="stTabs"] [data-baseweb="tab-list"] button div,
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="false"] p,
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="false"] span {
        color: #1A365D !important;
        -webkit-text-fill-color: #1A365D !important;
        font-family: 'Times New Roman', Times, Georgia, serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* Hover State on Unselected */
    [data-testid="stTabs"] [data-baseweb="tab-list"] button:hover {
        background-color: #F0F4F8 !important;
        border-color: #1A365D !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 10px rgba(26, 54, 93, 0.16) !important;
    }

    /* SELECTED ACTIVE TAB: Solid Dark Blue Fill + White Text */
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"],
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #1A365D !important;
        border: 2px solid #1A365D !important;
        border-radius: 26px !important;
        box-shadow: 0 4px 14px rgba(26, 54, 93, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Text inside selected active tab */
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"] p,
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"] span,
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"] div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-family: 'Times New Roman', Times, Georgia, serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* Action Buttons Inside Modules */
    div.stButton > button, div.stButton > button:first-child {
        background-color: #1A365D !important;
        border: 2px solid #1A365D !important;
        border-radius: 24px !important;
        padding: 10px 28px !important;
        box-shadow: 0 3px 8px rgba(26, 54, 93, 0.25) !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button, div.stButton > button p, div.stButton > button span, div.stButton > button div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-family: 'Times New Roman', Times, Georgia, serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    div.stButton > button:hover {
        background-color: #2B4C7E !important;
        border-color: #1A365D !important;
        box-shadow: 0 6px 16px rgba(26, 54, 93, 0.35) !important;
        transform: translateY(-2px) !important;
    }
    div.stButton > button:active {
        background-color: #0F243E !important;
        transform: translateY(1px) !important;
    }

    /* Rounded Section Container */
    .section-card {
        background: #FFFFFF;
        border: 1.5px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
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
    return pairs_df, nodes_df, edges_df

@st.cache_resource
def load_models():
    rec = joblib.load(BASE_DIR / "models" / "recommender.joblib") if (BASE_DIR / "models" / "recommender.joblib").exists() else None
    gnn = load_gnn_model(str(BASE_DIR / "models" / "gnn_link_predictor.joblib")) if (BASE_DIR / "models" / "gnn_link_predictor.joblib").exists() else None
    rl = joblib.load(BASE_DIR / "models" / "rl_nsga2_pareto.joblib") if (BASE_DIR / "models" / "rl_nsga2_pareto.joblib").exists() else None
    ner = joblib.load(BASE_DIR / "models" / "regulatory_ner.joblib") if (BASE_DIR / "models" / "regulatory_ner.joblib").exists() else None
    return rec, gnn, rl, ner

pairs_df, nodes_df, edges_df = load_all_data()
rec_model, gnn_model, rl_data, ner_data = load_models()

# Human-readable entity dictionaries
industry_names = {}
material_names = {}
if len(nodes_df) > 0:
    for _, r in nodes_df[nodes_df.node_type == "industry"].iterrows():
        raw_name = str(r.get("name", r.node_id))
        parts = raw_name.split("_")
        if len(parts) >= 2:
            sector_clean = parts[0].replace("-", " ").title()
            city_clean = parts[1].replace("-", " ").title()
            label = f"{city_clean} {sector_clean} Plant ({r.node_id})"
        else:
            label = f"{raw_name.title()} ({r.node_id})"
        industry_names[r.node_id] = label

    for _, r in nodes_df[nodes_df.node_type == "material"].iterrows():
        mat_clean = str(r.get("name", r.node_id)).title()
        material_names[r.node_id] = f"{mat_clean} ({r.node_id})"


# -------------------------------------------------------------
# Top Header Banner
# -------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <h1>Industrial Symbiosis AI Decision Platform</h1>
            <div class="header-tagline">Autonomous by-product valorization, multi-objective trade optimization and regulatory compliance verification</div>
        </div>
        <div>
            <span class="badge-success">Operational System</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Navigation Tabs (Rounded Pills)
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Match Recommender",
    "Profitability Simulator",
    "Explainability & Trade-offs",
    "Knowledge Graph (GNN)",
    "Multi-Objective Pareto RL",
    "Regulatory Compliance NER"
])


# =============================================================
# TAB 1: MATCH RECOMMENDER
# =============================================================
with tab1:
    st.markdown("### AI-Driven Industrial Waste Exchange Recommendations")
    st.caption("Matches waste producers with optimal consumer facilities based on material properties, logistic distance, and economic surplus.")
    st.write("")
    
    if rec_model is not None and len(pairs_df) > 0:
        producers = sorted(pairs_df.producer_id.unique().tolist())
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.8, 1.5, 1])
        with col_ctrl1:
            selected_producer = st.selectbox(
                "Select Waste Producer Facility",
                producers,
                index=0,
                format_func=lambda x: industry_names.get(x, x)
            )
        with col_ctrl2:
            avail_materials = sorted(pairs_df[pairs_df.producer_id == selected_producer].material_id.unique().tolist())
            mat_options = ["ALL"] + avail_materials
            material_filter = st.selectbox(
                "Filter by By-Product / Material",
                mat_options,
                format_func=lambda x: "All By-Products" if x == "ALL" else material_names.get(x, x)
            )
        with col_ctrl3:
            top_k = st.slider("Max Matches (k)", min_value=3, max_value=12, value=5)
        st.markdown('</div>', unsafe_allow_html=True)
            
        mat_arg = None if material_filter == "ALL" else material_filter
        recs = recommend(selected_producer, pairs_df, rec_model, material_id=mat_arg, top_k=top_k)
        
        if len(recs) > 0:
            best_match = recs.iloc[0]
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Highest Confidence</div>
                    <div class="metric-value">{best_match.score * 100:.1f}%</div>
                    <div class="metric-subtitle">Partner: {industry_names.get(best_match.consumer_id, best_match.consumer_id)}</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Max Unit Profit Margin</div>
                    <div class="metric-value">INR {recs.margin_per_tonne.max():,.0f}<span style="font-size: 0.85rem; font-weight: normal;"> /t</span></div>
                    <div class="metric-subtitle">Avg: INR {recs.margin_per_tonne.mean():,.0f}/t</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Carbon Abatement</div>
                    <div class="metric-value">{recs.co2_saving_per_tonne.max():,.0f}<span style="font-size: 0.85rem; font-weight: normal;"> kg/t</span></div>
                    <div class="metric-subtitle">Avoided vs Virgin Material</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Transit Distance</div>
                    <div class="metric-value">{recs.distance_km.mean():.1f}<span style="font-size: 0.85rem; font-weight: normal;"> km</span></div>
                    <div class="metric-subtitle">Min: {recs.distance_km.min():.1f} km</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.write("")
            
            c_table, c_chart = st.columns([1.4, 1.1])
            
            with c_table:
                st.markdown("##### Recommended Exchange Matches")
                styled_recs = recs.copy()
                styled_recs["consumer_name"] = styled_recs["consumer_id"].map(lambda x: industry_names.get(x, x))
                styled_recs["material_name"] = styled_recs["material_id"].map(lambda x: material_names.get(x, x))
                styled_recs["score_pct"] = (styled_recs["score"] * 100).round(1).astype(str) + "%"
                styled_recs["margin_str"] = "INR " + styled_recs["margin_per_tonne"].round(0).astype(int).astype(str)
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
                    use_container_width=True,
                    hide_index=True
                )

            with c_chart:
                st.markdown("##### Trade Viability vs Unit Margin")
                plot_df = recs.copy()
                plot_df["Consumer Name"] = plot_df["consumer_id"].map(lambda x: industry_names.get(x, x))
                plot_df["Material Name"] = plot_df["material_id"].map(lambda x: material_names.get(x, x))
                
                fig_rec = px.scatter(
                    plot_df,
                    x="distance_km",
                    y="margin_per_tonne",
                    size="score",
                    color="consumer_sector",
                    hover_name="Consumer Name",
                    hover_data=["Material Name", "application", "co2_saving_per_tonne"],
                    labels={"distance_km": "Transit Distance (km)", "margin_per_tonne": "Unit Margin (INR/tonne)", "consumer_sector": "Sector"},
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_rec.update_layout(
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font_family="Times New Roman",
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig_rec.update_xaxes(showgrid=True, gridcolor="#F0F2F5", zeroline=False)
                fig_rec.update_yaxes(showgrid=True, gridcolor="#F0F2F5", zeroline=False)
                st.plotly_chart(fig_rec, use_container_width=True)
    else:
        st.warning("Recommender model or data not found. Please train models first.")


# =============================================================
# TAB 2: PROFITABILITY SIMULATOR
# =============================================================
with tab2:
    st.markdown("### Circular Economy Profitability & Scenario Simulator")
    st.caption("Simulate deal profitability, contract pricing, freight costs, and capital payback with sensitivity analysis.")
    st.write("")
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("**Material & Pricing**")
        v_traded = st.number_input("Annual Traded Volume (tpy)", min_value=500.0, max_value=500000.0, value=15000.0, step=1000.0)
        p_sale = st.number_input("Negotiated By-Product Price (INR/t)", min_value=0.0, max_value=50000.0, value=1400.0, step=50.0)
        p_virgin = st.number_input("Virgin Material Benchmark (INR/t)", min_value=100.0, max_value=50000.0, value=3000.0, step=100.0)
        
    with col_s2:
        st.markdown("**Logistics & Processing**")
        d_km = st.number_input("Transit Distance (km)", min_value=5.0, max_value=3000.0, value=160.0, step=10.0)
        f_rate = st.number_input("Freight Rate (INR/tonne-km)", min_value=1.0, max_value=20.0, value=4.2, step=0.2)
        c_pre = st.number_input("Pretreatment Cost (INR/t)", min_value=0.0, max_value=10000.0, value=250.0, step=25.0)
        
    with col_s3:
        st.markdown("**Environmental & Capital**")
        co2_gross = st.number_input("Gross CO2 Avoided (kg CO2e/t)", min_value=0.0, max_value=5000.0, value=850.0, step=50.0)
        c_capex = st.number_input("Consumer Setup CAPEX (INR/tpy)", min_value=0.0, max_value=10000.0, value=120.0, step=10.0)
        c_carbon_price = st.number_input("Carbon Credit Benchmark (INR/tCO2e)", min_value=0.0, max_value=5000.0, value=500.0, step=50.0)
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
            <div class="metric-value">INR {sim_eval['annual_consumer_saving']:,.0f}</div>
            <div class="metric-subtitle">INR {sim_eval['consumer_saving_per_t']:,.1f}/tonne</div>
        </div>
        """, unsafe_allow_html=True)
    with res2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Producer Gain</div>
            <div class="metric-value">INR {sim_eval['annual_producer_gain']:,.0f}</div>
            <div class="metric-subtitle">INR {sim_eval['producer_gain_per_t']:,.1f}/tonne</div>
        </div>
        """, unsafe_allow_html=True)
    with res3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Net Joint Surplus</div>
            <div class="metric-value">INR {(sim_eval['annual_consumer_saving'] + sim_eval['annual_producer_gain']):,.0f}</div>
            <div class="metric-subtitle">Annual Total Value</div>
        </div>
        """, unsafe_allow_html=True)
    with res4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Annual Net CO2 Saved</div>
            <div class="metric-value">{sim_eval['annual_co2_saved_t']:,.1f}<span style="font-size: 0.85rem;"> tCO2e</span></div>
            <div class="metric-subtitle">Net transit deduction</div>
        </div>
        """, unsafe_allow_html=True)
    with res5:
        verdict = '<span class="badge-success">VIABLE</span>' if sim_eval['viable'] else '<span class="badge-danger">NON-VIABLE</span>'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Trade Feasibility</div>
            <div class="metric-value" style="font-size: 1.15rem; padding-top: 4px;">{verdict}</div>
            <div class="metric-subtitle">Max Dist: {sim_eval['max_viable_distance_km']:,.0f} km</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    c_mc, c_sens = st.columns([1.2, 1.2])
    
    with c_mc:
        st.markdown("##### Sensitivity Tornado (NPV Swing by +/-20% Parameter Shift)")
        sens_df = sensitivity(sim_scenario, pct=0.20, metric="joint_npv")
        fig_sens = px.bar(
            sens_df,
            x="swing",
            y="param",
            orientation="h",
            labels={"swing": "NPV Swing (INR)", "param": "Parameter"},
            color="swing",
            color_continuous_scale="Tealgrn"
        )
        fig_sens.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font_family="Times New Roman",
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False
        )
        fig_sens.update_xaxes(showgrid=True, gridcolor="#F0F2F5")
        st.plotly_chart(fig_sens, use_container_width=True)
        
    with c_sens:
        st.markdown("##### Price Sweep (Surplus Allocation vs Sale Price)")
        p_range = np.linspace(max(0, p_sale * 0.4), p_virgin * 1.1, 25)
        sweep_df = sweep(sim_scenario, "price", p_range)
        
        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(x=sweep_df.price, y=sweep_df.annual_consumer_saving, mode="lines+markers", name="Consumer Savings (INR)", line=dict(color="#2B6CB0", width=2.5)))
        fig_sw.add_trace(go.Scatter(x=sweep_df.price, y=sweep_df.annual_producer_gain, mode="lines+markers", name="Producer Gain (INR)", line=dict(color="#2F855A", width=2.5)))
        fig_sw.add_vline(x=p_sale, line_dash="dash", line_color="#E53E3E", annotation_text="Selected Price", annotation_position="top left")
        
        fig_sw.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font_family="Times New Roman",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_sw.update_xaxes(title="Waste Sale Price (INR/t)", showgrid=True, gridcolor="#F0F2F5")
        fig_sw.update_yaxes(title="Annual Economic Value (INR)", showgrid=True, gridcolor="#F0F2F5")
        st.plotly_chart(fig_sw, use_container_width=True)


# =============================================================
# TAB 3: EXPLAINABILITY & TRADE-OFFS
# =============================================================
with tab3:
    st.markdown("### Match Explainability & Factor Attribution")
    st.caption("Inspect the underlying rationale, positive drivers, and risk factors influencing any recommended symbiotic pairing.")
    st.write("")
    
    if rec_model is not None and len(pairs_df) > 0:
        explainer = MatchExplainer(str(BASE_DIR / "models" / "recommender.joblib"))
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            p_sel = st.selectbox(
                "Select Producer Facility",
                sorted(pairs_df.producer_id.unique()),
                key="exp_p",
                format_func=lambda x: industry_names.get(x, x)
            )
        with exp_col2:
            avail_c = pairs_df[pairs_df.producer_id == p_sel].consumer_id.unique()
            c_sel = st.selectbox(
                "Select Consumer Match to Explain",
                sorted(avail_c),
                key="exp_c",
                format_func=lambda x: industry_names.get(x, x)
            )
        st.markdown('</div>', unsafe_allow_html=True)
            
        pair_record = pairs_df[(pairs_df.producer_id == p_sel) & (pairs_df.consumer_id == c_sel)].iloc[0].to_dict()
        exp_res = explainer.explain_pair(pair_record)
        
        p_name = industry_names.get(p_sel, p_sel)
        c_name = industry_names.get(c_sel, c_sel)
        m_name = material_names.get(pair_record.get('material_id'), pair_record.get('material_id'))
        
        st.write("")
        c_score, c_detail = st.columns([1, 1.8])
        
        with c_score:
            st.markdown(f"""
            <div class="section-card" style="text-align: center;">
                <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; letter-spacing: 0.06em;">MATCH VIABILITY CONFIDENCE</div>
                <div style="font-size: 3rem; font-weight: 700; color: #1A365D; margin: 12px 0;">{exp_res['score'] * 100:.1f}%</div>
                <div>{'<span class="badge-success">HIGH SYNERGY FIT</span>' if exp_res['score'] > 0.7 else '<span class="badge-warning">MODERATE FIT</span>'}</div>
                <hr style="margin: 20px 0; border: none; border-top: 1.5px solid #E2E8F0;">
                <div style="text-align: left; font-size: 0.92rem; color: #334155; line-height: 1.8;">
                    <div>• <b>Material:</b> {m_name}</div>
                    <div>• <b>Consumer Sector:</b> {pair_record.get('consumer_sector')}</div>
                    <div>• <b>Application:</b> {pair_record.get('application')}</div>
                    <div>• <b>Distance:</b> {pair_record.get('distance_km'):.1f} km</div>
                    <div>• <b>Unit Margin:</b> INR {pair_record.get('margin_per_tonne'):,.0f}/t</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_detail:
            st.markdown("##### Key Driver Breakdown")
            
            st.markdown("###### Favorable Catalysts")
            for strength in exp_res["strengths"]:
                st.markdown(f"""
                <div style="background-color: #F0FFF4; border: 1.5px solid #C6F6D5; border-left: 5px solid #38A169; padding: 12px 16px; margin-bottom: 10px; border-radius: 8px; font-size: 0.95rem; color: #22543D;">
                    [POSITIVE] {strength}
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("###### Cautionary Friction Factors")
            if exp_res["cautions"]:
                for caution in exp_res["cautions"]:
                    st.markdown(f"""
                    <div style="background-color: #FFFAF0; border: 1.5px solid #FEEBC8; border-left: 5px solid #DD6B20; padding: 12px 16px; margin-bottom: 10px; border-radius: 8px; font-size: 0.95rem; color: #7B341E;">
                        [CAUTION] {caution}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #F8FAFC; border: 1.5px solid #E2E8F0; border-left: 5px solid #94A3B8; padding: 12px 16px; border-radius: 8px; font-size: 0.95rem; color: #475569;">
                    No prohibitive constraints or quality gaps detected.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Models not ready.")


# =============================================================
# TAB 4: KNOWLEDGE GRAPH (GNN)
# =============================================================
with tab4:
    st.markdown("### Heterogeneous Industrial Symbiosis Knowledge Graph")
    st.caption("Visual graph network of industrial facilities, waste materials, and active symbiosis relationships with GNN link prediction.")
    st.write("")
    
    if len(nodes_df) > 0 and len(edges_df) > 0:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        g_ctrl1, g_ctrl2, g_ctrl3 = st.columns([1.5, 1.5, 1.5])
        
        with g_ctrl1:
            all_sectors = ["All Sectors"] + sorted([s for s in nodes_df[nodes_df.node_type == "industry"].sector.dropna().unique()])
            filter_sector = st.selectbox("Filter by Industrial Sector", all_sectors)
            
        with g_ctrl2:
            filter_rel = st.selectbox("Filter Relation Type", ["All Relations", "produces", "consumes", "traded_with"])
            
        with g_ctrl3:
            max_nodes = st.slider("Display Subgraph Density (Nodes)", min_value=15, max_value=min(72, len(nodes_df)), value=40)
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
        if filter_rel != "All Relations":
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
            line=dict(width=1.3, color="#CBD5E1"),
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
            "steel": "#E53E3E", "thermal_power": "#DD6B20", "cement": "#319795",
            "chemicals": "#805AD5", "sugar": "#D69E2E", "textile": "#3182CE",
            "paper": "#38A169", "rice_mill": "#4FD1C5", "auto_service": "#718096",
            "fertilizer": "#ED8936", "construction_materials": "#9F7AEA",
            "oil_recycling": "#4A5568", "plastic_recycling": "#38B2AC", "material": "#2D3748"
        }

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            data = G.nodes[node]
            ntype = data.get("node_type", "industry")
            sector = data.get("sector", "material")
            
            if ntype == "material":
                node_colors.append("#2B6CB0")
                node_sizes.append(18)
                node_symbols.append("diamond")
                node_texts.append(f"Material: {material_names.get(node, node)}")
            else:
                c = sector_color_map.get(sector, "#4A5568")
                node_colors.append(c)
                node_sizes.append(14)
                node_symbols.append("circle")
                node_texts.append(f"Facility: {industry_names.get(node, node)}<br>Sector: {sector}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[material_names.get(n, n).split(' (')[0] if G.nodes[n].get("node_type") == "material" else "" for n in G.nodes()],
            textposition="top center",
            textfont=dict(family="Times New Roman", size=10, color="#1A202C"),
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
                title=f"Knowledge Graph Subnetwork ({len(G.nodes())} Active Nodes, {len(G.edges())} Directed Links)",
                titlefont=dict(family="Times New Roman", size=16),
                showlegend=False,
                hovermode="closest",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(b=20, l=20, r=20, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        st.plotly_chart(fig_kg, use_container_width=True)
        
        if gnn_model is not None:
            st.write("")
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("##### Zero-Shot GNN Link Prediction Between Unconnected Nodes")
            col_l1, col_l2, col_l3 = st.columns([1.5, 1.5, 1])
            ind_nodes = nodes_df[nodes_df.node_type == "industry"].node_id.tolist()
            with col_l1:
                src_n = st.selectbox(
                    "Source Facility (Producer)",
                    ind_nodes,
                    index=0,
                    key="lp_src",
                    format_func=lambda x: industry_names.get(x, x)
                )
            with col_l2:
                dst_n = st.selectbox(
                    "Target Facility (Consumer)",
                    ind_nodes,
                    index=min(4, len(ind_nodes)-1),
                    key="lp_dst",
                    format_func=lambda x: industry_names.get(x, x)
                )
            with col_l3:
                st.write("")
                st.write("")
                prob = gnn_model.predict_link_probability(src_n, dst_n)
                st.metric("GNN Synergy Likelihood", f"{prob * 100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Graph data files not found.")


# =============================================================
# TAB 5: MULTI-OBJECTIVE PARETO RL
# =============================================================
with tab5:
    st.markdown("### Multi-Objective Deep RL & NSGA-II Pareto Optimization")
    st.caption("Continuous Pareto trade-off optimization balancing Economic Profit (INR/t), CO2 Reductions, and Transit Logistics.")
    st.write("")
    
    if rl_data is not None:
        top_pareto = rl_data.get("top_pareto_trades", pd.DataFrame())
        
        col_rl1, col_rl2 = st.columns([1.4, 1.2])
        with col_rl1:
            st.markdown("##### 3D Pareto Frontier Space")
            plot_pareto = top_pareto.copy()
            plot_pareto["Producer Name"] = plot_pareto["producer_id"].map(lambda x: industry_names.get(x, x))
            plot_pareto["Consumer Name"] = plot_pareto["consumer_id"].map(lambda x: industry_names.get(x, x))
            plot_pareto["Material Name"] = plot_pareto["material_id"].map(lambda x: material_names.get(x, x))
            
            fig_3d = px.scatter_3d(
                plot_pareto,
                x="margin_per_tonne",
                y="co2_saving_per_tonne",
                z="distance_km",
                color="pareto_rank",
                size="avg_monthly_tonnes",
                hover_name="Producer Name",
                hover_data=["Consumer Name", "Material Name", "consumer_sector"],
                labels={
                    "margin_per_tonne": "Margin (INR/t)",
                    "co2_saving_per_tonne": "CO2 Saved (kg/t)",
                    "distance_km": "Distance (km)",
                    "pareto_rank": "Pareto Rank"
                },
                color_continuous_scale="Viridis_r"
            )
            fig_3d.update_layout(
                font_family="Times New Roman",
                margin=dict(l=10, r=10, t=10, b=10),
                scene=dict(
                    xaxis_title="Margin (INR/t)",
                    yaxis_title="CO2 Saved (kg/t)",
                    zaxis_title="Distance (km)",
                    bgcolor="#FFFFFF"
                )
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with col_rl2:
            st.markdown("##### Pareto-Optimal Solutions (Rank 1 Frontier)")
            rank1_df = top_pareto[top_pareto.pareto_rank == 1].copy()
            rank1_df["producer_name"] = rank1_df["producer_id"].map(lambda x: industry_names.get(x, x))
            rank1_df["consumer_name"] = rank1_df["consumer_id"].map(lambda x: industry_names.get(x, x))
            rank1_df["material_name"] = rank1_df["material_id"].map(lambda x: material_names.get(x, x))
            
            st.dataframe(
                rank1_df[["producer_name", "consumer_name", "material_name", "margin_per_tonne", "co2_saving_per_tonne", "distance_km"]].round(1),
                column_config={
                    "producer_name": "Producer Facility",
                    "consumer_name": "Consumer Facility",
                    "material_name": "Material",
                    "margin_per_tonne": "Margin (INR/t)",
                    "co2_saving_per_tonne": "CO2 Saved (kg/t)",
                    "distance_km": "Dist (km)"
                },
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("RL Pareto model checkpoint not found.")


# =============================================================
# TAB 6: REGULATORY COMPLIANCE NER
# =============================================================
with tab6:
    st.markdown("### Regulatory Constraint Extraction & Compliance Parser")
    st.caption("Extracts environmental directives, permissible concentration thresholds, and legal storage limits from statutory gazettes.")
    st.write("")
    
    if ner_data is not None:
        rules_df = ner_data.get("extracted_rules", pd.DataFrame())
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        c_search, c_filter = st.columns([2, 1])
        with c_search:
            search_sub = st.text_input("Search Extracted Regulations by Substance or Property", "")
        with c_filter:
            filter_ob = st.selectbox("Obligation Type", ["All Obligations"] + list(rules_df.obligation.dropna().unique()))
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
                "substance": "Substance",
                "property": "Constrained Property",
                "threshold": "Threshold",
                "unit": "Unit",
                "obligation": "Obligation Directive",
                "raw_text": "Original Legal Clause"
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.write("")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Interactive Environmental Clause Parser")
        test_text = st.text_area(
            "Enter Raw Legal / Environmental Regulation Text to Extract Entities",
            "Facilities shall not accept blast furnace slag with heavy metal content above 140 ppm or moisture above 8.5 %."
        )
        
        if st.button("Extract Entities & Directives"):
            extractor = RegulatoryEntityExtractor()
            parsed = extractor.parse_text(test_text)
            
            st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1.5px solid #CBD5E1; border-radius: 10px; padding: 18px 22px; margin-top: 16px;">
                <div style="font-size: 1rem; font-weight: 700; color: #1A365D; margin-bottom: 12px;">Extraction Result:</div>
                <div style="margin-bottom: 8px;">• <b>Identified Substance:</b> <span class="entity-substance">{parsed['substance'].upper()}</span></div>
                <div style="margin-bottom: 8px;">• <b>Target Property:</b> <code>{parsed['property']}</code></div>
                <div style="margin-bottom: 8px;">• <b>Permissible Threshold:</b> <span class="entity-threshold">{parsed['threshold']}</span> <span class="entity-unit">{parsed['unit']}</span></div>
                <div>• <b>Regulatory Obligation:</b> <span class="entity-obligation">{parsed['obligation']}</span></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("NER model not found.")
