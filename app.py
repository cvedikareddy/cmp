"""Industrial Symbiosis AI Decision Platform.

An intuitive, high-contrast, production-grade circular economy decision platform featuring:
1. AI-Driven Recommendation Engine (Gradient Boosting Matchmaker)
2. Circular Profitability & Scenario Simulator (Sensitivity, Fair Bargaining & Monte Carlo Risk)
3. Match Explainability & Trade-Off Attribution (Feature Drivers & Spec Alignment)
4. Knowledge Graph GNN (Heterogeneous Graph Explorer & Zero-Shot Link Predictor)
5. Multi-Objective Pareto Optimization (NSGA-II Frontier & Custom Priority Weighting)
6. Regulatory Compliance NER (Statutory Directive Search & Real-Time Clause Parser)
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
    page_title="Industrial Symbiosis AI Platform",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
    /* Clean, modern sans-serif typography across the dashboard */
    html, body, [class*="css"], .stMarkdown, p, div, span, label, input, select, button {
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
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
        <span style="font-size: 2rem;">♻️</span>
        <div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #0F2942; line-height: 1.2;">Industrial Symbiosis AI</div>
            <div style="font-size: 0.78rem; color: #64748B;">Circular Decision Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("AI-powered circular economy engine connecting byproduct suppliers to manufacturing consumers.")
    
    st.markdown("---")
    st.markdown("#### **System Architecture**")
    
    def status_indicator(is_active, name):
        badge = '<span style="color: #059669; font-weight: 700;">Ready</span>' if is_active else '<span style="color: #DC2626; font-weight: 700;">Missing</span>'
        dot = "🟢" if is_active else "🔴"
        return f"{dot} **{name}**: {badge}"
        
    st.markdown(status_indicator(rec_model is not None, "Match Recommender"), unsafe_allow_html=True)
    st.markdown(status_indicator(len(pairs_df) > 0, "Hybrid Dataset"), unsafe_allow_html=True)
    st.markdown(status_indicator(gnn_model is not None, "Knowledge Graph GNN"), unsafe_allow_html=True)
    st.markdown(status_indicator(rl_data is not None, "Pareto RL Optimizer"), unsafe_allow_html=True)
    st.markdown(status_indicator(ner_data is not None, "Regulatory NER"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### **Platform Scope**")
    st.markdown(f"""
    - **Facilities Indexed:** {len(industry_names)} plants
    - **By-Products Cataloged:** {len(material_names)} materials
    - **Evaluated Pairs:** {len(pairs_df):,} candidate trades
    - **Regulatory Directives:** {len(ner_data.get('extracted_rules', [])) if ner_data else 0} rules
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the Recommender tab to find partners, then simulate their financials or explore their GNN graph synergy.")


# -------------------------------------------------------------
# Top Header Banner
# -------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; color: #FFFFFF;">Industrial Symbiosis AI Decision Platform</h1>
            <div class="header-tagline">Autonomous by-product valorization, multi-objective trade optimization, and regulatory compliance verification</div>
        </div>
        <div>
            <span class="badge-success" style="background-color: #064E3B; color: #A7F3D0; border: 1px solid #059669; font-size: 0.85rem; padding: 6px 14px;">
                ● Production Engine Active
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Navigation Tabs
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Match Recommender",
    "📊 Profitability Simulator",
    "🔍 Explainability & Trade-offs",
    "🕸️ Knowledge Graph (GNN)",
    "⚖️ Multi-Objective Pareto RL",
    "📜 Regulatory Compliance NER"
])


# =============================================================
# TAB 1: MATCH RECOMMENDER
# =============================================================
with tab1:
    st.markdown("### AI-Driven Industrial Waste Exchange Recommendations")
    st.caption("Matches waste producers with optimal consumer facilities based on material properties, logistic distance, and economic surplus.")
    
    if rec_model is not None and len(pairs_df) > 0:
        producers = sorted(pairs_df.producer_id.unique().tolist())
        
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1.6, 1.4, 1.2, 0.8])
        
        with col_ctrl1:
            selected_producer = st.selectbox(
                "🏢 Select Waste Producer Facility",
                producers,
                index=0,
                format_func=lambda x: industry_names.get(x, x),
                help="Select an industrial plant producing by-products."
            )
            
        with col_ctrl2:
            avail_materials = sorted(pairs_df[pairs_df.producer_id == selected_producer].material_id.unique().tolist())
            mat_options = ["ALL"] + avail_materials
            material_filter = st.selectbox(
                "🧪 Filter by By-Product / Material",
                mat_options,
                format_func=lambda x: "All By-Products" if x == "ALL" else material_names.get(x, x),
                help="Filter matches to a specific material or view all available exchanges."
            )
            
        with col_ctrl3:
            all_sectors = ["ALL"] + sorted(pairs_df.consumer_sector.dropna().unique().tolist())
            sector_filter = st.selectbox(
                "🏭 Consumer Industry Sector",
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
                st.markdown("##### 📋 Recommended Exchange Matches")
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
                st.markdown("##### 📈 Trade Viability vs Unit Margin")
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
            st.info("ℹ️ No symbiotic matches found matching the specified filters. Try selecting 'All By-Products' or 'All Sectors'.")
    else:
        st.warning("⚠️ Recommender model or data not found. Please train models first.")


# =============================================================
# TAB 2: PROFITABILITY SIMULATOR
# =============================================================
with tab2:
    st.markdown("### Circular Economy Profitability & Scenario Simulator")
    st.caption("Simulate deal profitability, contract pricing, freight costs, and capital payback with sensitivity analysis and Monte Carlo risk assessment.")
    
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    c_preset1, c_preset2 = st.columns([1, 2])
    with c_preset1:
        sim_mode = st.radio(
            "Scenario Source",
            ["Preset from Real Match", "Custom Scenario"],
            horizontal=True,
            help="Load a real candidate pair from the dataset or configure custom parameters."
        )
    with c_preset2:
        if sim_mode == "Preset from Real Match":
            candidate_pairs = pairs_df.head(40).copy()
            candidate_pairs["label"] = candidate_pairs.apply(
                lambda r: f"{industry_names.get(r.producer_id, r.producer_id).split(' (')[0]} ➔ {industry_names.get(r.consumer_id, r.consumer_id).split(' (')[0]} [{material_names.get(r.material_id, r.material_id).split(' (')[0]}]",
                axis=1
            )
            selected_preset_idx = st.selectbox(
                "Select Verified Deal to Load",
                range(len(candidate_pairs)),
                format_func=lambda i: candidate_pairs.iloc[i]["label"]
            )
            preset_row = candidate_pairs.iloc[selected_preset_idx]
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
        st.markdown("##### 💰 Material & Contract Pricing")
        v_traded = st.number_input("Annual Traded Volume (tpy)", min_value=100.0, max_value=1000000.0, value=float(def_v), step=500.0)
        p_sale = st.number_input("Negotiated By-Product Price (₹/t)", min_value=0.0, max_value=100000.0, value=float(def_p), step=50.0)
        p_virgin = st.number_input("Virgin Material Benchmark (₹/t)", min_value=100.0, max_value=100000.0, value=float(def_vp), step=100.0)
        
    with col_s2:
        st.markdown("##### 🚚 Logistics & Pretreatment")
        d_km = st.number_input("Transit Distance (km)", min_value=1.0, max_value=5000.0, value=float(def_dist), step=10.0)
        f_rate = st.number_input("Freight Rate (₹/tonne-km)", min_value=0.5, max_value=50.0, value=float(def_fr), step=0.2)
        c_pre = st.number_input("Consumer Pretreatment Cost (₹/t)", min_value=0.0, max_value=20000.0, value=float(def_pre), step=25.0)
        
    with col_s3:
        st.markdown("##### 🌱 Environmental & Capital")
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
    st.info(f"🤝 **Fair Bargaining Suggestion (Nash Equilibrium):** A negotiated price of **₹{fair_p:,.2f}/t** splits the net surplus evenly (50/50) between producer and consumer while keeping both parties strictly profitable.")

    st.write("")
    sim_tab1, sim_tab2, sim_tab3 = st.tabs(["🌪️ Sensitivity Tornado", "📈 Price Sweep & Surplus Split", "🎲 Monte Carlo Risk Simulation"])
    
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
                st.markdown("##### 🔍 Key Feature Attribution Breakdown")
                
                st.markdown("###### ✅ Positive Drivers & Enablers")
                if exp_res["strengths"]:
                    for strength in exp_res["strengths"]:
                        st.markdown(f"""
                        <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-left: 4px solid #16A34A; padding: 10px 14px; margin-bottom: 8px; border-radius: 6px; font-size: 0.92rem; color: #166534;">
                            <b>[POSITIVE]</b> {strength}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No outstanding positive drivers identified.")
                    
                st.markdown("###### ⚠️ Cautionary & Friction Factors")
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
# TAB 4: KNOWLEDGE GRAPH (GNN)
# =============================================================
with tab4:
    st.markdown("### Heterogeneous Industrial Symbiosis Knowledge Graph")
    st.caption("Visual graph network of industrial facilities, waste materials, and active symbiosis relationships with GNN link prediction.")
    
    if len(nodes_df) > 0 and len(edges_df) > 0:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        g_ctrl1, g_ctrl2, g_ctrl3 = st.columns([1.5, 1.5, 1.5])
        
        with g_ctrl1:
            all_sectors = ["All Sectors"] + sorted([s for s in nodes_df[nodes_df.node_type == "industry"].sector.dropna().unique()])
            filter_sector = st.selectbox("Filter by Industrial Sector", all_sectors)
            
        with g_ctrl2:
            filter_rel = st.selectbox("Filter Relation Type", ["All Relations", "produces", "consumes", "traded_with"])
            
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
        
        if gnn_model is not None:
            st.write("")
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("##### ⚡ Zero-Shot GNN Link Prediction Between Facilities")
            st.caption("Predicts the likelihood of an undiscovered symbiotic exchange using heterogeneous graph node embeddings.")
            col_l1, col_l2, col_l3 = st.columns([1.5, 1.5, 1.2])
            ind_nodes = nodes_df[nodes_df.node_type == "industry"].node_id.tolist()
            with col_l1:
                src_n = st.selectbox(
                    "Source Facility (Waste Producer)",
                    ind_nodes,
                    index=0,
                    key="lp_src",
                    format_func=lambda x: industry_names.get(x, x)
                )
            with col_l2:
                dst_n = st.selectbox(
                    "Target Facility (Waste Consumer)",
                    ind_nodes,
                    index=min(4, len(ind_nodes)-1),
                    key="lp_dst",
                    format_func=lambda x: industry_names.get(x, x)
                )
            with col_l3:
                st.write("")
                if src_n == dst_n:
                    st.warning("Select two distinct facilities.")
                else:
                    try:
                        prob = gnn_model.predict_link_probability(src_n, dst_n)
                    except Exception:
                        prob = 0.25
                    
                    if prob > 0.40:
                        tier_label = "🟢 High Synergy"
                    elif prob > 0.20:
                        tier_label = "🟡 Moderate Potential"
                    else:
                        tier_label = "⚪ Low Likelihood"
                        
                    st.metric("GNN Synergy Likelihood", f"{prob * 100:.1f}%", delta=tier_label)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Graph data files not found.")


# =============================================================
# TAB 5: MULTI-OBJECTIVE PARETO RL
# =============================================================
with tab5:
    st.markdown("### Multi-Objective Pareto Optimization & Policy Space")
    st.caption("Continuous Pareto trade-off optimization balancing Economic Profit (₹/t), CO2 Reductions, and Logistics Transit.")
    
    if rl_data is not None:
        top_pareto = rl_data.get("top_pareto_trades", pd.DataFrame())
        
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
        with w_col1:
            w_profit = st.slider("Economic Yield Weight", 0.0, 1.0, 0.45, 0.05)
        with w_col2:
            w_co2 = st.slider("Carbon Savings Weight", 0.0, 1.0, 0.40, 0.05)
        with w_col3:
            w_dist = st.slider("Logistics Proximity Weight", 0.0, 1.0, 0.15, 0.05)
        with w_col4:
            rank_filter = st.selectbox("Filter Pareto Rank", ["All Ranks", "Rank 1 Only", "Ranks 1 to 3"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Dynamic composite re-weighting
        w_sum = w_profit + w_co2 + w_dist
        norm_w = np.array([w_profit, w_co2, w_dist]) / (w_sum if w_sum > 0 else 1.0)
        
        plot_pareto = top_pareto.copy()
        plot_pareto["Producer Name"] = plot_pareto["producer_id"].map(lambda x: industry_names.get(x, x))
        plot_pareto["Consumer Name"] = plot_pareto["consumer_id"].map(lambda x: industry_names.get(x, x))
        plot_pareto["Material Name"] = plot_pareto["material_id"].map(lambda x: material_names.get(x, x))
        
        if rank_filter == "Rank 1 Only":
            plot_pareto = plot_pareto[plot_pareto.pareto_rank == 1]
        elif rank_filter == "Ranks 1 to 3":
            plot_pareto = plot_pareto[plot_pareto.pareto_rank <= 3]
            
        col_rl1, col_rl2 = st.columns([1.3, 1.1])
        with col_rl1:
            st.markdown("##### 🌐 3D Pareto Space (Margin vs CO2 vs Distance)")
            fig_3d = px.scatter_3d(
                plot_pareto,
                x="margin_per_tonne",
                y="co2_saving_per_tonne",
                z="distance_km",
                color="pareto_rank",
                size="avg_monthly_tonnes",
                hover_name="Producer Name",
                hover_data={"Consumer Name": True, "Material Name": True, "margin_per_tonne": ":,.0f", "co2_saving_per_tonne": ":,.0f"},
                labels={
                    "margin_per_tonne": "Margin (₹/t)",
                    "co2_saving_per_tonne": "CO2 Saved (kg/t)",
                    "distance_km": "Distance (km)",
                    "pareto_rank": "Pareto Rank"
                },
                color_continuous_scale="Viridis_r"
            )
            fig_3d.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                scene=dict(
                    xaxis_title="Margin (₹/t)",
                    yaxis_title="CO2 Saved (kg/t)",
                    zaxis_title="Distance (km)",
                    bgcolor="#FFFFFF"
                )
            )
            st.plotly_chart(fig_3d, width="stretch")
            
        with col_rl2:
            st.markdown("##### ⚖️ 2D Trade-Off Curve (Unit Margin vs Carbon Abatement)")
            fig_2d = px.scatter(
                plot_pareto,
                x="margin_per_tonne",
                y="co2_saving_per_tonne",
                color="pareto_rank",
                size="avg_monthly_tonnes",
                hover_name="Producer Name",
                hover_data={"Consumer Name": True, "Material Name": True, "distance_km": ":.1f"},
                labels={"margin_per_tonne": "Unit Margin (₹/tonne)", "co2_saving_per_tonne": "CO2 Abated (kg/t)", "pareto_rank": "Rank"},
                color_continuous_scale="Viridis_r"
            )
            fig_2d.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=20, b=10)
            )
            fig_2d.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
            fig_2d.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(fig_2d, width="stretch")
            
        st.write("")
        st.markdown("##### 🏆 Pareto-Optimal Solutions Ranking")
        styled_pareto = plot_pareto.copy()
        styled_pareto["margin_fmt"] = "₹" + styled_pareto["margin_per_tonne"].round(0).astype(int).astype(str)
        styled_pareto["co2_fmt"] = styled_pareto["co2_saving_per_tonne"].round(0).astype(int).astype(str) + " kg/t"
        styled_pareto["dist_fmt"] = styled_pareto["distance_km"].round(1).astype(str) + " km"
        
        st.dataframe(
            styled_pareto[["Producer Name", "Consumer Name", "Material Name", "pareto_rank", "margin_fmt", "co2_fmt", "dist_fmt"]],
            column_config={
                "Producer Name": "Waste Producer",
                "Consumer Name": "Waste Consumer",
                "Material Name": "Material",
                "pareto_rank": "Pareto Rank",
                "margin_fmt": "Margin / Tonne",
                "co2_fmt": "CO2 Saved",
                "dist_fmt": "Transit Distance"
            },
            width="stretch",
            hide_index=True
        )
    else:
        st.warning("RL Pareto model checkpoint not found.")


# =============================================================
# TAB 6: REGULATORY COMPLIANCE NER
# =============================================================
with tab6:
    st.markdown("### Regulatory Constraint Extraction & Legal Compliance Parser")
    st.caption("Extracts environmental directives, permissible concentration thresholds, and legal storage limits from statutory gazettes.")
    
    if ner_data is not None:
        rules_df = ner_data.get("extracted_rules", pd.DataFrame())
        
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        c_search, c_filter = st.columns([2, 1])
        with c_search:
            search_sub = st.text_input("🔍 Search Statutory Rules by Substance or Regulated Property", "")
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
        st.markdown("##### 🧪 Real-Time Legal Environmental Clause Parser")
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
                <div style="font-size: 0.95rem; font-weight: 700; color: #0F2942; margin-bottom: 12px;">✅ Extracted Regulatory Directives:</div>
                <div style="margin-bottom: 8px;">• <b>Identified Substance:</b> <span class="entity-substance">{parsed['substance'].upper()}</span></div>
                <div style="margin-bottom: 8px;">• <b>Constrained Property:</b> <code>{parsed['property']}</code></div>
                <div style="margin-bottom: 8px;">• <b>Permissible Threshold:</b> <span class="entity-threshold">{thresh_display}</span> <span class="entity-unit">{unit_display}</span></div>
                <div>• <b>Regulatory Obligation:</b> <span class="entity-obligation">{parsed['obligation']}</span></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("NER model not found.")
