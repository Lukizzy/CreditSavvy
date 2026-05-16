"""
dashboard.py
------------
CreditSavvy — Credit Risk Scoring Dashboard
Main Streamlit application entry point.

Run with:
    streamlit run app/dashboard.py
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.explainer import get_shap_values, global_feature_importance

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditSavvy | Credit Risk Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; letter-spacing: -0.01em; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e8e8e8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label { color: #a0a0b0 !important; font-size: 0.8rem; }

/* Risk gauge colours */
.risk-low    { color: #22c55e; font-weight: 600; font-size: 1.8rem; }
.risk-medium { color: #f59e0b; font-weight: 600; font-size: 1.8rem; }
.risk-high   { color: #ef4444; font-weight: 600; font-size: 1.8rem; }

/* Metric cards */
.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.6rem;
}
.metric-card .label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .val   { font-size: 1.6rem; font-weight: 600; color: #f3f4f6; margin-top: 0.2rem; }

/* Verdict banner */
.verdict-low    { background:#052e16; border:1px solid #16a34a; border-radius:10px; padding:1rem 1.4rem; }
.verdict-medium { background:#451a03; border:1px solid #d97706; border-radius:10px; padding:1rem 1.4rem; }
.verdict-high   { background:#450a0a; border:1px solid #dc2626; border-radius:10px; padding:1rem 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Load model artefacts ───────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

@st.cache_resource
def load_artefacts():
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    lr_path   = os.path.join(MODEL_DIR, "lr_model.pkl")
    rf_path   = os.path.join(MODEL_DIR, "rf_model.pkl")

    if not os.path.exists(meta_path):
        return None, None, None, None

    with open(meta_path) as f:
        meta = json.load(f)
    with open(best_path, "rb") as f:
        best = pickle.load(f)
    with open(lr_path, "rb") as f:
        lr = pickle.load(f)
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)
    return meta, best, lr, rf

meta, best_model, lr_model, rf_model = load_artefacts()

# ── Sidebar — Applicant Input Form ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CreditSavvy")
    st.markdown("*Credit Risk Assessment System*")
    st.divider()
    st.markdown("### Applicant Details")

    limit_bal = st.number_input("Credit Limit (NT$)", min_value=10_000, max_value=1_000_000,
                                 value=200_000, step=10_000)
    age = st.slider("Age", 21, 75, 35)
    sex = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox("Education", [1, 2, 3, 4],
                              format_func=lambda x: {1:"Graduate", 2:"University",
                                                      3:"High School", 4:"Other"}[x])
    marriage = st.selectbox("Marital Status", [1, 2, 3],
                             format_func=lambda x: {1:"Married", 2:"Single", 3:"Other"}[x])

    st.markdown("### Recent Repayment Status")
    st.caption("-2 = No credit use, -1 = Paid in full, 0 = Revolving, 1+ = Months delayed")

    pay_map = {-2:"No use", -1:"Paid in full", 0:"Revolving",
                1:"1 mth delay", 2:"2 mth delay", 3:"3 mth delay",
                4:"4 mth delay", 5:"5 mth delay", 6:"6 mth delay",
                7:"7 mth delay", 8:"8 mth delay"}
    pay_options = list(pay_map.keys())

    pay0 = st.selectbox("Sep (most recent)", pay_options, index=2,
                         format_func=lambda x: f"{x}: {pay_map[x]}")
    pay2 = st.selectbox("Aug", pay_options, index=2,
                         format_func=lambda x: f"{x}: {pay_map[x]}")
    pay3 = st.selectbox("Jul", pay_options, index=2,
                         format_func=lambda x: f"{x}: {pay_map[x]}")
    pay4 = st.selectbox("Jun", pay_options, index=1,
                         format_func=lambda x: f"{x}: {pay_map[x]}")
    pay5 = st.selectbox("May", pay_options, index=1,
                         format_func=lambda x: f"{x}: {pay_map[x]}")
    pay6 = st.selectbox("Apr", pay_options, index=1,
                         format_func=lambda x: f"{x}: {pay_map[x]}")

    st.markdown("### Bill Amounts (NT$)")
    bill1 = st.number_input("Sep", 0, 500_000, 50_000, 1_000, key="b1")
    bill2 = st.number_input("Aug", 0, 500_000, 48_000, 1_000, key="b2")
    bill3 = st.number_input("Jul", 0, 500_000, 45_000, 1_000, key="b3")
    bill4 = st.number_input("Jun", 0, 500_000, 42_000, 1_000, key="b4")
    bill5 = st.number_input("May", 0, 500_000, 40_000, 1_000, key="b5")
    bill6 = st.number_input("Apr", 0, 500_000, 38_000, 1_000, key="b6")

    st.markdown("### Payment Amounts (NT$)")
    pay_a1 = st.number_input("Sep", 0, 200_000, 5_000, 500, key="pa1")
    pay_a2 = st.number_input("Aug", 0, 200_000, 5_000, 500, key="pa2")
    pay_a3 = st.number_input("Jul", 0, 200_000, 5_000, 500, key="pa3")
    pay_a4 = st.number_input("Jun", 0, 200_000, 5_000, 500, key="pa4")
    pay_a5 = st.number_input("May", 0, 200_000, 5_000, 500, key="pa5")
    pay_a6 = st.number_input("Apr", 0, 200_000, 5_000, 500, key="pa6")

    assess_btn = st.button("▶ Run Assessment", type="primary", use_container_width=True)

# ── Build input DataFrame ──────────────────────────────────────────────────────
input_data = {
    "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
    "MARRIAGE": marriage, "AGE": age,
    "PAY_0": pay0, "PAY_2": pay2, "PAY_3": pay3,
    "PAY_4": pay4, "PAY_5": pay5, "PAY_6": pay6,
    "BILL_AMT1": bill1, "BILL_AMT2": bill2, "BILL_AMT3": bill3,
    "BILL_AMT4": bill4, "BILL_AMT5": bill5, "BILL_AMT6": bill6,
    "PAY_AMT1": pay_a1, "PAY_AMT2": pay_a2, "PAY_AMT3": pay_a3,
    "PAY_AMT4": pay_a4, "PAY_AMT5": pay_a5, "PAY_AMT6": pay_a6,
}

feature_cols = list(input_data.keys())
X_input = pd.DataFrame([input_data])

# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("# Credit Risk Assessment Dashboard")
st.markdown("Powered by machine learning · Explainable AI · Built for loan officers")
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Risk Assessment", "📈 Model Performance", "ℹ️ About"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — RISK ASSESSMENT
# ════════════════════════════════════════════════════════════════════
with tab1:
    if meta is None:
        st.warning("⚠️ No trained model found. Please run `python models/train.py` first, "
                   "then restart the app.")
    else:
        feature_labels = meta["feature_labels"]
        prob = best_model.predict_proba(X_input)[0, 1]
        lr_prob = lr_model.predict_proba(X_input)[0, 1]
        rf_prob = rf_model.predict_proba(X_input)[0, 1]

        # Risk tier
        if prob < 0.25:
            tier, tier_css, tier_icon = "LOW RISK", "risk-low", "✅"
            verdict_css = "verdict-low"
            verdict_text = "This applicant presents a low probability of default. Standard approval criteria apply."
        elif prob < 0.55:
            tier, tier_css, tier_icon = "MEDIUM RISK", "risk-medium", "⚠️"
            verdict_css = "verdict-medium"
            verdict_text = "This applicant presents moderate default risk. Consider requesting additional documentation or reducing the credit limit."
        else:
            tier, tier_css, tier_icon = "HIGH RISK", "risk-high", "🚨"
            verdict_css = "verdict-high"
            verdict_text = "This applicant presents a high probability of default. Exercise caution — manual review is recommended."

        # ── Row 1: Gauge + metrics ─────────────────────────────────────────
        col_gauge, col_metrics = st.columns([1.4, 1])

        with col_gauge:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 52, "color": "#f3f4f6"}},
                title={"text": "Default Probability", "font": {"size": 16, "color": "#9ca3af"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#4b5563",
                              "tickfont": {"color": "#9ca3af"}},
                    "bar": {"color": "#ef4444" if prob >= 0.55 else
                                     "#f59e0b" if prob >= 0.25 else "#22c55e",
                             "thickness": 0.25},
                    "bgcolor": "#1f2937",
                    "steps": [
                        {"range": [0, 25],  "color": "#052e16"},
                        {"range": [25, 55], "color": "#1c1408"},
                        {"range": [55, 100],"color": "#1f0a0a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2},
                                   "thickness": 0.75, "value": prob * 100},
                }
            ))
            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(t=40, b=0, l=30, r=30),
                font={"color": "#f3f4f6"}
            )
            st.plotly_chart(gauge, use_container_width=True)

            # Risk tier badge
            st.markdown(f"""
            <div style="text-align:center; margin-top:-1rem;">
                <span class="{tier_css}">{tier_icon} {tier}</span>
            </div>""", unsafe_allow_html=True)

        with col_metrics:
            st.markdown("<br>", unsafe_allow_html=True)
            for label, val in [
                ("Best Model Estimate", f"{prob*100:.1f}%"),
                ("Logistic Regression", f"{lr_prob*100:.1f}%"),
                ("Random Forest",       f"{rf_prob*100:.1f}%"),
                ("Model Agreement",     "High" if abs(lr_prob-rf_prob)<0.08 else "Low"),
            ]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="val">{val}</div>
                </div>""", unsafe_allow_html=True)

        # Verdict banner
        st.markdown(f"""
        <div class="{verdict_css}" style="margin-top:1rem;">
            <strong>{tier_icon} Verdict</strong><br>
            <span style="color:#d1d5db;">{verdict_text}</span>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Row 2: SHAP explanation ────────────────────────────────────────
        st.markdown("### What's driving this prediction?")
        st.caption("Factors ranked by their influence on the default probability for this applicant.")

        shap_df = get_shap_values(best_model, X_input, feature_labels)
        top_n   = shap_df.head(10)

        colors = ["#ef4444" if v > 0 else "#22c55e" for v in top_n["shap_value"]]
        fig_shap = go.Figure(go.Bar(
            x=top_n["shap_value"],
            y=top_n["label"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in top_n["shap_value"]],
            textposition="outside",
        ))
        fig_shap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, margin=dict(t=10, b=10, l=10, r=60),
            xaxis=dict(title="SHAP value (impact on log-odds of default)",
                       gridcolor="#2a2d3a", color="#9ca3af"),
            yaxis=dict(autorange="reversed", color="#d1d5db"),
            font={"color": "#f3f4f6"},
        )
        fig_shap.add_vline(x=0, line_color="#4b5563", line_width=1)
        st.plotly_chart(fig_shap, use_container_width=True)

        st.caption("🔴 Red bars increase default risk · 🟢 Green bars decrease default risk")

        # ── Row 3: Bill vs Payment history ────────────────────────────────
        st.divider()
        st.markdown("### Payment Behaviour History")
        months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
        bills   = [bill6, bill5, bill4, bill3, bill2, bill1]
        payments= [pay_a6, pay_a5, pay_a4, pay_a3, pay_a2, pay_a1]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(name="Bill Amount", x=months, y=bills,
                                   marker_color="#3b82f6", opacity=0.7))
        fig_hist.add_trace(go.Bar(name="Payment Made", x=months, y=payments,
                                   marker_color="#22c55e", opacity=0.9))
        fig_hist.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=10, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#d1d5db")),
            xaxis=dict(gridcolor="#2a2d3a", color="#9ca3af"),
            yaxis=dict(title="NT$", gridcolor="#2a2d3a", color="#9ca3af"),
            font={"color": "#f3f4f6"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════
with tab2:
    if meta is None:
        st.warning("Run the training script first.")
    else:
        st.markdown("### Model Evaluation Summary")
        lrm = meta["lr_metrics"]
        rfm = meta["rf_metrics"]

        col1, col2 = st.columns(2)

        def model_card(m, cv, title):
            st.markdown(f"#### {title}")
            cols = st.columns(3)
            cols[0].metric("ROC-AUC", f"{m['roc_auc']:.4f}")
            cols[1].metric("Accuracy", f"{m['accuracy']:.4f}")
            cols[2].metric("Brier Score", f"{m['brier']:.4f}")
            st.caption(f"5-fold CV AUC: {cv['mean']:.4f} ± {cv['std']:.4f}")

            cm = np.array(m["confusion_matrix"])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["No Default", "Default"], y=["No Default", "Default"],
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=280, margin=dict(t=10, b=10),
                font={"color": "#f3f4f6"},
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col1:
            model_card(lrm, meta["cv_lr"], "Logistic Regression")
        with col2:
            model_card(rfm, meta["cv_rf"], "Random Forest")

        st.divider()
        st.markdown("### Global Feature Importance")
        st.caption("Which features matter most across all predictions?")

        fi_df = global_feature_importance(best_model, meta["feature_cols"], meta["feature_labels"])
        fig_fi = go.Figure(go.Bar(
            x=fi_df["importance"].head(15),
            y=fi_df["label"].head(15),
            orientation="h",
            marker_color="#6366f1",
        ))
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=420, margin=dict(t=10, b=10, l=10, r=20),
            xaxis=dict(title="Relative Importance", gridcolor="#2a2d3a", color="#9ca3af"),
            yaxis=dict(autorange="reversed", color="#d1d5db"),
            font={"color": "#f3f4f6"},
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # Dataset summary
        st.divider()
        st.markdown("### Training Data Summary")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Training Samples", f"{meta['train_size']:,}")
        col_b.metric("Test Samples", f"{meta['test_size']:,}")
        col_c.metric("Default Rate", f"{meta['default_rate']:.1%}")
        st.caption(f"Best model selected: **{meta['best_model']}**")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    ### About CreditSavvy

    **CreditSavvy** is a proof-of-concept credit risk scoring dashboard designed for
    loan officers and credit analysts who need fast, explainable risk assessments.

    #### Data Source
    - **Dataset**: Default of Credit Card Clients
    - **Source**: UCI Machine Learning Repository (ID 350)
    - **Reference**: Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining
      techniques for the predictive accuracy of probability of default of credit card clients.*
      Expert Systems with Applications, 36(2), 2473–2480.

    #### Models
    | Model | Why it's used |
    |---|---|
    | Logistic Regression | Statistically interpretable; each coefficient maps to a log-odds change; defensible under regulatory frameworks (EU AI Act, Basel III) |
    | Random Forest | Captures non-linear interactions; higher predictive accuracy; feature importances provide global explainability |

    #### Explainability
    Individual predictions are explained using **SHAP (SHapley Additive exPlanations)**
    values, which decompose the model's output fairly across all input features using
    cooperative game theory (Lundberg & Lee, 2017).

    #### Disclaimer
    This tool is a research prototype for educational purposes. It must not be used
    for real credit decisions without regulatory approval, bias auditing, and compliance
    review. Credit decisions affecting individuals are subject to the Equal Credit
    Opportunity Act (ECOA) and equivalent legislation.

    ---
    *Developed as part of CETM46 Data Science Product Development — University of Sunderland*
    """)
