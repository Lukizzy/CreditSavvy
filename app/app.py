"""
app.py
------
CreditSavvy — Credit Risk Scoring Dashboard
Built with Streamlit for the CETM46 Data Science Product Development module.

Run with:
    streamlit run app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from models.predictor import predict, model_loaded, get_model_name, get_feature_names

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditSavvy — Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS — refined financial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  h1, h2, h3 { font-family: 'DM Serif Display', serif; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stNumberInput label,
  section[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* Main area */
  .main .block-container { padding-top: 2rem; max-width: 1200px; }

  /* Metric cards */
  .metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .metric-card .label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.35rem;
  }
  .metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #0f172a;
    line-height: 1;
  }
  .metric-card .sub {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.3rem;
  }

  /* Risk badge */
  .risk-badge {
    display: inline-block;
    padding: 0.4rem 1.1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }

  /* Section heading */
  .section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #0f172a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
  }

  /* Warning banner */
  .warn-banner {
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    color: #92400e;
    font-size: 0.85rem;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("<div style='font-size:2.6rem;margin-top:0.2rem'>💳</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin:0;font-size:2rem;color:#0f172a'>CreditSavvy</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin:0;color:#64748b;font-size:0.9rem'>Credit Risk Scoring & Explainability Dashboard</p>", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:1rem 0 1.5rem'>", unsafe_allow_html=True)

# Model status check
if not model_loaded():
    st.markdown("""
    <div class='warn-banner'>
      ⚠️ <strong>Model not trained yet.</strong>
      Run <code>python models/train_model.py</code> from the project root, then refresh this page.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  Sidebar — applicant input form
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Applicant Details")
    st.markdown("<p style='font-size:0.78rem;color:#64748b;margin-top:-0.5rem'>Enter the applicant's financial profile</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Account Information**")
    credit_limit = st.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000,
                                    value=100000, step=10000)
    age = st.slider("Age", 20, 80, 35)

    st.markdown("**Demographics**")
    sex = st.selectbox("Sex", ["Male", "Female"])
    education = st.selectbox("Education Level",
                              ["Graduate School", "University", "High School", "Other"])
    marriage = st.selectbox("Marital Status", ["Single", "Married", "Other"])

    st.markdown("**Payment History (recent 6 months)**")
    st.caption("−2 = no consumption, −1 = paid in full, 0 = revolving credit, 1–8 = months delayed")
    pay_sep = st.slider("September", -2, 8, 0)
    pay_aug = st.slider("August", -2, 8, 0)
    pay_jul = st.slider("July", -2, 8, 0)
    pay_jun = st.slider("June", -2, 8, 0)
    pay_may = st.slider("May", -2, 8, 0)
    pay_apr = st.slider("April", -2, 8, 0)

    st.markdown("**Bill Amounts (NT$)**")
    bill_sep = st.number_input("September Bill", 0, 500000, 50000, 5000)
    bill_aug = st.number_input("August Bill",    0, 500000, 48000, 5000)
    bill_jul = st.number_input("July Bill",      0, 500000, 45000, 5000)
    bill_jun = st.number_input("June Bill",      0, 500000, 43000, 5000)
    bill_may = st.number_input("May Bill",       0, 500000, 40000, 5000)
    bill_apr = st.number_input("April Bill",     0, 500000, 38000, 5000)

    st.markdown("**Payments Made (NT$)**")
    paid_sep = st.number_input("September Payment", 0, 500000, 5000, 1000)
    paid_aug = st.number_input("August Payment",    0, 500000, 5000, 1000)
    paid_jul = st.number_input("July Payment",      0, 500000, 5000, 1000)
    paid_jun = st.number_input("June Payment",      0, 500000, 5000, 1000)
    paid_may = st.number_input("May Payment",       0, 500000, 5000, 1000)
    paid_apr = st.number_input("April Payment",     0, 500000, 5000, 1000)

    run_btn = st.button("🔍 Assess Risk", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
#  Encode categorical inputs
# ─────────────────────────────────────────────
SEX_MAP       = {"Male": 1, "Female": 2}
EDU_MAP       = {"Graduate School": 1, "University": 2, "High School": 3, "Other": 4}
MARRIAGE_MAP  = {"Single": 2, "Married": 1, "Other": 3}

features = {
    "credit_limit": credit_limit,
    "sex":          SEX_MAP[sex],
    "education":    EDU_MAP[education],
    "marriage":     MARRIAGE_MAP[marriage],
    "age":          age,
    "pay_sep":      pay_sep,
    "pay_aug":      pay_aug,
    "pay_jul":      pay_jul,
    "pay_jun":      pay_jun,
    "pay_may":      pay_may,
    "pay_apr":      pay_apr,
    "bill_sep":     bill_sep,
    "bill_aug":     bill_aug,
    "bill_jul":     bill_jul,
    "bill_jun":     bill_jun,
    "bill_may":     bill_may,
    "bill_apr":     bill_apr,
    "paid_sep":     paid_sep,
    "paid_aug":     paid_aug,
    "paid_jul":     paid_jul,
    "paid_jun":     paid_jun,
    "paid_may":     paid_may,
    "paid_apr":     paid_apr,
}


# ─────────────────────────────────────────────
#  Default display before assessment
# ─────────────────────────────────────────────
if "result" not in st.session_state:
    st.info("👈 Fill in the applicant details in the sidebar and click **Assess Risk** to generate a prediction.")

    # Show sample dataset stats
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "credit_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.markdown("<div class='section-heading'>📊 Training Dataset Overview</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='label'>Total Records</div>
              <div class='value'>{len(df):,}</div>
              <div class='sub'>applicant profiles</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            default_rate = df["default"].mean()
            st.markdown(f"""
            <div class='metric-card'>
              <div class='label'>Default Rate</div>
              <div class='value'>{default_rate:.1%}</div>
              <div class='sub'>historical average</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='label'>Features</div>
              <div class='value'>{len(df.columns)-1}</div>
              <div class='sub'>predictive variables</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='label'>Model</div>
              <div class='value' style='font-size:1rem;padding-top:0.4rem'>{get_model_name()}</div>
              <div class='sub'>production classifier</div>
            </div>""", unsafe_allow_html=True)

        # Default rate by education
        st.markdown("<br>", unsafe_allow_html=True)
        edu_map_r = {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}
        df["education_label"] = df["education"].map(edu_map_r)
        edu_default = df.groupby("education_label")["default"].mean().reset_index()
        fig_edu = px.bar(
            edu_default, x="education_label", y="default",
            labels={"education_label": "Education Level", "default": "Default Rate"},
            title="Default Rate by Education Level",
            color="default",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
        )
        fig_edu.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="DM Sans",
            coloraxis_showscale=False,
            title_font_family="DM Serif Display",
        )
        fig_edu.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_edu, use_container_width=True)


# ─────────────────────────────────────────────
#  Run prediction
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner("Analysing applicant profile …"):
        result = predict(features)
    st.session_state["result"] = result
    st.session_state["features"] = features


# ─────────────────────────────────────────────
#  Display results
# ─────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    prob   = result["probability"]
    band   = result["risk_band"]
    colour = result["band_colour"]
    shap   = result["shap_values"]

    # ── Summary row ──
    st.markdown("<div class='section-heading'>Assessment Result</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns([2, 2, 3])

    with r1:
        gauge_pct = int(prob * 100)
        st.markdown(f"""
        <div class='metric-card' style='text-align:center'>
          <div class='label'>Default Probability</div>
          <div class='value' style='color:{colour};font-size:2.6rem'>{gauge_pct}%</div>
          <div class='sub'>P(default next month)</div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class='metric-card' style='text-align:center'>
          <div class='label'>Risk Classification</div>
          <div style='margin:0.5rem 0'>
            <span class='risk-badge' style='background:{colour}22;color:{colour};border:1px solid {colour}'>
              {band}
            </span>
          </div>
          <div class='sub'>Based on 6-month profile</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        # Simple gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 28, "family": "DM Serif Display"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": colour, "thickness": 0.3},
                "steps": [
                    {"range": [0, 20],  "color": "#dcfce7"},
                    {"range": [20, 40], "color": "#fef9c3"},
                    {"range": [40, 60], "color": "#ffedd5"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": colour, "width": 3},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            height=200, margin=dict(t=20, b=10, l=20, r=20),
            paper_bgcolor="white", font_family="DM Sans",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP feature importance ──
    st.markdown("<div class='section-heading'>🔍 Feature Contribution Analysis</div>", unsafe_allow_html=True)
    st.caption("Shows which features pushed the risk score up (positive) or down (negative). Helps explain the model decision to the applicant.")

    # Build dataframe and sanitise: replace NaN/Inf with 0 so the chart always renders
    shap_df = pd.DataFrame(
        list(shap.items()), columns=["Feature", "Contribution"]
    )
    shap_df["Contribution"] = pd.to_numeric(shap_df["Contribution"], errors="coerce").fillna(0.0)
    shap_df["Contribution"] = shap_df["Contribution"].replace([float("inf"), float("-inf")], 0.0)

    shap_df = shap_df.sort_values("Contribution", key=abs, ascending=False).head(12)

    shap_df["Colour"] = shap_df["Contribution"].apply(
        lambda v: "#ef4444" if v > 0 else "#22c55e"
    )
    shap_df["Label"] = shap_df["Contribution"].apply(
        lambda v: f"{v:+.4f}" if v != 0 else "0.0000"
    )

    fig_shap = go.Figure(go.Bar(
        x=shap_df["Contribution"],
        y=shap_df["Feature"],
        orientation="h",
        marker_color=shap_df["Colour"],
        text=shap_df["Label"],
        textposition="outside",
    ))
    # Detect if values are all zero (last-resort fallback) or unsigned (importance fallback)
    all_zero    = shap_df["Contribution"].abs().sum() == 0
    using_shap  = not all_zero and shap_df["Contribution"].min() < 0  # true SHAP has negatives

    if all_zero:
        st.info("Feature contributions could not be computed for this prediction. The chart will update once the model is fully loaded.", icon="ℹ️")
    elif not using_shap:
        st.caption("ℹ️ Showing signed feature importance (SHAP library not available). Red = above-average feature value increases risk; green = below-average decreases risk.")

    x_axis_label = "SHAP Contribution to Default Probability" if using_shap else "Feature Importance (signed)"

    fig_shap.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="DM Sans",
        xaxis_title=x_axis_label,
        yaxis={"autorange": "reversed"},
        height=420,
        margin=dict(t=10, b=30, l=160, r=80),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # ── Payment timeline ──
    st.markdown("<div class='section-heading'>📈 Payment History Timeline</div>", unsafe_allow_html=True)
    feat = st.session_state["features"]
    months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    pay_vals  = [feat["pay_apr"], feat["pay_may"], feat["pay_jun"],
                 feat["pay_jul"], feat["pay_aug"], feat["pay_sep"]]
    bill_vals = [feat["bill_apr"], feat["bill_may"], feat["bill_jun"],
                 feat["bill_jul"], feat["bill_aug"], feat["bill_sep"]]
    paid_vals = [feat["paid_apr"], feat["paid_may"], feat["paid_jun"],
                 feat["paid_jul"], feat["paid_aug"], feat["paid_sep"]]

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Bar(
        x=months, y=bill_vals, name="Bill Amount",
        marker_color="#cbd5e1", opacity=0.9
    ))
    fig_timeline.add_trace(go.Bar(
        x=months, y=paid_vals, name="Payment Made",
        marker_color="#3b82f6"
    ))
    fig_timeline.add_trace(go.Scatter(
        x=months, y=pay_vals, name="Payment Status",
        mode="lines+markers", yaxis="y2",
        line=dict(color="#f59e0b", width=2),
        marker=dict(size=8),
    ))
    fig_timeline.update_layout(
        barmode="overlay",
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="DM Sans",
        yaxis=dict(title="Amount (NT$)"),
        yaxis2=dict(title="Delay (months)", overlaying="y", side="right",
                    range=[-2, 8]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=320,
        margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── Recommendation box ──
    if prob >= 0.60:
        rec_colour = "#fee2e2"
        rec_border = "#ef4444"
        rec_text   = "⛔ <strong>Decline recommended.</strong> This applicant presents a very high default risk. Consider requesting additional security or guarantor arrangements."
    elif prob >= 0.40:
        rec_colour = "#ffedd5"
        rec_border = "#f97316"
        rec_text   = "⚠️ <strong>Further review required.</strong> The applicant shows elevated risk. A reduced credit limit or closer monitoring may be appropriate."
    elif prob >= 0.20:
        rec_colour = "#fef9c3"
        rec_border = "#f59e0b"
        rec_text   = "🔶 <strong>Approve with caution.</strong> Moderate risk profile. Standard approval with routine monitoring is suggested."
    else:
        rec_colour = "#dcfce7"
        rec_border = "#22c55e"
        rec_text   = "✅ <strong>Approve.</strong> Low default probability. The applicant presents a healthy credit profile."

    st.markdown(f"""
    <div style='background:{rec_colour};border-left:4px solid {rec_border};
                border-radius:8px;padding:1rem 1.2rem;margin-top:0.5rem;font-size:0.92rem'>
      {rec_text}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "⚠️ CreditSavvy provides a data-driven risk estimate to support, not replace, "
        "human judgement. All credit decisions must comply with applicable regulations."
    )
