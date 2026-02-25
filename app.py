import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os

st.set_page_config(layout="wide", page_title="CreditSentinel")

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("creditsentinel_model.json")
    return model
model = load_model()

def load_data():
    if os.path.exists("stock_cache.json"):
        with open("stock_cache.json", "r") as f: return json.load(f)
    return {}
cache = load_data()

# --- SIDEBAR ---
st.sidebar.title("CreditSentinel Terminal")
st.sidebar.caption("Advanced Credit Risk Analytics")
st.sidebar.divider()

ticker_list = ["Manual Entry"] + sorted(list(cache.keys()))
selected_ticker = st.sidebar.selectbox("Select Issuer / Ticker:", ticker_list)

d_val, i_val, p_val, c_val = 3.5, 2.0, 0.0, 1.2
full_name = "Manual Entry"
mkt_cap, high_52, low_52 = 0, 0, 0
sec_val = "General Industry"
hist_data = []
rev_data = []

if selected_ticker != "Manual Entry":
    data = cache[selected_ticker]
    d_val = float(data['debt_ebitda'])
    i_val = float(data['int_cov'])
    p_val = float(data['pledged'])
    c_val = float(data['curr_ratio'])
    full_name = data.get('name', selected_ticker)
    mkt_cap = data.get('market_cap', 0)
    high_52 = data.get('high_52', 0)
    low_52 = data.get('low_52', 0)
    sec_val = data.get('sector', "General Industry")
    hist_data = data.get('history', [])
    rev_data = data.get('revenue', [])
    st.sidebar.success(f"Loaded: {selected_ticker}")

# SECTOR LIMITS
thresholds = {
    "Information Technology": {"safe": 1.5, "risk": 3.0},
    "Utilities": {"safe": 6.0, "risk": 9.0},
    "Financial": {"safe": 10.0, "risk": 12.0},
    "Manufacturing": {"safe": 3.0, "risk": 5.0},
    "Real Estate": {"safe": 5.0, "risk": 7.0},
    "Energy": {"safe": 4.0, "risk": 6.5},
    "Communication": {"safe": 4.0, "risk": 7.0}
}
matched_sector = "Manufacturing"
for key in thresholds.keys():
    if sec_val and key in sec_val:
        matched_sector = key
        break
sector_limits = thresholds.get(matched_sector, thresholds["Manufacturing"])

st.sidebar.markdown("### Sector Covenants")
st.sidebar.info(f"**Peer Group:** {matched_sector}")
c1, c2 = st.sidebar.columns(2)
c1.metric("Safe Leverage", f"<{sector_limits['safe']}x")
c2.metric("Breach Level", f">{sector_limits['risk']}x", delta_color="inverse")

st.sidebar.divider()
st.sidebar.markdown("### Sensitivity Analysis")
with st.sidebar.form("analysis"):
    debt_ebitda = st.number_input("Debt / EBITDA (x)", -100.0, 1000000.0, d_val)
    int_cov = st.number_input("Interest Coverage (x)", -100.0, 1000000.0, i_val)
    pledged = st.number_input("Promoter Pledge (%)", 0.0, 100.0, p_val)
    curr_ratio = st.number_input("Current Ratio (x)", 0.0, 1000000.0, c_val)
    submitted = st.form_submit_button("Run Credit Model")

# --- MAIN DASHBOARD ---
st.title("CreditSentinel: Emerging Markets Risk Framework")
st.markdown("---")

def format_currency(value):
    if value >= 10**7: return f"₹{value/10**7:.2f} Cr"
    return f"₹{value:.2f}"

if submitted or selected_ticker != "Manual Entry":
    prob = model.predict_proba(pd.DataFrame([[debt_ebitda, int_cov, pledged, curr_ratio]], columns=['Debt_to_EBITDA', 'Interest_Coverage', 'Pledged_Pct', 'Current_Ratio']))[:, 1][0]
    risk_adj = 0
    if debt_ebitda > sector_limits['risk']: risk_adj += 0.20
    if pledged > 20: risk_adj += 0.10
    final_prob = np.clip(prob + risk_adj, 0.01, 0.99)
    score = final_prob * 100

    if score < 35: color, status = "#00C853", "INVESTMENT GRADE (IG)"
    elif score < 70: color, status = "#FFA000", "SPECULATIVE / HY"
    else: color, status = "#D50000", "DISTRESSED / DEFAULT"

    # TOP METRICS
    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.markdown(f"<h2 style='color: {color}; margin-bottom:0px;'>{status}</h2>", unsafe_allow_html=True)
        st.caption(f"Default Probability Score: {int(score)}/100")

        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = score,
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "black", 'thickness': 0.3},
                'steps': [
                    {'range': [0, 35], 'color': "#00C853"},
                    {'range': [35, 70], 'color': "#FFA000"},
                    {'range': [70, 100], 'color': "#D50000"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': score}
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(f"### {full_name}")
        st.metric("Market Cap", format_currency(mkt_cap))
        c2a, c2b = st.columns(2)
        c2a.metric("52W High", f"₹{high_52}")
        c2b.metric("52W Low", f"₹{low_52}")

        st.markdown("#### Credit Memo")
        with st.container(border=True):
            if debt_ebitda <= sector_limits['safe']: st.markdown("- **Leverage:** Conservative vs Peers.")
            elif debt_ebitda > sector_limits['risk']: st.markdown("- **Leverage:** Breach of Sector Norms.")

            if int_cov > 5.0: st.markdown("- **Coverage:** Strong Serviceability.")
            elif int_cov < 2.0: st.markdown("- **Coverage:** Cash Flow Stress.")

            if pledged == 0: st.markdown("- **Structure:** Clean Equity.")
            else: st.markdown(f"- **Governance:** {pledged}% Shares Encumbered.")

    # CHARTS ROW
    st.markdown("---")
    b1, b2 = st.columns(2)

    with b1:
        if hist_data:
            st.subheader("Capital Structure Trend")
            df_hist = pd.DataFrame(hist_data)
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            df_hist = df_hist.sort_values('date')
            fig_trend = px.line(df_hist, x='date', y='debt_ebitda', title="Historical Net Debt / EBITDA", markers=True)
            fig_trend.add_hline(y=sector_limits['risk'], line_dash="dash", line_color="red", annotation_text="Covenant Breach")
            st.plotly_chart(fig_trend, use_container_width=True)
        else: st.info("No historical leverage data.")

    with b2:
        if rev_data:
            st.subheader("Top Line Performance")
            df_rev = pd.DataFrame(rev_data).sort_values("year")
            fig_rev = px.bar(df_rev, x='year', y='revenue', title="Annual Revenue", color_discrete_sequence=['#1f77b4'])
            fig_rev.update_layout(xaxis_type='category')
            st.plotly_chart(fig_rev, use_container_width=True)
        else: st.info("No revenue data available.")
