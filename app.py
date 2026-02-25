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
    debt_ebitda = st.number_input("Debt / EBITDA
