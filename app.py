import os
import time
import subprocess
import sys
import json
import re

# --- STEP 1: INSTALL LIBRARIES ---
print("CHECKING LIBRARIES...")
try:
    import streamlit
    import yfinance
    import pyngrok
    print("Libraries already installed.")
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "streamlit", "xgboost", "plotly", "pyngrok", "yfinance", "openpyxl", "tqdm"])
    print("Installation Complete!")

import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from google.colab import files
from tqdm import tqdm

# --- STEP 2: TRAIN MODEL ---
print("\nTRAINING AI MODEL...")
np.random.seed(42)
n_samples = 1000
debt = np.random.gamma(2, 2, n_samples)
int_cov = np.random.uniform(0, 20, n_samples)
pledged = np.random.uniform(0, 100, n_samples)
curr_ratio = np.random.uniform(0.2, 5, n_samples)
risk_score = (debt * 0.4) + (pledged * 0.02) - (int_cov * 0.3) - (curr_ratio * 0.8)
status = [1 if (1 / (1 + np.exp(-risk_score))) > np.random.random() else 0 for risk_score in risk_score]
df_train = pd.DataFrame({'Debt_to_EBITDA': debt, 'Interest_Coverage': int_cov, 'Pledged_Pct': pledged, 'Current_Ratio': curr_ratio, 'Status': status})
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05)
model.fit(df_train[['Debt_to_EBITDA', 'Interest_Coverage', 'Pledged_Pct', 'Current_Ratio']], df_train['Status'])
model.save_model("creditsentinel_model.json")
print("Model Trained!")

# --- STEP 3: UPLOAD FILE ---
print("\nUPLOAD 'Filtered Nifty data.csv' NOW:")
uploaded = files.upload()

if not uploaded:
    print("No file uploaded. Please re-run.")
else:
    filename = next(iter(uploaded))
    print(f"Processing: {filename}")

    try:
        if filename.endswith('.csv'): df = pd.read_csv(filename)
        else: df = pd.read_excel(filename)
        df.columns = df.columns.str.strip().str.upper()

        # 1. IDENTIFY SYMBOL COLUMN
        symbol_col = next((c for c in df.columns if 'YAHOO' in c), None)
        if not symbol_col:
            symbol_col = next((c for c in df.columns if 'SYMBOL' in c or 'TICKER' in c), None)

        # 2. IDENTIFY PLEDGE COLUMN
        pledge_col = next((c for c in df.columns if 'PLEDGE' in c and 'DEMAT' in c), None)
        if not pledge_col: pledge_col = next((c for c in df.columns if 'PLEDGE' in c and '%' in c), None)

        # 3. IDENTIFY SECTOR COLUMN
        sector_col = next((c for c in df.columns if 'SECTOR' in c or 'INDUSTRY' in c), None)

        print(f"TICKER SOURCE: '{symbol_col}'")

        tickers = []
        pledge_map = {}
        sector_map = {}

        for index, row in df.iterrows():
            raw_ticker = str(row[symbol_col]).strip().upper()
            if not raw_ticker or raw_ticker == "NAN": continue

            tickers.append(raw_ticker)

            try:
                val = float(str(row[pledge_col]).replace("%", "").strip()) if pledge_col else 0.0
                pledge_map[raw_ticker] = val
            except: pledge_map[raw_ticker] = 0.0

            if sector_col: sector_map[raw_ticker] = str(row[sector_col])

        tickers = sorted(list(set(tickers)))
        print(f"Extracted {len(tickers)} symbols to scan.")

    except Exception as e:
        print(f"Error: {e}")
        tickers = []

# --- STEP 4: HARVESTER ---
def fetch_data(ticker):
    try:
        stock = yf.Ticker(ticker)

        # 1. Info Fetch & Market Cap Check
        try:
            info = stock.info
            if not info: return None
            mkt_cap = info.get('marketCap', 0)
            if mkt_cap is None or mkt_cap == 0:
                return None
        except: return None

        # 2. Financials
        q_inc = stock.quarterly_income_stmt
        q_bal = stock.quarterly_balance_sheet
        a_inc = stock.income_stmt
        a_bal = stock.balance_sheet

        def get_history(financials, balance_sheet, is_annual=False):
            hist = []
            if financials.empty or balance_sheet.empty: return []
            for date in financials.columns:
                try:
                    ebitda = financials.loc['EBITDA', date] if 'EBITDA' in financials.index else financials.loc['Net Income', date]
                    debt = balance_sheet.loc['Total Debt', date] if 'Total Debt' in balance_sheet.index else 0.0

                    if pd.isna(ebitda): ebitda = 1.0
                    if pd.isna(debt): debt = 0.0

                    final_ebitda = ebitda * 4 if not is_annual else ebitda
                    if final_ebitda == 0: final_ebitda = 1.0

                    d_e = round(debt / final_ebitda, 2)
                    hist.append({"date": date.strftime('%Y-%m-%d'), "debt_ebitda": d_e})
                except: pass
            return hist

        history = get_history(q_inc, q_bal, is_annual=False)
        if not history: history = get_history(a_inc, a_bal, is_annual=True)

        revenue_data = []
        if not a_inc.empty and 'Total Revenue' in a_inc.index:
            for date in a_inc.columns[:4]:
                try:
                    rev = a_inc.loc['Total Revenue', date]
                    revenue_data.append({"year": date.year, "revenue": rev})
                except: pass

        latest_debt_val = 0.0
        if not history:
            try:
                latest_debt = info.get('totalDebt', 0.0)
                latest_ebitda = info.get('ebitda', 1.0)
                d_e = round(latest_debt / latest_ebitda, 2) if latest_ebitda else 0.0
                history = [{"date": "2025-01-01", "debt_ebitda": d_e}]
                latest_debt_val = d_e
            except: return None
        else:
            latest_debt_val = history[0]['debt_ebitda']

        curr_ratio = info.get('currentRatio', 1.2)
        if not curr_ratio: curr_ratio = 1.2

        try:
            inc = q_inc if not q_inc.empty else a_inc
            if not inc.empty:
                intr = abs(inc.iloc[:,0].get('Interest Expense', 0.0))
                ebit = inc.iloc[:,0].get('EBIT', 1.0)
                int_cov = round(ebit/intr, 2) if intr > 0 else 100.0
            else: int_cov = 2.0
        except: int_cov = 2.0

        return {
            "name": info.get('longName', ticker),
            "ticker": ticker,
            "debt_ebitda": latest_debt_val,
            "curr_ratio": curr_ratio,
            "int_cov": int_cov,
            "pledged": pledge_map.get(ticker, 0.0),
            "market_cap": mkt_cap,
            "high_52": info.get('fiftyTwoWeekHigh', 0),
            "low_52": info.get('fiftyTwoWeekLow', 0),
            "sector": sector_map.get(ticker, info.get('sector', 'Manufacturing')),
            "history": history,
            "revenue": revenue_data
        }
    except: return None

print(f"SCANNING {len(tickers)} STOCKS...")
cache_data = {}
valid_count = 0
pbar = tqdm(tickers)

for ticker in pbar:
    data = fetch_data(ticker)
    if data:
        cache_data[ticker] = data
        valid_count += 1
        pbar.set_description(f"Found: {valid_count}")
    if valid_count % 50 == 0:
        with open("stock_cache.json", "w") as f: json.dump(cache_data, f)

with open("stock_cache.json", "w") as f: json.dump(cache_data, f)
print(f"FINAL LIST: {valid_count} Companies.")

# --- STEP 5: APP CODE ---
app_code = """
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
"""
with open("app.py", "w") as f: f.write(app_code)

# --- STEP 6: LAUNCH ---
print("LAUNCHING SERVERS...")
!pkill -9 -f streamlit
!pkill -9 -f cloudflared
!pkill -9 -f localtunnel

subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"])
if not os.path.exists("cloudflared"):
    !wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    !chmod +x cloudflared
!nohup ./cloudflared tunnel --url http://localhost:8501 > cf_tunnel.log 2>&1 &
!npm install -g localtunnel > /dev/null 2>&1
subprocess.Popen(["npx", "localtunnel", "--port", "8501"], stdout=open("lt_tunnel.log", "w"), stderr=subprocess.STDOUT)

time.sleep(10)
print("\nSERVER READY! Use either link:")
try:
    with open("cf_tunnel.log", "r") as f:
        for line in f:
            if ".trycloudflare.com" in line:
                print(f"LINK 1: https://{line.split('https://')[1].split('.trycloudflare.com')[0]}.trycloudflare.com")
                break
except: pass
try:
    with open("lt_tunnel.log", "r") as f:
        for line in f:
            if "your url is" in line:
                print(f"LINK 2: {line.split('is: ')[1].strip()}")
                break
except: pass
