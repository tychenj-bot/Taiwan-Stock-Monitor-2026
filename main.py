import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# C. 規格化配置 (新增 2026 封關參數)
# ==========================================
SYSTEM_CONFIG = {
    "VERSION": "v13.1 封關守護版",
    "ADR_THRESHOLD": 5.0,  
    "CHIP_DAYS": 150,
    "CLOSING_DATE": "2026-02-11", # 2026 封關日
    "STOCKS": {
        "🔥 成長進攻": {
            "台積電 (2330)": "2330", "鴻海 (2317)": "2317", "聯發科 (2454)": "2454", 
            "復華50 (00991A)": "00991A", "統一主動 (00981A)": "00981A"
        },
        "🛡️ 市值防禦": {
            "元大50 (0050)": "0050", "富邦50 (006208)": "006208", "國泰50 (00922)": "00922",
            "台達電 (2308)": "2308", "台泥 (1101)": "1101"
        },
        "💰 穩健領息": {
            "元大高息 (0056)": "0056", "國泰高息 (00878)": "00878", "群益高息 (00919)": "00919", 
            "復華優息 (00929)": "00929"
        }
    }
}

# --- 1. 系統環境配置 ---
st.set_page_config(page_title=f"戰略指揮中心 {SYSTEM_CONFIG['VERSION']}", layout="wide")
if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN"); st.stop()
FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]

# --- 2. 核心運算引擎 ---
class TaiwanStockCommander2026:
    def __init__(self, token):
        self.api = DataLoader()
        try:
            if hasattr(self.api, 'login'): self.api.login(token=token.strip())
            else: self.api.token = token.strip()
        except: pass

    @st.cache_data(ttl=300)
    def get_global_weather(_self):
        try:
            tsm_adr = yf.Ticker("TSM").history(period="2d")
            sox = yf.Ticker("^SOX").history(period="2d")
            twd = yf.Ticker("TWD=X").history(period="2d") 
            tsm_tw = yf.Ticker("2330.TW").history(period="2d")
            fx = twd['Close'].iloc[-1]
            adr_c = tsm_adr['Close'].iloc[-1]
            sox_p = ((sox['Close'].iloc[-1] / sox['Close'].iloc[-2]) - 1) * 100
            tw_c = tsm_tw['Close'].iloc[-1]
            premium = (((adr_c * fx) / 5) / tw_c - 1) * 100
            return premium, fx, sox_p
        except: return 0, 32.5, 0

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id):
        days = SYSTEM_CONFIG["CHIP_DAYS"]
        df = yf.Ticker(f"{stock_id}.TW").history(period=f"{days}d")
        if df.empty: return pd.DataFrame(), 0, 0, 0, 0, 0
        df.index = df.index.tz_localize(None).normalize()
        
        try:
            mkt = yf.Ticker("0050.TW").history(period=f"{days}d")
            mkt.index = mkt.index.tz_localize(None).normalize()
            df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100
        except: df['RS_Index'] = 0

        # KD 計算
        l9, h9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
        df['K'] = ((df['Close'] - l9) / (h9 - l9) * 100).ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()

        # 籌碼與 VWAP
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            for name in ['Foreign', 'Investment']:
                sub = df_chip[df_chip['name'].str.contains(name, case=False)].copy()
                sub['date'] = pd.to_datetime(sub['date']).dt.normalize()
                sub = sub.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
                df[f'{name.lower()}_net'] = sub['buy'] - sub['sell']
        except: df['foreign_net'] = df['investment_net'] = 0
        
        df = df.fillna(0)
        def calc_vwap(net_col):
            costs = []; last = np.nan
            for i in range(len(df)):
                win = df.iloc[max(0, i-19) : i+1]
                buys = win[win[net_col] > 0]
                if not buys.empty: last = (buys['Close'] * buys[net_col]).sum() / buys[net_col].sum()
                costs.append(last)
            return pd.Series(costs, index=df.index).ffill().bfill()
        
        df['Foreign_Cost'] = calc_vwap('foreign_net')
        df['Invest_Cost'] = calc_vwap('investment_net')
        return df, df['Foreign_Cost'].iloc[-1], df['Invest_Cost'].iloc[-1], df['RS_Index'].iloc[-1], df['K'].iloc[-1], df['D'].iloc[-1]

# --- 3. UI 介面 ---
commander = TaiwanStockCommander2026(FINMIND_TOKEN)

# 側邊欄
st.sidebar.header(f"🦅 指揮中心 {SYSTEM_CONFIG['VERSION']}")
if st.sidebar.button("🔄 核心數據強制刷新"):
    st.cache_data.clear(); st.rerun()

st.sidebar.divider()
c_cat = st.sidebar.selectbox("引擎分類", list(SYSTEM_CONFIG["STOCKS"].keys()))
c_name = st.sidebar.selectbox("監控標的", list(SYSTEM_CONFIG["STOCKS"][c_cat].keys()))
stock_id = SYSTEM_CONFIG["STOCKS"][c_cat][c_name]

# 主畫面氣候
adr_p, fx_now, sox_p = commander.get_global_weather()
st.markdown(f"### 🌍 全球氣候看板 (ADR: **{adr_p:.1f}%** | USD/TWD: **{fx_now:.2f}**)")

# 2026 封關倒數提醒
days_to_closing = (datetime.strptime(SYSTEM_CONFIG["CLOSING_DATE"], "%Y-%m-%d") - datetime.now()).days
if 0 < days_to_closing <= 10:
    st.warning(f"🧧 2026 農曆封關倒數 **{days_to_closing}** 個日曆日。歷史經驗：封關前 2-5 天易有紅包行情。")

st.divider()

# 分頁決策系統
tab_open, tab_post, tab_adr = st.tabs(["☀️ 09:05 決斷", "📊 15:30 盤後分析", "🌌 22:30 美股觀察"])
df_main, f_m, i_m, rs_m, k_m, d_m = commander.get_strategic_data(stock_id)
m_cost = i_m if "高息" in c_cat else f_m
price_now = yf.Ticker(f"{stock_id}.TW").fast_info.last_price

with tab_open:
    st.subheader(f"⚔️ {c_name} 開盤執行")
    k1, k2 = st.columns([1, 2])
    k1.metric("目前價格", f"${price_now:.2f}", delta=f"${price_now - m_cost:.1f}")
    with k2:
        budget = st.number_input("預算 (NTD)", value=100000)
        st.info(f"建議：{int(budget/price_now)//1000} 張 又 {int(budget/price_now)%1000} 股")

with tab_post:
    st.subheader("🧧 封關留倉健檢儀 (Scan All)")
    
    # 執行留倉掃描
    def scan_for_closing():
        results = []
        for eng, stocks in SYSTEM_CONFIG["STOCKS"].items():
            for name, sid in stocks.items():
                d, fc, ic, rs, k, d_val = commander.get_strategic_data(sid)
                cost = ic if "高息" in eng else fc
                p = yf.Ticker(f"{sid}.TW").fast_info.last_price
                
                # 留倉標準：守穩成本 + RS強 + KD黃金交叉
                is_safe = p > cost
                score = (1 if rs > 0 else 0) + (1 if k > d_val else 0) + (1 if is_safe else 0)
                
                status = "🟢 建議留倉" if score == 3 else ("🟡 減碼續抱" if is_safe else "🔴 建議出清")
                results.append({"引擎": eng[0:2], "標的": name, "分數": "⭐"*score, "建議": status})
        return pd.DataFrame(results)

    if st.button("🚀 啟動 2026 全標的留倉健檢"):
        scan_df = scan_for_closing()
        st.table(scan_df.sort_values("分數", ascending=False))
        st.success("💡 建議僅保留 ⭐⭐⭐ 標的過年，降低休市期間波動風險。")

    st.divider()
    st.write(f"📊 **{c_name}** 個股深度評分：RS={rs_m:.1f} | KD={'黃金交叉' if k_m > d_m else '死亡交叉'}")

with tab_adr:
    st.subheader("🌌 全球連動位階")
    st.metric("ADR 溢價率", f"{adr_p:.2f}%", delta="過熱" if adr_p > 5 else "正常")
    st.caption("溢價 > 17% 屬於罕見過熱，歷史上收斂機率達 58%。")

st.caption(f"{SYSTEM_CONFIG['VERSION']} | 核心判定：法人成本線 (VWAP)")
