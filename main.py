import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# C. 規格化配置 (14 檔監控標的完整保留)
# ==========================================
SYSTEM_CONFIG = {
    "VERSION": "v13.2 完整旗艦版",
    "ADR_THRESHOLD": 5.0,  
    "CHIP_DAYS": 150,
    "CLOSING_DATE": "2026-02-11", # 2026 農曆封關日
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

# --- 2. 核心運算引擎 (多重指標與對齊防錯) ---
class TaiwanStockCommander2026:
    def __init__(self, token):
        self.api = DataLoader()
        try:
            if hasattr(self.api, 'login'): self.api.login(token=token.strip())
            else: self.api.token = token.strip()
        except: pass

    @st.cache_data(ttl=300)
    def get_global_weather(_self):
        """全球氣候監控 (ADR 與匯率)"""
        try:
            tsm_adr = yf.Ticker("TSM").history(period="2d")
            sox = yf.Ticker("^SOX").history(period="2d")
            twd = yf.Ticker("TWD=X").history(period="2d") 
            tsm_tw = yf.Ticker("2330.TW").history(period="2d")
            fx = twd['Close'].iloc[-1]
            adr_c = tsm_adr['Close'].iloc[-1]
            tw_c = tsm_tw['Close'].iloc[-1]
            sox_p = ((sox['Close'].iloc[-1] / sox['Close'].iloc[-2]) - 1) * 100
            premium = (((adr_c * fx) / 5) / tw_c - 1) * 100
            return premium, fx, sox_p
        except: return 0, 32.5, 0

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id):
        """戰略數據運算 (RS, KD, VWAP)"""
        days = SYSTEM_CONFIG["CHIP_DAYS"]
        df = yf.Ticker(f"{stock_id}.TW").history(period=f"{days}d")
        if df.empty: return pd.DataFrame(), 0, 0, 0, 0, 0
        df.index = df.index.tz_localize(None).normalize()
        
        # RS 相對強度對齊
        try:
            mkt = yf.Ticker("0050.TW").history(period=f"{days}d")
            mkt.index = mkt.index.tz_localize(None).normalize()
            df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100
        except: df['RS_Index'] = 0

        # KD 指標計算 (9, 3, 3)
        l9, h9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
        df['K'] = ((df['Close'] - l9) / (h9 - l9) * 100).ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()

        # 籌碼面與法人成本 (VWAP)
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

# (1) 側邊欄：流程優化排序 (v12.9)
st.sidebar.header(f"🦅 指揮中心 {SYSTEM_CONFIG['VERSION']}")
if st.sidebar.button("🔄 核心數據強制刷新"):
    st.cache_data.clear(); st.rerun()

st.sidebar.divider()
c_cat = st.sidebar.selectbox("引擎分類", list(SYSTEM_CONFIG["STOCKS"].keys()))
c_name = st.sidebar.selectbox("監控標的", list(SYSTEM_CONFIG["STOCKS"][c_cat].keys()))
stock_id = SYSTEM_CONFIG["STOCKS"][c_cat][c_name]

st.sidebar.divider()
with st.sidebar.expander("🛡️ 戰略指令判定指南", expanded=True):
    st.markdown("""
    | 狀態 | ADR 溢價 | 指令 |
    | :--- | :--- | :--- |
    | **🟢 守穩** | < 5% | **✅ 執行** |
    | **🟢 守穩** | > 5% | **🟡 觀望** |
    | **🔴 破線** | > 5% | **❌ 取消** |
    | **🔴 破線** | < -2% | **💎 校正** |
    """)

with st.sidebar.expander("📝 操作紀律提醒 (SOP)", expanded=False):
    st.markdown("""
    **1. 盤後選股 (15:30)**
    - RS > 0 + KD黃金交叉
    
    **2. 盤前定調 (22:30)**
    - ADR > 5% 絕不追高
    
    **3. 開盤決斷 (09:05)**
    - 價格需 > 法人成本
    """)

# (2) 主畫面：置頂看板
adr_p, fx_now, sox_p = commander.get_global_weather()
st.markdown(f"### 🌍 全球氣候看板 (ADR: **{adr_p:.1f}%** | USD/TWD: **{fx_now:.2f}**)")

# 封關倒數提醒
closing_dt = datetime.strptime(SYSTEM_CONFIG["CLOSING_DATE"], "%Y-%m-%d")
days_left = (closing_dt - datetime.now()).days
if 0 < days_left <= 12:
    st.warning(f"🧧 2026 農曆封關倒數 **{days_left}** 天。最後交易日：{SYSTEM_CONFIG['CLOSING_DATE']}")

st.divider()

# (3) 分頁顯示
tab_open, tab_post, tab_adr = st.tabs(["☀️ 09:05 決斷", "📊 15:30 盤後分析", "🌌 22:30 美股觀察"])
df_main, f_m, i_m, rs_m, k_m, d_m = commander.get_strategic_data(stock_id)
m_cost = i_m if "高息" in c_cat else f_m
price_now = yf.Ticker(f"{stock_id}.TW").fast_info.last_price

with tab_open:
    st.subheader(f"⚔️ {c_name} 指令與建議")
    k1, k2 = st.columns([1, 2])
    with k1:
        st.metric("目前價格", f"${price_now:.2f}", delta=f"${price_now - m_cost:.1f}")
        st.write("狀態：" + ("✅ 守穩執行" if price_now > m_cost else "🛑 破線觀望"))
    with k2:
        budget = st.number_input("今日預算 (NTD)", value=100000, step=10000)
        total_s = int(budget / price_now) if price_now > 0 else 0
        st.info(f"建議：**{total_s // 1000}** 張 又 **{total_s % 1000}** 股")

with tab_post:
    # A. 封關留倉健檢儀 (v13.1)
    st.subheader("🧧 2026 封關留倉戰略掃描")
    if st.button("🚀 啟動全標的留倉健檢"):
        def scan_closing():
            res = []
            for eng, stocks in SYSTEM_CONFIG["STOCKS"].items():
                for name, sid in stocks.items():
                    _, fc, ic, rs, k, d_v = commander.get_strategic_data(sid)
                    cost = ic if "高息" in eng else fc
                    p = yf.Ticker(f"{sid}.TW").fast_info.last_price
                    score = (1 if rs > 0 else 0) + (1 if k > d_v else 0) + (1 if p > cost else 0)
                    status = "🟢 建議留倉" if score == 3 else ("🟡 減碼續抱" if p > cost else "🔴 建議出清")
                    res.append({"引擎": eng[0:2], "標的名稱": name, "戰略評分": "⭐"*score, "留倉建議": status})
            return pd.DataFrame(res)
        st.table(scan_closing().sort_values("戰略評分", ascending=False))
        st.caption("註：建議僅保留 ⭐⭐⭐ 標的過年。")

    st.divider()

    # B. 個股深度指標 (v13.0)
    st.subheader(f"📊 {c_name} 深度指標分析")
    score_m = (1 if rs_m > 0 else 0) + (1 if k_m > d_m else 0) + (1 if price_now > m_cost else 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("戰略星等", "⭐"*score_m if score_m > 0 else "❌")
    c2.metric("RS 強度", f"{rs_m:.1f}", delta="強勢" if rs_m > 0 else "弱勢")
    c3.metric("KD 指標", f"K:{k_m:.1f}", delta="黃金交叉" if k_m > d_m else "死亡交叉", delta_color="normal" if k_m > d_m else "inverse")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_main.index[-90:], y=df_main['Close'].iloc[-90:], name="價格"))
    c_line = df_main['Invest_Cost'] if "高息" in c_cat else df_main['Foreign_Cost']
    fig.add_trace(go.Scatter(x=df_main.index[-90:], y=c_line.iloc[-90:], name="法人防線 (VWAP)", line=dict(dash='dot')))
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

with tab_adr:
    st.subheader("🌌 全球連動資訊")
    st.metric("ADR 溢價率", f"{adr_p:.2f}%", delta="過熱" if adr_p > 5 else "正常")
    st.metric("即時台幣匯率", f"{fx_now:.2f}")
    st.caption("溢價 > 17% 屬罕見過熱，收斂壓力極大，歷史回檔機率 58%。")

st.caption(f"系統規格：{SYSTEM_CONFIG['VERSION']} | 核心判定：法人成本線 (VWAP)")
