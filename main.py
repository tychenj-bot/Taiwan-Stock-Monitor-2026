import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# C. 系統規格化配置
# ==========================================
SYSTEM_CONFIG = {
    "VERSION": "v12.9 流程優化版",
    "ADR_THRESHOLD": 5.0,  
    "MA_PERIOD": 20,
    "CHIP_DAYS": 150,
    "STOCKS": {
        "🔥 成長進攻": {
            "台積電 (2330)": "2330", 
            "鴻海 (2317)": "2317", 
            "聯發科 (2454)": "2454", 
            "復華50 (00991A)": "00991A", 
            "統一主動 (00981A)": "00981A"
        },
        "🛡️ 市值防禦": {
            "元大50 (0050)": "0050", 
            "富邦50 (006208)": "006208", 
            "國泰50 (00922)": "00922",
            "台達電 (2308)": "2308", 
            "台泥 (1101)": "1101"
        },
        "💰 穩健領息": {
            "元大高息 (0056)": "0056", 
            "國泰高息 (00878)": "00878", 
            "群益高息 (00919)": "00919", 
            "復華優息 (00929)": "00929"
        }
    }
}

# --- 1. 系統環境配置 ---
st.set_page_config(page_title=f"戰略指揮中心 {SYSTEM_CONFIG['VERSION']}", layout="wide")

if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN")
    st.stop()
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
        if df.empty: return pd.DataFrame(), 0, 0, 0
        df.index = df.index.tz_localize(None).normalize()
        try:
            mkt = yf.Ticker("0050.TW").history(period=f"{days}d")
            mkt.index = mkt.index.tz_localize(None).normalize()
            df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100
        except: df['RS_Index'] = 0
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
        return df, df['Foreign_Cost'].iloc[-1], df['Invest_Cost'].iloc[-1], df['RS_Index'].iloc[-1]

    def get_realtime_status(self, stock_id):
        try:
            info = yf.Ticker(f"{stock_id}.TW").fast_info
            return info.open if info.open else info.last_price
        except: return 0

# --- 3. UI 介面 ---
commander = TaiwanStockCommander2026(FINMIND_TOKEN)

# (1) 側邊欄：重新編排順序
st.sidebar.header(f"🦅 指揮中心 {SYSTEM_CONFIG['VERSION']}")

# 刷新按鈕放在最上方以利操作
if st.sidebar.button("🔄 核心數據強制刷新"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

# A. 標的選擇優先
c_cat = st.sidebar.selectbox("引擎分類 (Engine)", list(SYSTEM_CONFIG["STOCKS"].keys()))
c_name = st.sidebar.selectbox("監控標的 (Target)", list(SYSTEM_CONFIG["STOCKS"][c_cat].keys()))
stock_id = SYSTEM_CONFIG["STOCKS"][c_cat][c_name]

st.sidebar.divider()

# B. 指令提示與紀律提醒
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
    - 主力連買 3天 + RS > 0
    
    **2. 盤前定調 (22:30)**
    - ADR > 5% 絕不追高
    
    **3. 開盤決斷 (09:05)**
    - 股價需 **> 法人成本線**
    - 使用下方金流計算器下單
    """)

# (2) 主畫面：置頂氣候
adr_p, fx_now, sox_p = commander.get_global_weather()
st.markdown(f"### 🌍 全球氣候看板 (ADR: **{adr_p:.1f}%** | USD/TWD: **{fx_now:.2f}**)")

# (3) 三引擎核心視覺卡片
st.divider()
core_list = [("🔥 成長", "00991A", "復華未來50 (00991A)"), 
             ("🛡️ 市值", "0050", "元大台灣50 (0050)"), 
             ("💰 高息", "00878", "國泰高息 (00878)")]
cols = st.columns(3)
for i, (tag, sid, sname) in enumerate(core_list):
    with cols[i]:
        df_c, fc, ic, _ = commander.get_strategic_data(sid)
        price_c = commander.get_realtime_status(sid)
        target_cost = ic if "高息" in tag else fc
        st.metric(sname, f"${price_c:.1f}", delta=f"{((price_c/target_cost)-1)*100:.1f}%")
        if adr_p > SYSTEM_CONFIG["ADR_THRESHOLD"]: st.warning("🔴 過熱觀望")
        elif price_c > target_cost: st.success("🟢 守穩執行")
        else: st.error("🔴 破線取消")

# (4) 全標的一覽矩陣
with st.expander(f"📊 全標的戰略矩陣 (14 檔)", expanded=False):
    all_targets = []
    for eng, stocks in SYSTEM_CONFIG["STOCKS"].items():
        for n, sid in stocks.items(): all_targets.append((eng, n, sid))
    def fetch_row(item):
        eng, n, sid = item
        df_m, fc, ic, rs = commander.get_strategic_data(sid)
        price = commander.get_realtime_status(sid)
        c = ic if "高息" in eng else fc
        return {"引擎": eng[0:3], "標的名稱 (股號)": n, "現價": f"${price:.1f}", "法人成本": f"${c:.1f}", "狀態": "🟢 守穩" if price > c else "🔴 破線"}
    with ThreadPoolExecutor(max_workers=5) as executor:
        matrix_df = pd.DataFrame(list(executor.map(fetch_row, all_targets)))
    st.table(matrix_df)

st.divider()

# (5) 決策系統 Tab
tab_open, tab_post, tab_adr = st.tabs(["☀️ 09:05 決斷", "📊 15:30 盤後分析", "🌌 22:30 美股觀察"])
df_main, f_m, i_m, rs_m = commander.get_strategic_data(stock_id)
p_main = commander.get_realtime_status(stock_id)
m_cost = i_m if "高息" in c_cat else f_m

with tab_open:
    st.subheader(f"⚔️ {c_name} 指令與建議")
    k1, k2 = st.columns([1, 2])
    with k1:
        st.metric("目前價格", f"${p_main:.2f}", delta=f"${p_main - m_cost:.1f}")
        st.write("狀態：" + ("✅ 守穩執行" if p_main > m_cost else "🛑 破線觀望"))
    with k2:
        budget = st.number_input("今日預算 (NTD)", value=100000, step=10000)
        total_s = int(budget / p_main) if p_main > 0 else 0
        st.info(f"建議：**{total_s // 1000}** 張 又 **{total_s % 1000}** 股")

with tab_post:
    st.subheader(f"📊 {c_name} RS 相對強度與成本圖")
    st.metric("RS 指數", f"{rs_m:.1f}", delta="領先" if rs_m > 0 else "落後")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_main.index[-90:], y=df_main['Close'].iloc[-90:], name="價格"))
    c_series = df_main['Invest_Cost'] if "高息" in c_cat else df_main['Foreign_Cost']
    fig.add_trace(go.Scatter(x=df_main.index[-90:], y=c_series.iloc[-90:], name="法人成本 (VWAP)", line=dict(dash='dot')))
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

with tab_adr:
    st.subheader("🌌 全球市場環境")
    st.metric("TSM ADR 溢價率", f"{adr_p:.2f}%")
    st.metric("即時台幣匯率", f"{fx_now:.2f}")

st.caption(f"系統規格：{SYSTEM_CONFIG['VERSION']} | 以法人成本線 (VWAP) 為核心基準")
