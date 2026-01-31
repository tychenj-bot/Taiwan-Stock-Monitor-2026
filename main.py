import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# C. 規格化配置 (MODULAR CONFIG) - 標的擴充
# ==========================================
SYSTEM_CONFIG = {
    "VERSION": "v12.3 標項擴充版",
    "ADR_THRESHOLD": 5.0,  # ADR 溢價警戒線 (%)
    "MA_PERIOD": 20,
    "CHIP_DAYS": 150,
    "STOCKS": {
        "🔥 成長": {
            "台積電": "2330", 
            "復華50": "00991A", 
            "統一主動": "00981A", 
            "群益精選": "00982A", 
            "復華好收益": "00980A"
        },
        "🛡️ 市值": {
            "元大50": "0050", 
            "富邦50": "006208", 
            "國泰50": "00922",
            "台泥": "1101"  # <-- 新增標的
        },
        "💰 高息": {
            "元大高息": "0056", 
            "國泰高息": "00878", 
            "群益高息": "00919", 
            "復華優息": "00929"
        }
    }
}

# --- 1. 系統環境配置 ---
st.set_page_config(page_title=f"戰略指揮中心 {SYSTEM_CONFIG['VERSION']}", layout="wide")

if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN，請檢查 Secrets 設定。")
    st.stop()
FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]

# --- 2. 核心運算引擎 (穩定防禦版) ---
class TaiwanStockCommander2026:
    def __init__(self, token):
        self.api = DataLoader()
        try:
            if hasattr(self.api, 'login'): self.api.login(token=token.strip())
            else: self.api.token = token.strip()
        except: pass

    @st.cache_data(ttl=300)
    def get_global_weather(_self):
        """抓取全球氣候指標 (含即時匯率)"""
        try:
            tsm_adr = yf.Ticker("TSM").history(period="2d")
            sox = yf.Ticker("^SOX").history(period="2d")
            tsm_tw = yf.Ticker("2330.TW").history(period="2d")
            twd = yf.Ticker("TWD=X").history(period="2d") 
            
            fx = twd['Close'].iloc[-1]
            adr_c = tsm_adr['Close'].iloc[-1]
            sox_p = ((sox['Close'].iloc[-1] / sox['Close'].iloc[-2]) - 1) * 100
            tw_c = tsm_tw['Close'].iloc[-1]
            premium = (((adr_c * fx) / 5) / tw_c - 1) * 100
            return premium, fx, sox_p
        except: return 0, 32.5, 0

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id):
        """深度指標運算 (數據對齊修復)"""
        days = SYSTEM_CONFIG["CHIP_DAYS"]
        df = yf.Ticker(f"{stock_id}.TW").history(period=f"{days}d")
        if df.empty: return pd.DataFrame(), 0, 0, 0
        df.index = df.index.tz_localize(None).normalize()

        # RS 指數對齊
        try:
            mkt = yf.Ticker("0050.TW").history(period=f"{days}d")
            mkt.index = mkt.index.tz_localize(None).normalize()
            df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100
        except:
            df['RS_Index'] = 0

        # 籌碼面邏輯
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            for name in ['Foreign', 'Investment']:
                sub = df_chip[df_chip['name'].str.contains(name, case=False)].copy()
                sub['date'] = pd.to_datetime(sub['date']).dt.normalize()
                sub = sub.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
                df[f'{name.lower()}_net'] = sub['buy'] - sub['sell']
        except: 
            df['foreign_net'] = df['investment_net'] = 0

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

# 側邊欄：強制刷新與標的選擇
if st.sidebar.button("🔄 核心數據強制刷新"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
c_cat = st.sidebar.selectbox("引擎分類", list(SYSTEM_CONFIG["STOCKS"].keys()))
c_name = st.sidebar.selectbox("監控標的", list(SYSTEM_CONFIG["STOCKS"][c_cat].keys()))
stock_id = SYSTEM_CONFIG["STOCKS"][c_cat][c_name]

# 置頂看板
adr_p, fx_now, sox_p = commander.get_global_weather()
st.markdown(f"### 🌍 全球氣候 (ADR: **{adr_p:.1f}%** | USD/TWD: **{fx_now:.2f}**)")

# 三引擎視覺看板
st.divider()
core_list = [("🔥 成長", "00991A", "復華未來50"), ("🛡️ 市值", "0050", "元大台灣50"), ("💰 高息", "00878", "國泰高息")]
cols = st.columns(3)
for i, (tag, sid, sname) in enumerate(core_list):
    with cols[i]:
        df_c, fc, ic, _ = commander.get_strategic_data(sid)
        price_c = commander.get_realtime_status(sid)
        target_cost = ic if "高息" in tag else fc
        st.metric(sname, f"${price_c:.1f}", delta=f"{((price_c/target_cost)-1)*100:.1f}%")
        if adr_p > SYSTEM_CONFIG["ADR_THRESHOLD"]: st.warning("🔴 過熱禁追")
        elif price_c > target_cost: st.success("🟢 守穩進攻")
        else: st.error("🔴 破線觀望")

# 全標的一覽矩陣 (含台泥)
with st.expander("📊 全標的一覽 (包含台泥 1101)", expanded=False):
    all_targets = []
    for eng, stocks in SYSTEM_CONFIG["STOCKS"].items():
        for n, sid in stocks.items(): all_targets.append((eng, n, sid))
    
    def fetch_row(item):
        eng, n, sid = item
        df_m, fc, ic, rs = commander.get_strategic_data(sid)
        price = commander.get_realtime_status(sid)
        c = ic if "高息" in eng else fc
        return {"引擎": eng[0:3], "標的": n, "現價": f"${price:.1f}", "法人成本": f"${c:.1f}", "狀態": "🟢 守穩" if price > c else "🔴 破線"}

    with ThreadPoolExecutor(max_workers=5) as executor:
        matrix_df = pd.DataFrame(list(executor.map(fetch_row, all_targets)))
    st.table(matrix_df)

st.divider()

# 分頁功能
tab_open, tab_post, tab_adr = st.tabs(["☀️ 09:05 決斷", "📊 15:30 盤後", "🌌 22:30 美股"])
df_main, f_m, i_m, rs_m = commander.get_strategic_data(stock_id)
p_main = commander.get_realtime_status(stock_id)
m_cost = i_m if "高息" in c_cat else f_m

with tab_open:
    st.subheader(f"⚔️ {c_name} 指令與建議")
    k1, k2 = st.columns([1, 2])
    with k1:
        st.metric("現價", f"${p_main:.2f}", delta=f"${p_main - m_cost:.1f}")
        st.write("狀態：" + ("✅ 守穩" if p_main > m_cost else "🛑 破線"))
    with k2:
        budget = st.number_input("今日預算 (NTD)", value=100000, step=10000)
        total_s = int(budget / p_main) if p_main > 0 else 0
        st.info(f"建議：**{total_s // 1000}** 張又 **{total_s % 1000}** 股")

with tab_post:
    st.subheader(f"📊 {c_name} RS 強度與成本圖")
    st.metric("RS 指數", f"{rs_m:.1f}", delta="強勢" if rs_m > 0 else "弱勢")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_main.index[-60:], y=df_main['Close'].iloc[-60:], name="價格"))
    c_series = df_main['Invest_Cost'] if "高息" in c_cat else df_main['Foreign_Cost']
    fig.add_trace(go.Scatter(x=df_main.index[-60:], y=c_series.iloc[-60:], name="法人成本線", line=dict(dash='dot')))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab_adr:
    st.subheader("🌌 全球連動資訊")
    st.metric("ADR 溢價率", f"{adr_p:.2f}%")
    st.metric("即時匯率 (USD/TWD)", f"{fx_now:.2f}")

st.caption(f"系統版本：{SYSTEM_CONFIG['VERSION']} | 核心判定：法人成本線")
