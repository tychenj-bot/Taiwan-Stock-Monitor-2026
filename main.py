import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統環境配置 ---
st.set_page_config(page_title="2026 戰略指揮中心 v10.2", layout="wide")

if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN，請檢查 Secrets 設定。")
    st.stop()
else:
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]

# --- 2. 核心運算引擎 ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        self.api = DataLoader()
        try:
            if hasattr(self.api, 'login'): self.api.login(token=token.strip())
            else: self.api.token = token.strip()
        except: pass

    @st.cache_data(ttl=300)
    def get_global_weather(_self):
        """置頂區：氣候指標 (ADR + SOX)"""
        try:
            tsm_adr = yf.Ticker("TSM").history(period="5d")
            sox = yf.Ticker("^SOX").history(period="5d")
            tsm_tw = yf.Ticker("2330.TW").history(period="5d")
            adr_c = tsm_adr['Close'].iloc[-1]
            sox_p = ((sox['Close'].iloc[-1] / sox['Close'].iloc[-2]) - 1) * 100
            tw_c = tsm_tw['Close'].iloc[-1]
            # 溢價率計算：((ADR股價 * 匯率 / 5) / 台股股價 - 1) * 100
            premium = (((adr_c * 32.5) / 5) / tw_c - 1) * 100
            return premium, adr_c, sox_p
        except: return 0, 0, 0

    def get_strategic_matrix(self, targets_dict, adr_premium):
        """置頂區：全標的智慧戰略矩陣 (展開版)"""
        results = []
        for engine_name, stocks in targets_dict.items():
            for name, stock_id in stocks.items():
                try:
                    # 抓取數據以取得法人成本
                    df, _, _, _, _, _ = self.get_strategic_data(stock_id)
                    latest = df.iloc[-1]
                    info = yf.Ticker(f"{stock_id}.TW").fast_info
                    real_open = info.open if info.open else info.last_price
                    
                    # 統一判定基準：法人成本
                    cost_line = latest['Invest_Cost'] if "高息" in engine_name else latest['Foreign_Cost']
                    status = "守穩" if real_open > cost_line else "破線"
                    
                    # 智慧指令邏輯 (結合氣候與個股狀態)
                    if adr_premium > 5:
                        action = "🔴 觀望 (ADR過熱)"
                    elif real_open > cost_line:
                        action = "🟢 積極進攻" if adr_premium > 0 else "🟡 穩健佈局"
                    else:
                        action = "💎 逢低校正" if adr_premium < -2 else "🔴 取消交易"

                    results.append({
                        "分類": engine_name[0:3], # 取圖示+名稱前二字
                        "標的": name,
                        "今日開盤": f"${real_open:.1f}",
                        "法人成本": f"${cost_line:.1f}",
                        "狀態": status,
                        "最終指令": action
                    })
                except: pass
        return pd.DataFrame(results)

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id, days=150):
        """深度指標運算"""
        ticker_yf = f"{stock_id}.TW"
        df = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df.empty: return pd.DataFrame(), 0, 0, 0, 0, 0
        df.index = df.index.tz_localize(None).normalize()

        mkt = yf.Ticker("0050.TW").history(period=f"{days}d")
        mkt.index = mkt.index.tz_localize(None).normalize()
        df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            for name in ['Foreign', 'Investment']:
                sub = df_chip[df_chip['name'].str.contains(name, case=False)].copy()
                sub['date'] = pd.to_datetime(sub['date'])
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
        df['Concentration'] = (df['foreign_net'] + df['investment_net']) / df['Volume'] * 100
        low_min, high_max = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
        df['K'] = ((df['Close'] - low_min) / (high_max - low_min) * 100).ewm(com=2).mean()
        
        net_list = df['foreign_net'].tolist()
        con = 0
        if net_list:
            if net_list[-1] > 0:
                for v in reversed(net_list): 
                    if v > 0: con += 1
                    else: break
            else:
                for v in reversed(net_list): 
                    if v < 0: con -= 1
                    else: break
        
        try:
            divs = yf.Ticker(ticker_yf).dividends
            if divs.index.tz is not None: divs.index = divs.index.tz_localize(None)
            yld = (divs[divs.index > (pd.Timestamp.now() - pd.DateOffset(months=12))].sum() / df['Close'].iloc[-1]) * 100
        except: yld = 0
        return df, con, yld, df['K'].iloc[-1], df['Concentration'].iloc[-1], df['Volume'].iloc[-1]

    def get_realtime_open(self, stock_id):
        try:
            info = yf.Ticker(f"{stock_id}.TW").fast_info
            return (info.open if info.open else info.last_price), info.last_price, info.last_volume
        except: return 0, 0, 0

# --- 3. UI 介面實作 ---
monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# A. 側邊欄：所有標的設定
st.sidebar.header("🦅 2026 戰略控制台")
targets = {
    "🔥 引擎一：成長進攻": {"台積電 (2330)": "2330", "復華未來50 (00991A)": "00991A", "統一主動 (00981A)": "00981A", "群益精選 (00982A)": "00982A", "復華好收益 (00980A)": "00980A"},
    "🛡️ 引擎二：市值防禦": {"元大台灣50 (0050)": "0050", "富邦台50 (006208)": "006208", "國泰領袖50 (00922)": "00922"},
    "💰 引擎三：穩健領息": {"元大高股息 (0056)": "0056", "國泰永續高股息 (00878)": "00878", "群益台灣精選高息 (00919)": "00919", "復華台灣科技優息 (00929)": "00929"}
}
c_cat = st.sidebar.selectbox("引擎分類", list(targets.keys()))
c_name = st.sidebar.selectbox("監控標的", list(targets[c_cat].keys()))
stock_id = targets[c_cat][c_name]

st.sidebar.divider()
with st.sidebar.expander("📖 每日操作 SOP", expanded=True):
    st.markdown("""
    **1️⃣ 15:30 (選股)**: 主力連買 >= 3 天 | RS > 0
    **2️⃣ 22:30 (定調)**: ADR > 5% 不追 | < -2% 買
    **3️⃣ 09:05 (執行)**: 開盤 > **法人成本** 買進
    """)
with st.sidebar.expander("🗺️ 2026 季度佈局", expanded=False):
    st.info("Q1: 00991A / 00981A 競速期")

# B. 置頂區：智慧戰略矩陣 (全展開)
adr_p, adr_v, sox_p = monitor.get_global_weather()
c1, c2, c3 = st.columns(3)
c1.metric("🌍 TSM ADR 溢價", f"{adr_p:.2f}%", delta="過熱" if adr_p > 5 else "正常")
c2.metric("💻 費半指數 (SOX)", f"{sox_p:.2f}%")
c3.metric("💰 匯率環境", "台幣升值趨勢", delta="外資流入")

st.markdown("### 🦅 09:00 指揮中心：智慧戰略矩陣 (全標的一覽)")
df_matrix = monitor.get_strategic_matrix(targets, adr_p)
st.table(df_matrix.style.map(lambda x: 'color: #00ff00; font-weight: bold' if '進攻' in str(x) or '佈局' in str(x) else ('color: #ff4b4b; font-weight: bold' if '觀望' in str(x) or '取消' in str(x) else ''), subset=['最終指令']))

st.divider()

# C. 深度分頁：保持原有深度分析功能
tab_open, tab_post, tab_adr = st.tabs(["☀️ 09:05 開盤執行", "📊 15:30 盤後分析", "🌌 22:30 美股觀察"])
df, con_days, yld, k_val, conc, _ = monitor.get_strategic_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    is_high_div = "高息" in c_cat or "穩健領息" in c_cat
    main_cost = latest['Invest_Cost'] if is_high_div else latest['Foreign_Cost']
    cost_label = "投信成本" if is_high_div else "外資成本"
    real_open, real_last, _ = monitor.get_realtime_open(stock_id)

    with tab_open:
        st.subheader(f"⚔️ {c_name} 指令執行細節")
        m1, m2, m3 = st.columns(3)
        m1.metric("今日開盤", f"${real_open:.2f}")
        m2.metric(f"法人成本 ({cost_label})", f"${main_cost:.1f}")
        m3.metric("狀態", "守穩" if real_open > main_cost else "破線")
        if real_open > main_cost: st.success(f"✅ 開盤守穩法人成本，參考上方矩陣指令執行。")
        else: st.error("🛑 跌破關鍵成本防線，取消交易。")

    with tab_post:
        st.subheader(f"📊 {c_name} 深度指標")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RS 相對強度", f"{latest['RS_Index']:.2f}", delta="強勢" if latest['RS_Index'] > 0 else "弱勢")
        c2.metric("主力動向", f"{con_days} 天")
        c3.metric("籌碼集中度", f"{conc:.2f}%")
        c4.metric("乖離率", f"{(real_last/main_cost-1)*100:.2f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="股價", line=dict(width=3)))
        cost_series = df['Invest_Cost'] if is_high_div else df['Foreign_Cost']
        fig.add_trace(go.Scatter(x=df.index[-60:], y=cost_series.iloc[-60:], name=cost_label, line=dict(dash='dot', color='orange')))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab_adr:
        st.subheader("🌌 全球連動與位階校正")
        k1, k2, k3 = st.columns(3)
        k1.metric("ADR 溢價", f"{adr_p:.2f}%")
        k2.metric("KD K值", f"{k_val:.1f}")
        k3.metric("殖利率 (預估)", f"{yld:.2f}%")

st.caption("v10.2 智慧戰略矩陣全展開版：統一法人成本判定邏輯。")
