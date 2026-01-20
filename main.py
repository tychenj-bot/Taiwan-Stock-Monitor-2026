import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統環境配置 ---
st.set_page_config(page_title="2026 戰略指揮中心 v9.3", layout="wide")

# 安全檢查：Token
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
    def get_global_tsm_signal(_self):
        """全域 TSM ADR 訊號"""
        try:
            tsm_adr = yf.Ticker("TSM").history(period="5d")
            tsm_tw = yf.Ticker("2330.TW").history(period="5d")
            tsm_adr.index = tsm_adr.index.tz_localize(None).normalize()
            tsm_tw.index = tsm_tw.index.tz_localize(None).normalize()
            adr_close = tsm_adr['Close'].iloc[-1]
            tw_close = tsm_tw['Close'].iloc[-1]
            fx_rate = 32.5 
            implied_price = (adr_close * fx_rate) / 5
            premium = ((implied_price / tw_close) - 1) * 100
            return premium, adr_close
        except:
            return 0, 0

    def get_morning_brief(self, target_list):
        """09:00 置頂戰報表格邏輯"""
        results = []
        for stock_id, name, engine in target_list:
            try:
                info = yf.Ticker(f"{stock_id}.TW").fast_info
                real_open = info.open if info.open else info.last_price
                df = yf.Ticker(f"{stock_id}.TW").history(period="100d")
                ma20 = df['Close'].rolling(20).mean().iloc[-1]
                action = "🟢 買進" if real_open > ma20 else "🔴 觀望"
                results.append({
                    "引擎": engine,
                    "標的": name,
                    "今日開盤": f"${real_open:.1f}",
                    "防守月線": f"${ma20:.1f}",
                    "狀態": "守穩" if real_open > ma20 else "破線",
                    "戰略指令": action
                })
            except: pass
        return pd.DataFrame(results)

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id, days=150):
        """詳細指標數據 (RS, 集中度, VWAP, KD, 量比)"""
        ticker_yf = f"{stock_id}.TW"
        df = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df.empty: return pd.DataFrame(), 0, 0, 0, 0, 0
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]

        # 技術指標
        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        vol_ma20 = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / vol_ma20

        # 殖利率
        try:
            divs = yf.Ticker(ticker_yf).dividends
            if divs.index.tz is not None: divs.index = divs.index.tz_localize(None)
            est_yield = (divs[divs.index > (pd.Timestamp.now() - pd.DateOffset(months=12))].sum() / df['Close'].iloc[-1]) * 100
        except: est_yield = 0

        # RS 強度 (vs 0050)
        mkt = yf.Ticker("0050.TW").history(period=f"{days}d")
        mkt.index = mkt.index.tz_localize(None).normalize()
        df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100

        # 籌碼數據
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            for name in ['Foreign', 'Investment']:
                sub = df_chip[df_chip['name'].str.contains(name, case=False)].copy()
                sub['date'] = pd.to_datetime(sub['date'])
                sub = sub.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
                df[f'{name.lower()}_net'] = sub['buy'] - sub['sell']
        except:
            df['foreign_net'] = df['investment_net'] = 0

        df = df.fillna(0)
        df['Concentration'] = (df['foreign_net'] + df['investment_net']) / df['Volume'] * 100

        # VWAP 成本線
        def calc_vwap(net_col):
            costs = []
            last = np.nan
            for i in range(len(df)):
                win = df.iloc[max(0, i-19) : i+1]
                buys = win[win[net_col] > 0]
                if not buys.empty: last = (buys['Close'] * buys[net_col]).sum() / buys[net_col].sum()
                costs.append(last)
            return pd.Series(costs, index=df.index).ffill().bfill()

        df['Foreign_Cost'] = calc_vwap('foreign_net')
        df['Invest_Cost'] = calc_vwap('investment_net')
        
        # 主力動向天數
        net_list = df['foreign_net'].tolist()
        consecutive = 0
        if net_list:
            if net_list[-1] > 0:
                for v in reversed(net_list): 
                    if v > 0: consecutive += 1
                    else: break
            else:
                for v in reversed(net_list): 
                    if v < 0: consecutive -= 1
                    else: break
        return df, consecutive, est_yield, df['K'].iloc[-1], df['Concentration'].iloc[-1], vol_ma20.iloc[-1]

    def get_realtime_open(self, stock_id):
        try:
            info = yf.Ticker(f"{stock_id}.TW").fast_info
            return (info.open if info.open else info.last_price), info.last_price, info.last_volume
        except: return 0, 0, 0

# --- 3. UI 介面 ---
monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# (1) 側邊欄：完整 SOP 與 策略
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
    **1️⃣ 15:30 (選股)**
    - 主力連買 >= 3 天 | RS 指標 > 0
    **2️⃣ 22:30 (定調)**
    - ADR > 5% 不追 | ADR < -2% 買
    **3️⃣ 09:05 (執行)**
    - 開盤 > 成本線：買進
    """)
with st.sidebar.expander("🗺️ 2026 季度佈局", expanded=False):
    st.info("Q1 核心：00991A/00981A 競速 Alpha")
    st.markdown("Q2: 轉進高息避險 | Q3: 加碼主動型攻擊 | Q4: 回防 0050")

# (2) 置頂區：ADR 天氣 + 指揮中心表格
adr_p, adr_v = monitor.get_global_tsm_signal()
st.metric("🌍 TSM ADR 溢價率 (全域風向)", f"{adr_p:.2f}%", 
          delta="過熱不追" if adr_p > 5 else ("錯殺機會" if adr_p < -2 else "盤向正常"),
          delta_color="inverse" if adr_p > 5 else ("off" if adr_p < -2 else "normal"))

st.markdown("### ☀️ 09:00 指揮中心戰報")
leaders = [("00991A", "復華未來50 (主動)", "🔥 成長"), ("0050", "元大台灣50 (市值)", "🛡️ 市值"), ("00878", "國泰永續高股息", "💰 高息")]
df_brief = monitor.get_morning_brief(leaders)
if not df_brief.empty:
    if adr_p > 5: df_brief["戰略指令"] = "🔴 觀望 (ADR過熱)"
    st.table(df_brief.style.map(lambda x: 'color: green' if '買進' in str(x) else ('color: red' if '觀望' in str(x) else ''), subset=['戰略指令']))

st.divider()

# (3) 分頁區：三大時段深度分析
tab_open, tab_post, tab_adr = st.tabs(["☀️ 09:05 開盤執行", "📊 15:30 盤後分析", "🌌 22:30 美股觀察"])
df, con_days, yld, k_val, conc, avg_v = monitor.get_strategic_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    is_high_div = "高息" in c_cat or "穩健領息" in c_cat
    
    # --- 修正點：分開定義「單點數值」與「繪圖序列」 ---
    cost_series = df['Invest_Cost'] if is_high_div else df['Foreign_Cost']
    main_cost_val = latest['Invest_Cost'] if is_high_div else latest['Foreign_Cost']
    cost_label = "投信成本" if is_high_div else "外資成本"
    
    real_open, real_last, real_vol = monitor.get_realtime_open(stock_id)

    with tab_open:
        st.subheader(f"⚔️ {c_name} 開盤指令決斷")
        m1, m2, m3 = st.columns(3)
        m1.metric("今日開盤", f"${real_open:.2f}")
        m2.metric("主力防線", f"${main_cost_val:.1f}")
        m3.metric("狀態", "守穩" if real_open > main_cost_val else "破線", 
                  delta_color="normal" if real_open > main_cost_val else "inverse")
        if real_open > main_cost_val: st.success(f"✅ 符合進場條件，守穩 {cost_label}。")
        else: st.error(f"🛑 跌破 {cost_label} 防線，取消交易。")

    with tab_post:
        st.subheader(f"📊 {c_name} 深度籌碼與強度")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RS 強度", f"{latest['RS_Index']:.2f}", delta="強勢" if latest['RS_Index']>0 else "弱勢")
        c2.metric("主力連動", f"{con_days}天")
        c3.metric("籌碼集中度", f"{conc:.2f}%")
        c4.metric("乖離率", f"{(real_last/main_cost_val-1)*100:.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="股價", line=dict(width=3)))
        # 使用修正後的序列進行繪圖
        fig.add_trace(go.Scatter(x=df.index[-60:], y=cost_series.iloc[-60:], name=cost_label, line=dict(dash='dot', color='orange')))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=df.index[-90:], y=df['RS_Index'].iloc[-90:], fill='tozeroy', name="RS Index"))
        fig_rs.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_rs, use_container_width=True)

    with tab_adr:
        st.subheader("🌌 全球連動與位階校正")
        k1, k2, k3 = st.columns(3)
        k1.metric("ADR 溢價", f"{adr_p:.2f}%")
        k2.metric("KD 位階", f"{k_val:.1f}")
        k3.metric("預估殖利率", f"{yld:.2f}%")
        st.info("💡 提醒：若 ADR 大跌但籌碼連買，隔日開低即為『校正買點』。")

st.caption("v9.3 終極整合版：置頂戰報 + 深度分頁 + 側欄完整 SOP。")
