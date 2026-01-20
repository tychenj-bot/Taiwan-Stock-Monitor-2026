import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統環境配置 ---
st.set_page_config(page_title="2026 戰略指揮中心 v8.0", layout="wide")

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
        """09:00 快速掃描戰報"""
        results = []
        for stock_id, name, engine in target_list:
            try:
                info = yf.Ticker(f"{stock_id}.TW").fast_info
                real_open = info.open if info.open else info.last_price
                
                # 使用 20MA 作為快速戰報防線
                df = yf.Ticker(f"{stock_id}.TW").history(period="100d")
                ma20 = df['Close'].rolling(20).mean().iloc[-1]
                
                diff = (real_open / ma20 - 1) * 100
                action = "🟢 買進 (守穩)" if real_open > ma20 else "🔴 觀望 (破線)"

                results.append({
                    "引擎": engine,
                    "標的": name,
                    "今日開盤": f"${real_open:.2f}",
                    "防守線 (20MA)": f"${ma20:.1f}",
                    "開盤狀態": "守穩" if real_open > ma20 else "破線",
                    "戰略指令": action
                })
            except:
                pass
        return pd.DataFrame(results)

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id, days=150):
        """深度分析數據 (保留所有監控指標)"""
        ticker_yf = f"{stock_id}.TW"
        df = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df.empty: return pd.DataFrame(), 0, 0, "無數據", 0, 0, 0, 0
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]

        # 技術面: KD & 量比
        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        vol_ma20 = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / vol_ma20

        # 基本面: 殖利率
        try:
            divs = yf.Ticker(ticker_yf).dividends
            if divs.index.tz is not None: divs.index = divs.index.tz_localize(None)
            est_yield = (divs[divs.index > (pd.Timestamp.now() - pd.DateOffset(months=12))].sum() / df['Close'].iloc[-1]) * 100
        except: est_yield = 0

        # 強度面: RS Index
        mkt = yf.Ticker("0050.TW").history(period=f"{days}d")
        mkt.index = mkt.index.tz_localize(None).normalize()
        df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100

        # 籌碼面: FinMind
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
        
        # 連續動向
        net_list = df['foreign_net'].tolist() # 預設看外資
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
        
        return df, consecutive, est_yield, "主力成本線", df['K'].iloc[-1], df['Concentration'].iloc[-1], vol_ma20.iloc[-1]

# --- 3. UI 介面 ---
monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# (1) 頂部：TSM ADR 天氣預報
adr_p, adr_v = monitor.get_global_tsm_signal()
st.metric("🌍 TSM ADR 溢價率 (國際風向)", f"{adr_p:.2f}%", 
          delta="過熱禁止追價" if adr_p > 5 else ("錯殺黃金買點" if adr_p < -2 else "盤向正常"),
          delta_color="inverse" if adr_p > 5 else ("off" if adr_p < -2 else "normal"))

if adr_p > 5: st.warning("⚠️ ADR 過熱，今日台股易開高走低，請嚴格執行『不追高』策略。")

st.divider()

# (2) 中間：09:00 開盤指揮中心 (直覺化表格)
st.markdown("### ☀️ 09:00 開盤三引擎決策")
if st.button("🔄 刷新開盤即時建議", type="primary"): st.cache_data.clear()

leaders = [("00991A", "復華未來50 (主動)", "🔥 成長引擎"), 
           ("0050", "元大台灣50 (市值)", "🛡️ 市值引擎"), 
           ("00878", "國泰永續高股息", "💰 高息引擎")]

df_brief = monitor.get_morning_brief(leaders)
if not df_brief.empty:
    if adr_p > 5: df_brief["戰略指令"] = "🔴 觀望 (ADR過熱)"
    st.table(df_brief.style.map(lambda x: 'color: green' if '買進' in str(x) else ('color: red' if '觀望' in str(x) else ''), subset=['戰略指令']))
else: st.info("等待開盤數據中...")

st.divider()

# (3) 底部：深度監控項目 (保留所有原有功能)
st.markdown("### 🔍 詳細指標分析 (盤後與深度校正)")
targets = {
    "🔥 引擎一：成長進攻": {"台積電 (2330)": "2330", "復華未來50 (00991A)": "00991A", "統一主動 (00981A)": "00981A"},
    "🛡️ 引擎二：市值防禦": {"元大台灣50 (0050)": "0050", "富邦台50 (006208)": "006208"},
    "💰 引擎三：穩健領息": {"群益台灣精選高息 (00919)": "00919", "復華台灣科技優息 (00929)": "00929"}
}
c_cat = st.sidebar.selectbox("引擎分類", list(targets.keys()))
c_name = st.sidebar.selectbox("監控標的", list(targets[c_cat].keys()))
stock_id = targets[c_cat][c_name]

df, con_days, yld, src_label, k_val, conc, avg_v = monitor.get_strategic_data(stock_id)

if not df.empty:
    # 這裡顯示您原本所有的監控項目
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("RS 強度", f"{df['RS_Index'].iloc[-1]:.2f}")
    k2.metric("KD 位階", f"{k_val:.0f}")
    k3.metric("籌碼集中度", f"{conc:.2f}%")
    k4.metric("主力動向", f"{'連買' if con_days>0 else '連賣'} {abs(con_days)}天")

    # 圖表：保留成本線與股價
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="股價"))
    cost_line = df['Invest_Cost'] if "高息" in c_cat else df['Foreign_Cost']
    fig.add_trace(go.Scatter(x=df.index[-60:], y=cost_line.iloc[-60:], name="主力成本", line=dict(dash='dot')))
    st.plotly_chart(fig, use_container_width=True)

# 側邊欄保留 SOP 與 季度策略
st.sidebar.markdown("---")
with st.sidebar.expander("📖 2026 季度戰略回顧"):
    st.write("Q1: 成長引擎 (00991A) 衝刺 | Q2: 高息引擎避險")
