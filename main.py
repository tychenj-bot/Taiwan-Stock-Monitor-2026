import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統環境配置 ---
st.set_page_config(page_title="2026 ADR 戰情系統 v6.2", layout="wide")

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

    @st.cache_data(ttl=600)
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

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id, days=150):
        # A. 價格數據 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df = yf.Ticker(ticker_yf).history(period=f"{days}d")
        
        if df.empty: return pd.DataFrame(), 0, 0, "無數據"
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]

        # 估算殖利率
        try:
            divs = yf.Ticker(ticker_yf).dividends
            if divs.index.tz is not None: divs.index = divs.index.tz_localize(None)
            one_year_ago = pd.Timestamp.now() - pd.DateOffset(months=12)
            est_yield = (divs[divs.index > one_year_ago].sum() / df['Close'].iloc[-1]) * 100
        except:
            est_yield = 0

        # RS 相對強度
        mkt = yf.Ticker("0050.TW").history(period=f"{days}d")
        mkt.index = mkt.index.tz_localize(None).normalize()
        df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100

        # B. 籌碼數據 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            for name in ['Foreign', 'Investment']:
                sub = df_chip[df_chip['name'].str.contains(name, case=False)].copy()
                sub['date'] = pd.to_datetime(sub['date'])
                sub = sub.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
                df[f'{name.lower()}_net'] = sub['buy'] - sub['sell']
        except:
            df['foreign_net'] = 0
            df['investment_net'] = 0

        df = df.fillna(0)
        
        # C. 智慧成本線演算法 (VWAP + Fallback)
        def calculate_vwap_safe(net_buy_col):
            costs = []
            last_valid = np.nan
            has_data = False
            
            for i in range(len(df)):
                win = df.iloc[max(0, i-19) : i+1]
                buys = win[win[net_buy_col] > 0]
                
                if not buys.empty:
                    val = (buys['Close'] * buys[net_buy_col]).sum() / buys[net_buy_col].sum()
                    last_valid = val
                    has_data = True
                
                costs.append(last_valid)
            return pd.Series(costs, index=df.index).ffill().bfill(), has_data

        # 計算外資與投信成本
        f_cost_series, f_has_data = calculate_vwap_safe('foreign_net')
        i_cost_series, i_has_data = calculate_vwap_safe('investment_net')
        
        # 存入 DataFrame
        df['Foreign_Cost'] = f_cost_series
        df['Invest_Cost'] = i_cost_series
        
        # 決定最終使用的成本線 (Cost Source)
        # 邏輯：如果投信有數據就用投信，否則用外資，再沒有就用季線 (SMA60)
        if i_has_data:
            used_source = "投信成本"
        elif f_has_data:
            used_source = "外資成本 (備援)"
            df['Invest_Cost'] = df['Foreign_Cost'] # 覆蓋以便統一調用
        else:
            used_source = "季線 (SMA60)"
            df['Invest_Cost'] = df['Close'].rolling(60).mean() # 最終防線

        # 連買天數 (以外資為主，若為高息股可看投信)
        target_net = df['investment_net'] if 'Invest' in used_source and not '備援' in used_source else df['foreign_net']
        net_list = target_net.tolist()
        consecutive = 0
        for val in reversed(net_list):
            if val > 0: consecutive += 1
            elif val < 0: break
            
        return df, consecutive, est_yield, used_source

# --- 3. UI 介面 ---
st.title("🦅 2026 ADR 戰情系統 v6.2")

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# ADR 儀表板
st.markdown("### 🌎 全球戰略風向 (TSM ADR)")
adr_premium, adr_price = monitor.get_global_tsm_signal()
col_m, col_i = st.columns([1, 2])
with col_m:
    d_col = "inverse" if adr_premium > 5 else ("off" if adr_premium < 0 else "normal")
    st.metric("TSM ADR 溢價率", f"{adr_premium:.2f}%", f"美股收盤 ${adr_price:.2f}", delta_color=d_col)
with col_i:
    if adr_premium > 5: st.warning("🔥 **過熱**：嚴禁追價，留意開高走低。")
    elif adr_premium < -2: st.error("💎 **校正**：負溢價錯殺，留意開低買點。")
    else: st.info("🟢 **正常**：回歸個股籌碼判斷。")

st.divider()

# 標的選擇
st.markdown("### 🔍 標的驗證 (ETF 籌碼優化版)")
targets = {
    "🔥 引擎一：成長進攻": {
        "台積電 (2330)": "2330",
        "中信上游半導體 (00991A)": "00991A",
        "統一主動 (00981A)": "00981A", 
        "群益精選 (00982A)": "00982A",
        "復華台灣好收益 (00980A)": "00980A"
    },
    "🛡️ 引擎二：市值防禦": {
        "元大台灣50 (0050)": "0050", 
        "富邦台50 (006208)": "006208",
        "國泰領袖50 (00922)": "00922"
    },
    "💰 引擎三：穩健領息": {
        "元大高股息 (0056)": "0056", 
        "國泰永續高股息 (00878)": "00878", 
        "群益台灣精選高息 (00919)": "00919",
        "復華台灣科技優息 (00929)": "00929"
    }
}

c1, c2 = st.columns(2)
with c1: cat = st.selectbox("引擎分類", list(targets.keys()))
with c2: name = st.selectbox("監控標的", list(targets[cat].keys()))
stock_id = targets[cat][name]

df, con_buy, yield_rate, source_name = monitor.get_strategic_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    
    # 決定顯示哪條線
    is_high_div = "高股息" in cat or "穩健領息" in cat
    # 若為高息股，優先用計算出來的 Invest_Cost (可能已經備援切換過)
    # 若為成長股，優先用 Foreign_Cost
    if is_high_div:
        main_cost = latest['Invest_Cost']
        cost_label = source_name 
    else:
        main_cost = latest['Foreign_Cost']
        cost_label = "外資成本"

    bias = (latest['Close'] / main_cost - 1) * 100
    
    # 數據看板
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"參考：{cost_label}", f"${main_cost:.1f}")
    k2.metric("參考：籌碼乖離", f"{bias:.2f}%")
    k3.metric("參考：RS 強度", f"{latest['RS_Index']:.2f}")
    k4.metric("參考：殖利率", f"{yield_rate:.2f}%")

    # 戰略建議
    st.markdown("#### 📝 戰略建議")
    if stock_id in ["0050", "006208"]:
        st.info("ℹ️ **基準標的**：大盤觀測基準。")
    elif adr_premium < -1 and con_buy > 0:
        st.success(f"🎯 **校正機會**：ADR 跌但籌碼支撐，留意買點。")
    elif bias < 2 and latest['Close'] > main_cost:
        st.success(f"✅ **順勢佈局**：股價守穩 {cost_label}。")
    else:
        st.warning("⚠️ **觀望/警戒**：無明確訊號。")

    # 圖表
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="股價", line=dict(color='#1f77b4', width=3)))
    
    line_col = '#ff7f0e' if is_high_div else '#d62728'
    cost_series = df['Invest_Cost'] if is_high_div else df['Foreign_Cost']
    
    fig.add_trace(go.Scatter(x=df.index[-60:], y=cost_series.iloc[-60:], name=cost_label, line=dict(color=line_col, dash='dot')))
    
    fig.update_layout(template="plotly_dark", height=350, margin=dict(t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    if "SMA" in cost_label:
        st.caption("註：因法人籌碼數據不足，系統已自動切換為「技術面均線」作為防守參考。")

st.caption("v6.2 修正：針對 00919/00929 導入智慧備援機制 (投信 -> 外資 -> 季線)，確保防守線不中斷。")
