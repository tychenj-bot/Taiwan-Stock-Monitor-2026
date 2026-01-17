import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統設定與頁面配置 ---
st.set_page_config(page_title="2026 台股雙核監控系統", layout="wide")

# 從 Streamlit Secrets 讀取 Token
try:
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]
except:
    st.error("請在 Streamlit Cloud 的 Secrets 中設定 FINMIND_TOKEN")
    st.stop()

# --- 2. 核心分析類別 ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        self.api = DataLoader()
        self.api.login_token(token)
        
    @st.cache_data(ttl=3600) # 快取數據一小時，避免重複請求
    def get_full_analysis_data(_self, stock_id, days=60):
        # A. 抓取價格 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df_price = yf.Ticker(ticker_yf).history(period=f"{days}d")
        df_price.index = df_price.index.tz_localize(None).normalize()

        # B. 抓取籌碼 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        df_chip = _self.api.taiwan_stock_institutional_investors(
            data_id=stock_id,
            start_date=start_date
        )
        
        # 過濾外資數據
        df_foreign = df_chip[df_chip['name'] == 'Foreign_Investor'].copy()
        df_foreign['date'] = pd.to_datetime(df_foreign['date'])
        df_foreign = df_foreign.set_index('date')

        # C. 合併並計算成本線
        combined = pd.concat([df_price, df_foreign[['net_buy']]], axis=1).dropna(subset=['Close'])
        
        # 核心公式：外資買進日加權平均
        def get_weighted_cost(window_df):
            buys = window_df[window_df['net_buy'] > 0]
            if buys.empty: return np.nan
            return (buys['Close'] * buys['net_buy']).sum() / buys['net_buy'].sum()

        costs = []
        window = 20
        for i in range(len(combined)):
            if i < window: costs.append(np.nan)
            else:
                win = combined.iloc[i-window+1 : i+1]
                costs.append(get_weighted_cost(win))
        
        combined['Foreign_Cost_Line'] = costs
        combined['Foreign_Cost_Line'] = combined['Foreign_Cost_Line'].ffill()
        return combined

    def get_realtime_signal(self, stock_id):
        ticker = yf.Ticker(f"{stock_id}.TW")
        fast = ticker.fast_info
        last, open_p, prev_c = fast.last_price, fast.open, fast.previous_close
        
        if last > open_p and open_p > prev_c: signal = "🟢 強勢多頭"
        elif last < open_p: signal = "🟡 留意回檔"
        else: signal = "⚪ 震盪整理"
        
        return last, round((open_p/prev_c-1)*100, 2), signal

# --- 3. Streamlit 介面實作 ---
st.title("🚀 2026 台股雙核監控系統")
st.sidebar.header("監控參數")

# 選擇標的
stock_options = {
    "台積電": "2330",
    "元大台灣50": "0050",
    "富邦台50": "006208",
    "國泰領袖50": "00922",
    "統一台股主動": "00981A",
    "群益精選主動": "00982A"
}
target_name = st.sidebar.selectbox("選擇監控標的", list(stock_options.keys()))
target_id = stock_options[target_name]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# A. 即時監控區
st.subheader(f"📡 即時獵手：{target_name} ({target_id})")
last, gap, sig = monitor.get_realtime_signal(target_id)
c1, c2, c3 = st.columns(3)
c1.metric("當前股價", f"${last:.2f}")
c2.metric("開盤漲幅", f"{gap}%")
c3.warning(f"當前燈號：{sig}")

# B. 籌碼深度分析區
st.divider()
st.subheader("📊 外資成本線與乖離分析")
with st.spinner("正在分析籌碼數據..."):
    df = monitor.get_full_analysis_data(target_id)
    latest = df.iloc[-1]
    f_cost = latest['Foreign_Cost_Line']
    bias = (latest['Close'] / f_cost - 1) * 100 if f_cost > 0 else 0

    # 繪製圖表
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="收盤價", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=df.index, y=df['Foreign_Cost_Line'], name="外資成本線", line=dict(color="#d62728", dash='dot')))
    fig.update_layout(title=f"{target_name} 成本防線圖", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"💡 目前股價距離外資 20 日成本乖離率：**{bias:.2f}%** (小於 5% 為法人安全區)")

# C. 2026 四季佈局策略
st.divider()
st.subheader("📅 2026 投資布局指引")
month = datetime.now().month
strategies = {
    "Q1": "✨ 佈局期：台積電 2nm 產能預訂熱絡。資金配置建議：60% 市值型 + 40% 主動型。",
    "Q2": "📉 防禦期：報稅季與電子淡季。觀察外資成本線，若不破則為長線分批買點。",
    "Q3": "🚀 噴發期：AI 伺服器供應鏈進入出貨高峰。提高主動型 ETF 權重至 70% 捕捉超額報酬。",
    "Q4": "💰 收穫期：法人年終作帳。回歸大型權值股，鎖定年度獲利，避開投信結帳賣壓。"
}
curr_q = f"Q{(month-1)//3 + 1}"
st.success(strategies[curr_q])
