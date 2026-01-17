import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統設定與頁面配置 ---
st.set_page_config(page_title="2026 台股雙核監控系統", layout="wide")

# 從 Streamlit Secrets 安全讀取 Token
try:
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]
except Exception:
    st.error("❌ 找不到 FINMIND_TOKEN，請前往 Streamlit Cloud 的 Settings -> Secrets 進行設定。")
    st.stop()

# --- 2. 核心分析類別 ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        # 初始化 FinMind 載入器
        self.api = DataLoader()
        # 修正：FinMind 新版 API 登入指令為 login
        try:
            self.api.login(token=token)
        except Exception as e:
            st.error(f"FinMind 登入失敗：{e}")

    @st.cache_data(ttl=3600) # 快取數據 1 小時，避免頻繁請求 API
    def get_full_analysis_data(_self, stock_id, days=60):
        # A. 抓取價格數據 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df_price = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df_price.empty:
            return pd.DataFrame()
        df_price.index = df_price.index.tz_localize(None).normalize()

        # B. 抓取籌碼數據 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(
                data_id=stock_id,
                start_date=start_date
            )
            # 過濾外資數據 (Foreign_Investor)
            df_foreign = df_chip[df_chip['name'] == 'Foreign_Investor'].copy()
            df_foreign['date'] = pd.to_datetime(df_foreign['date'])
            df_foreign = df_foreign.set_index('date')
        except Exception:
            # 若籌碼抓取失敗，回傳純價格數據以維持運作
            return df_price

        # C. 合併並計算外資成本線
        # 核心公式：僅計算外資「買超日」的加權平均價格
        combined = pd.concat([df_price, df_foreign[['net_buy']]], axis=1).dropna(subset=['Close'])
        
        def get_weighted_cost(window_df):
            buys = window_df[window_df['net_buy'] > 0]
            if buys.empty: 
                return np.nan
            # 加權平均公式: (價格 * 買超張數) / 總買超張數
            return (buys['Close'] * buys['net_buy']).sum() / buys['net_buy'].sum()

        costs = []
        window = 20
        for i in range(len(combined)):
            if i < window: 
                costs.append(np.nan)
            else:
                win = combined.iloc[i-window+1 : i+1]
                costs.append(get_weighted_cost(win))
        
        combined['Foreign_Cost_Line'] = costs
        combined['Foreign_Cost_Line'] = combined['Foreign_Cost_Line'].ffill() # 缺值向下填充
        return combined

    def get_realtime_signal(self, stock_id):
        try:
            ticker = yf.Ticker(f"{stock_id}.TW")
            fast = ticker.fast_info
            last = fast.last_price
            open_p = fast.open
            prev_c = fast.previous_close
            
            if last > open_p and open_p > prev_c: 
                signal = "🟢 強勢多頭 (開高走高)"
            elif last < open_p: 
                signal = "🟡 留意回檔 (開高走低)"
            else: 
                signal = "⚪ 震盪整理"
            return last, round((open_p/prev_c-1)*100, 2), signal
        except:
            return 0.0, 0.0, "數據讀取中..."

# --- 3. Streamlit UI 介面 ---
st.title("🚀 2026 台股雙核監控系統")
st.markdown("---")

# 側邊欄：標的選擇
stock_options = {
    "台積電 (TSMC)": "2330",
    "元大台灣50 (0050)": "0050",
    "富邦台50 (006208)": "006208",
    "國泰領袖50 (00922)": "00922",
    "統一台股 (主動型)": "00981A",
    "群益精選 (主動型)": "00982A"
}
target_name = st.sidebar.selectbox("🎯 選擇監控標的", list(stock_options.keys()))
target_id = stock_options[target_name]

# 初始化監控器
monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# A. 即時獵手區
st.subheader(f"📡 即時行情驗證：{target_name}")
last, gap, sig = monitor.get_realtime_signal(target_id)
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("當前成交價", f"${last:.2f}")
with c2:
    st.metric("開盤漲幅 %", f"{gap}%")
with c3:
    st.info(f"盤中訊號：{sig}")

# B. 籌碼深度分析區
st.divider()
st.subheader("📊 外資加權成本線 (籌碼防線分析)")

with st.spinner("正在對接 FinMind 獲取法人籌碼..."):
    df = monitor.get_full_analysis_data(target_id)
    
    if not df.empty and 'Foreign_Cost_Line' in df.columns:
        latest = df.iloc[-1]
        f_cost = latest['Foreign_Cost_Line']
        # 計算乖離率
        bias = (latest['Close'] / f_cost - 1) * 100 if f_cost > 0 else 0

        # Plotly 圖表繪製
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="日 K 收盤價", line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Foreign_Cost_Line'], name="外資 20 日成本線", line=dict(color="#d62728", dash='dot', width=2)))
        
        fig.update_layout(
            title=f"{target_name} 股價 vs. 法人成本",
            template="plotly_dark",
            height=500,
            xaxis_title="日期",
            yaxis_title="價格 (TWD)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 顯示警示文字
        if bias < 3:
            st.success(f"✅ 當前股價離外資成本僅 **{bias:.2f}%**。屬於法人防禦區，長線佈局勝率高。")
        elif bias > 10:
            st.warning(f"⚠️ 當前乖離率高達 **{bias:.2f}%**。短線漲幅過快，建議等待回測成本線再進場。")
        else:
            st.info(f"🔍 目前乖離率為 **{bias:.2f}%**。趨勢穩定，持續觀察外資買超連續性。")
    else:
        st.error("暫時無法獲取籌碼數據，請檢查 Token 額度或稍後再試。")

# C. 2026 投資布局指引
st.divider()
st.subheader("📅 2026 年度投資布局戰略")
curr_month = datetime.now().month
strategies = {
    "Q1": "✨ **第一季：擴張期**。台積電法說上修資本支出。策略：市值型 ETF 為主，捕捉大盤向上推升動能。",
    "Q2": "📉 **第二季：防禦期**。留意繳稅季資金壓力與毛利率震盪。策略：觀察外資成本線，回測不破則是絕佳買點。",
    "Q3": "🚀 **第三季：噴發期**。2nm 與先進封裝進入出貨高峰。策略：提高「主動型 ETF」比例，捕捉供應鏈超額報酬。",
    "Q4": "💰 **第四季：收穫期**。法人年終作帳與明年展望。策略：回歸 0050/006208 等權值標的，落袋為安。"
}
curr_q = f"Q{(curr_month-1)//3 + 1}"
st.success(strategies[curr_q])
