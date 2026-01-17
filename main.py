import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統設定 ---
st.set_page_config(page_title="2026 台股雙核監控系統", layout="wide")

if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN，請檢查 Secrets 設定。")
    st.stop()
else:
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]

# --- 2. 核心分析類別 ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        self.api = DataLoader()
        self.login_status = False
        
        with st.sidebar.expander("🛠️ 系統診斷報告 (v1.9.3)", expanded=True):
            clean_token = token.strip()
            import FinMind
            st.write(f"📦 FinMind 版本: `{FinMind.__version__}`")
            
            try:
                self.api.login(token=clean_token)
                st.success("✅ 帳號登入成功")
                self.login_status = True
            except Exception as e:
                st.warning(f"⚠️ 登入提示: {e}")

    @st.cache_data(ttl=3600)
    def get_full_analysis_data(_self, stock_id, days=120): # 增加天數以確保計算 20 日均線穩定
        # A. 抓取價格 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df_price = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df_price.empty: return pd.DataFrame()
        df_price.index = df_price.index.tz_localize(None).normalize()
        # 確保價格索引唯一 (去重)
        df_price = df_price[~df_price.index.duplicated(keep='last')]

        # B. 抓取籌碼 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(
                stock_id=stock_id,
                start_date=start_date
            )
            # 過濾外資
            df_foreign = df_chip[df_chip['name'].str.contains('Foreign', case=False, na=False)].copy()
            df_foreign['date'] = pd.to_datetime(df_foreign['date'])
            df_foreign = df_foreign.set_index('date')
            
            # --- 關鍵修正：解決 InvalidIndexError ---
            # 將同一天的數據加總，確保每個日期只有一筆數據
            df_foreign = df_foreign.groupby(df_foreign.index).agg({
                'buy': 'sum',
                'sell': 'sum'
            })
            df_foreign['net_buy'] = df_foreign['buy'] - df_foreign['sell']
        except Exception as e:
            st.sidebar.error(f"籌碼處理錯誤: {e}")
            return df_price

        # C. 合併數據 (處理索引對齊)
        # 使用 how='left' 以價格日期為主
        combined = pd.concat([df_price, df_foreign[['net_buy']]], axis=1)
        combined = combined.dropna(subset=['Close']) # 移除沒有價格的日期
        combined['net_buy'] = combined['net_buy'].fillna(0) # 籌碼空值補 0

        # D. 計算外資加權成本線 (20日)
        def get_weighted_cost(window_df):
            buys = window_df[window_df['net_buy'] > 0]
            if buys.empty: return np.nan
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
        combined['Foreign_Cost_Line'] = combined['Foreign_Cost_Line'].ffill()
        return combined

    def get_realtime_signal(self, stock_id):
        try:
            ticker = yf.Ticker(f"{stock_id}.TW")
            fast = ticker.fast_info
            return fast.last_price, round((fast.open/fast.previous_close-1)*100, 2)
        except:
            return 0.0, 0.0

# --- 3. UI 呈現 ---
st.title("🚀 2026 台股雙核監控系統")
st.write(f"📅 **數據更新至：{datetime.now().strftime('%Y-%m-%d %H:%M')}** (週末顯示前一交易日數據)")

stock_options = {
    "台積電 (2330)": "2330", 
    "元大台灣50 (0050)": "0050", 
    "富邦台50 (006208)": "006208", 
    "統一台股(主動型)": "00981A"
}
target_name = st.sidebar.selectbox("🎯 選擇監控標的", list(stock_options.keys()))
target_id = stock_options[target_name]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# 即時區
last, gap = monitor.get_realtime_signal(target_id)
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("當前股價", f"${last:.2f}")
c2.metric("開盤漲跌 %", f"{gap}%")
c3.info("🎯 **策略提示**：2026 年市場聚焦 2nm 量產進度，建議於乖離率 < 3% 時進場。")

# 籌碼圖表
st.divider()
st.subheader("📊 外資加權成本分析 (Foreign VWAP)")

with st.spinner("正在對齊價格與籌碼數據..."):
    df = monitor.get_full_analysis_data(target_id)
    if not df.empty and 'Foreign_Cost_Line' in df.columns:
        latest = df.iloc[-1]
        f_cost = latest['Foreign_Cost_Line']
        bias = (latest['Close'] / f_cost - 1) * 100 if f_cost > 0 else 0
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="日 K 收盤價", line=dict(color="#1f77b4", width=2.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Foreign_Cost_Line'], name="外資 20 日加權成本", line=dict(color="#d62728", dash='dot', width=2)))
        
        fig.update_layout(
            template="plotly_dark", 
            height=550, 
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 顯示乖離率診斷
        if bias < 3:
            st.success(f"💎 **黃金區**：當前乖離率僅 **{bias:.2f}%**。股價極貼近外資成本 ({f_cost:.2f})，支撐力道強。")
        elif bias > 10:
            st.warning(f"🔥 **過熱區**：當前乖離率達 **{bias:.2f}%**。短線獲利了結壓力大，建議等待回測。")
        else:
            st.info(f"⚖️ **中性區**：目前乖離率 **{bias:.2f}%**。趨勢維持穩定。")
    else:
        st.warning("⚠️ 無法合併數據，可能是因為該標的最近 20 日外資無買超紀錄。")

st.divider()
st.caption("註：本系統之『外資成本線』僅計算外資買超日之成交價量權重，較傳統移動平均線更具籌碼參考價值。")
