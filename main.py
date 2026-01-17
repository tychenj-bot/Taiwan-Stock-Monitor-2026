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

# --- 2. 核心分析類別 (自癒邏輯) ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        self.api = DataLoader()
        self.login_status = False
        
        with st.sidebar.expander("🛠️ 系統診斷報告", expanded=True):
            clean_token = token.strip()
            import FinMind
            st.write(f"📦 FinMind 版本: `{FinMind.__version__}`")
            
            # 列出所有可用方法 (供偵錯參考)
            all_methods = dir(self.api)
            
            # --- 自癒登入邏輯 ---
            try:
                if 'login' in all_methods:
                    self.api.login(token=clean_token)
                    st.success("✅ 使用 `login` 登入成功")
                    self.login_status = True
                elif 'login_token' in all_methods:
                    self.api.login_token(token=clean_token)
                    st.success("✅ 使用 `login_token` 登入成功")
                    self.login_status = True
                else:
                    # 如果找不到登入指令，嘗試直接在請求時帶入 token (部分版本的做法)
                    st.warning("⚠️ 找不到標準登入指令，嘗試直接抓取數據...")
                    # 測試抓取一筆小數據驗證權限
                    test_df = self.api.taiwan_stock_daily(data_id="2330", start_date="2026-01-01")
                    if not test_df.empty:
                        st.success("✅ 數據連接正常 (匿名/隱含模式)")
                        self.login_status = True
            except Exception as e:
                st.error(f"❌ 診斷發現問題: {e}")

    @st.cache_data(ttl=3600)
    def get_full_analysis_data(_self, stock_id, days=60):
        # A. 抓取價格 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df_price = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df_price.empty: return pd.DataFrame()
        df_price.index = df_price.index.tz_localize(None).normalize()

        # B. 抓取籌碼 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(
                data_id=stock_id,
                start_date=start_date
            )
            # 兼容性過濾：尋找包含 'Foreign' 的欄位
            df_foreign = df_chip[df_chip['name'].str.contains('Foreign', case=False, na=False)].copy()
            df_foreign['date'] = pd.to_datetime(df_foreign['date'])
            df_foreign = df_foreign.set_index('date')
        except:
            return df_price

        # C. 計算成本線
        combined = pd.concat([df_price, df_foreign[['net_buy']]], axis=1).dropna(subset=['Close'])
        
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
        try:
            ticker = yf.Ticker(f"{stock_id}.TW")
            fast = ticker.fast_info
            return fast.last_price, round((fast.open/fast.previous_close-1)*100, 2)
        except:
            return 0.0, 0.0

# --- 3. 介面呈現 ---
st.title("🚀 2026 台股雙核監控系統")
st.markdown("---")

stock_options = {"台積電": "2330", "0050": "0050", "006208": "006208", "00981A": "00981A"}
target_name = st.sidebar.selectbox("🎯 監控標的", list(stock_options.keys()))
target_id = stock_options[target_name]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# 顯示即時指標
last, gap = monitor.get_realtime_signal(target_id)
c1, c2 = st.columns(2)
c1.metric("當前股價", f"${last:.2f}")
c2.metric("開盤漲幅", f"{gap}%")

# 顯示籌碼圖表
st.subheader("📊 外資加權成本分析")
with st.spinner("同步數據中..."):
    df = monitor.get_full_analysis_data(target_id)
    if not df.empty and 'Foreign_Cost_Line' in df.columns:
        latest = df.iloc[-1]
        bias = (latest['Close'] / latest['Foreign_Cost_Line'] - 1) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="收盤價"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Foreign_Cost_Line'], name="外資成本", line=dict(dash='dot')))
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"💡 目前股價距外資成本乖離率：**{bias:.2f}%**")
    else:
        st.warning("⚠️ 籌碼數據獲取中...")
