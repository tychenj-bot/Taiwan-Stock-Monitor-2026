import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統設定 ---
st.set_page_config(page_title="2026 台股雙核監控系統", layout="wide")

# 安全讀取 Secrets
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
                # 修正：最新版 login 邏輯
                self.api.login(token=clean_token)
                st.success("✅ 帳號登入成功")
                self.login_status = True
            except Exception as e:
                # 即使失敗也嘗試抓取 (某些版本支援隱含登入)
                st.warning(f"⚠️ 登入提示: {e}")
                try:
                    # 診斷測試：修正為 stock_id
                    test_df = self.api.taiwan_stock_daily(stock_id="2330", start_date="2026-01-01")
                    if not test_df.empty:
                        st.success("✅ 數據連接正常")
                        self.login_status = True
                except Exception as e2:
                    st.error(f"❌ 診斷發現問題: {e2}")

    @st.cache_data(ttl=3600)
    def get_full_analysis_data(_self, stock_id, days=60):
        """整合 yfinance 價格與 FinMind 籌碼"""
        # A. 抓取價格 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df_price = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df_price.empty: return pd.DataFrame()
        df_price.index = df_price.index.tz_localize(None).normalize()

        # B. 抓取籌碼 (FinMind 1.9.3 使用 stock_id)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(
                stock_id=stock_id,  # 修正：data_id -> stock_id
                start_date=start_date
            )
            # 過濾外資
            df_foreign = df_chip[df_chip['name'].str.contains('Foreign', case=False, na=False)].copy()
            df_foreign['date'] = pd.to_datetime(df_foreign['date'])
            df_foreign = df_foreign.set_index('date')
            
            # 確保有 net_buy 欄位
            if 'net_buy' not in df_foreign.columns:
                df_foreign['net_buy'] = df_foreign['buy'] - df_foreign['sell']
        except:
            return df_price

        # C. 計算外資加權成本線
        combined = pd.concat([df_price, df_foreign[['net_buy']]], axis=1).dropna(subset=['Close'])
        
        # 公式：$Foreign\ Cost = \frac{\sum (Price \times Net\ Buy)}{\sum Net\ Buy}$ (僅計算買超日)
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
st.markdown(f"**當前日期：2026-01-18 (週末數據更新)**")

stock_options = {"台積電": "2330", "元大台灣50": "0050", "富邦台50": "006208", "統一台股(主動型)": "00981A"}
target_name = st.sidebar.selectbox("🎯 監控標的", list(stock_options.keys()))
target_id = stock_options[target_name]

# 初始化監控器
monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# 即時區
last, gap = monitor.get_realtime_signal(target_id)
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("當前股價", f"${last:.2f}")
c2.metric("開盤漲跌", f"{gap}%")
c3.success("✨ 2026 戰略：關注 2nm 供應鏈回測外資成本線之買點。")

# 籌碼圖表
st.divider()
st.subheader("📊 外資加權成本分析")

with st.spinner("同步數據中..."):
    df = monitor.get_full_analysis_data(target_id)
    if not df.empty and 'Foreign_Cost_Line' in df.columns:
        latest = df.iloc[-1]
        f_cost = latest['Foreign_Cost_Line']
        bias = (latest['Close'] / f_cost - 1) * 100 if f_cost > 0 else 0
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="收盤價", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df.index, y=df['Foreign_Cost_Line'], name="外資成本", line=dict(color="#d62728", dash='dot')))
        fig.update_layout(template="plotly_dark", height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"💡 目前 **{target_name}** 股價距外資成本乖離率：**{bias:.2f}%**")
    else:
        st.warning("⚠️ 週末時段或 Token 權限受限，僅能顯示基礎價格資訊。")
