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
if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN，請前往 Streamlit Cloud 的 Settings -> Secrets 進行設定。")
    st.stop()
else:
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]

# --- 2. 核心分析類別 ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        # 建立側邊欄診斷區
        with st.sidebar.expander("🛠️ 系統診斷資訊", expanded=True):
            if not token:
                st.error("❌ Token 為空值")
                self.login_status = False
            else:
                st.write(f"🔑 Token 前綴: `{token[:6]}...`")
                self.login_status = True

            import FinMind
            st.write(f"📦 FinMind 版本: `{FinMind.__version__}`")
            
            self.api = DataLoader()
            
            # 自動偵測登入指令相容性
            try:
                if hasattr(self.api, 'login'):
                    self.api.login(token=token)
                    st.success("✅ 成功呼叫 login")
                elif hasattr(self.api, 'login_token'):
                    self.api.login_token(token=token)
                    st.success("✅ 成功呼叫 login_token")
                else:
                    st.warning("⚠️ 找不到登入指令")
            except Exception as e:
                st.error(f"❌ 登入報錯: {e}")

    @st.cache_data(ttl=3600)
    def get_full_analysis_data(_self, stock_id, days=60):
        # A. 價格數據 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df_price = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df_price.empty:
            return pd.DataFrame()
        df_price.index = df_price.index.tz_localize(None).normalize()

        # B. 籌碼數據 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(
                data_id=stock_id,
                start_date=start_date
            )
            df_foreign = df_chip[df_chip['name'] == 'Foreign_Investor'].copy()
            df_foreign['date'] = pd.to_datetime(df_foreign['date'])
            df_foreign = df_foreign.set_index('date')
        except Exception:
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
            last, open_p, prev_c = fast.last_price, fast.open, fast.previous_close
            if last > open_p and open_p > prev_c: signal = "🟢 強勢多頭"
            elif last < open_p: signal = "🟡 留意回檔"
            else: signal = "⚪ 震盪整理"
            return last, round((open_p/prev_c-1)*100, 2), signal
        except:
            return 0.0, 0.0, "數據讀取中..."

# --- 3. UI 介面 ---
st.title("🚀 2026 台股雙核監控系統")
st.markdown("---")

stock_options = {
    "台積電 (2330)": "2330",
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

# A. 即時行情
last, gap, sig = monitor.get_realtime_signal(target_id)
c1, c2, c3 = st.columns(3)
c1.metric("當前成交價", f"${last:.2f}")
c2.metric("開盤漲幅 %", f"{gap}%")
c3.info(f"盤中訊號：{sig}")

# B. 成本線圖表
st.divider()
st.subheader("📊 外資加權成本分析")

with st.spinner("正在對接 FinMind 獲取籌碼..."):
    df = monitor.get_full_analysis_data(target_id)
    if not df.empty and 'Foreign_Cost_Line' in df.columns:
        latest = df.iloc[-1]
        f_cost = latest['Foreign_Cost_Line']
        bias = (latest['Close'] / f_cost - 1) * 100 if f_cost > 0 else 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="收盤價", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df.index, y=df['Foreign_Cost_Line'], name="外資成本線", line=dict(color="#d62728", dash='dot')))
        fig.update_layout(template="plotly_dark", height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"💡 目前乖離率：**{bias:.2f}%** (外資加權成本: {f_cost:.2f})")
    else:
        st.warning("⚠️ 籌碼數據載入中或 Token 權限不足，目前僅顯示價格。")

# C. 布局策略
st.divider()
st.success(f"📅 2026 年度戰略：當前月份建議執行 {datetime.now().month} 月份佈局計劃。")
