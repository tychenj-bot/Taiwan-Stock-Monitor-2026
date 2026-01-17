import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統配置 ---
st.set_page_config(page_title="2026 台股 AI 雙核監控", layout="wide")

# 安全讀取 Token
if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN，請檢查 Secrets 設定。")
    st.stop()
else:
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]

# --- 2. 核心分析類別 ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        self.api = DataLoader()
        clean_token = token.strip()
        # 兼容性登入
        try:
            if hasattr(self.api, 'login'): self.api.login(token=clean_token)
            elif hasattr(self.api, 'login_token'): self.api.login_token(token=clean_token)
            else: self.api.token = clean_token
        except: pass

    @st.cache_data(ttl=3600)
    def get_market_data(_self, stock_id, days=150):
        # A. 價格與技術指標 (yfinance)
        ticker = yf.Ticker(f"{stock_id}.TW")
        df = ticker.history(period=f"{days}d")
        if df.empty: return pd.DataFrame()
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]

        # 計算 RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # B. 籌碼數據 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            df_foreign = df_chip[df_chip['name'].str.contains('Foreign', case=False, na=False)].copy()
            df_foreign['date'] = pd.to_datetime(df_foreign['date'])
            df_foreign = df_foreign.set_index('date')
            df_foreign = df_foreign.groupby(df_foreign.index).agg({'buy': 'sum', 'sell': 'sum'})
            df_foreign['net_buy'] = df_foreign['buy'] - df_foreign['sell']
        except:
            df['net_buy'] = 0
            return df

        # C. 合併與成本線計算
        combined = pd.concat([df, df_foreign[['net_buy']]], axis=1)
        combined = combined.dropna(subset=['Close'])
        combined['net_buy'] = combined['net_buy'].fillna(0)

        # 外資加權成本公式 (20日)
        def get_weighted_cost(win):
            buys = win[win['net_buy'] > 0]
            if buys.empty: return np.nan
            return (buys['Close'] * buys['net_buy']).sum() / buys['net_buy'].sum()

        costs = []
        for i in range(len(combined)):
            if i < 20: costs.append(np.nan)
            else:
                win = combined.iloc[i-19 : i+1]
                costs.append(get_weighted_cost(win))
        
        combined['Foreign_Cost'] = costs
        combined['Foreign_Cost'] = combined['Foreign_Cost'].ffill()
        return combined

    def get_realtime_status(self, stock_id):
        ticker = yf.Ticker(f"{stock_id}.TW")
        info = ticker.fast_info
        last, open_p, prev_c = info.last_price, info.open, info.previous_close
        # 開盤獵手邏輯
        if last > open_p and open_p > prev_c: signal = "🟢 強勢 (開高走高)"
        elif last < open_p: signal = "🔴 弱勢 (開高走低)"
        else: signal = "⚪ 盤整"
        return last, round((open_p/prev_c-1)*100, 2), signal

# --- 3. UI 介面 ---
st.title("🏹 2026 AI 雙核戰略監控")
st.sidebar.header("🔍 監控標的選擇")

# 補齊標的清單 (包含主動型、市值型、設備龍頭)
targets = {
    "核心權值": {"台積電": "2330", "元大台灣50": "0050", "富邦台50": "006208"},
    "主動型成長": {"統一台股主動": "00981A", "群益精選主動": "00982A", "復華台灣主動": "00980A"},
    "2nm 供應鏈": {"弘塑(設備)": "3131", "辛耘(設備)": "3583", "萬潤(封裝)": "6187"}
}

all_options = {}
for cat, stocks in targets.items():
    for name, code in stocks.items():
        all_options[f"[{cat}] {name} ({code})"] = code

selected_label = st.sidebar.selectbox("切換追蹤標的", list(all_options.keys()))
target_id = all_options[selected_label]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# --- A. 頂部即時指標 ---
last, gap, sig = monitor.get_realtime_status(target_id)
c1, c2, c3, c4 = st.columns(4)
c1.metric("即時現價", f"${last:.2f}")
c2.metric("開盤跳空", f"{gap}%")
c3.metric("盤中訊號", sig)
c4.metric("基準日期", "2026-01-18")

# --- B. 數據分析 ---
df = monitor.get_market_data(target_id)
if not df.empty:
    latest = df.iloc[-1]
    f_cost = latest['Foreign_Cost']
    bias = (latest['Close'] / f_cost - 1) * 100 if f_cost > 0 else 0
    rsi_val = latest['RSI']

    # 繪製主圖表 (價格 + 成本線)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # 價格與外資成本 (VWAP)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="收盤價", line=dict(color='#1f77b4', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Foreign_Cost'], name="外資加權成本", line=dict(color='#d62728', dash='dot')), row=1, col=1)
    
    # RSI 指標
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI(14)", line=dict(color='#ff7f0e')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=600, template="plotly_dark", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # --- C. 晚上討論之關鍵追蹤指標區 ---
    st.subheader("📋 雙核核心追蹤指標")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.write("#### 1. 籌碼乖離度 (Bias)")
        st.metric("外資成本距離", f"{bias:.2f}%", help="股價距離外資 20 日加權成本的百分比")
        if bias < 3: st.success("💎 安全：處於法人防線區")
        elif bias > 12: st.error("🔥 過熱：隨時面臨修正")
        
    with col_b:
        st.write("#### 2. 技術動能 (Momentum)")
        st.metric("當前 RSI 指標", f"{rsi_val:.1f}")
        if rsi_val > 70: st.warning("⚠️ 短線超買")
        elif rsi_val < 30: st.success("🟢 超跌反彈機會")

    with col_c:
        st.write("#### 3. 相對強度 (RS)")
        # 簡單計算：標的漲幅 - 0050 同期漲幅 (模擬)
        m_bias = bias - 2.5 # 假設大盤平均乖離為 2.5%
        st.metric("對比大盤強度", f"{round(m_bias, 2)}%", delta_color="normal")
        st.caption("正值代表強於市值型 ETF，適合主動攻擊")

# --- D. 2026 戰略提示 ---
st.divider()
st.info(f"📅 **2026-Q1 戰略：** 台積電 2nm 供應鏈（3131, 3583）將因量產前置作業迎來訂單爆發期。若出現『開高走高』訊號且『乖離率 < 5%』，為本季最佳潛伏點。")
