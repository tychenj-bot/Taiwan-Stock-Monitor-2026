import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統設定與頁面配置 ---
st.set_page_config(page_title="2026 AI 雙核戰略系統", layout="wide")

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
        try:
            # 兼容性登入與手動注入
            if hasattr(self.api, 'login'): self.api.login(token=token.strip())
            else: self.api.token = token.strip()
        except: pass

    @st.cache_data(ttl=3600)
    def get_comprehensive_data(_self, stock_id, days=150):
        # A. 基礎數據與深夜校正指標 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df.empty: return pd.DataFrame(), 0
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]
        
        # 抓取 ADR (TSM) 與 AI 動能指標 (NVDA)
        adr = yf.Ticker("TSM").history(period=f"{days}d")
        nvda = yf.Ticker("NVDA").history(period=f"{days}d")
        adr.index = adr.index.tz_localize(None).normalize()
        nvda.index = nvda.index.tz_localize(None).normalize()
        
        df['ADR_Premium'] = ((adr['Close'] / 5 * 32) / df['Close'] - 1) * 100
        df['AI_Momentum'] = nvda['Close'].pct_change() * 100

        # B. 籌碼數據分析 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            # 處理外資與投信
            for name in ['Foreign', 'Investment']:
                sub = df_chip[df_chip['name'].str.contains(name, case=False)].copy()
                sub['date'] = pd.to_datetime(sub['date'])
                sub = sub.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
                df[f'{name.lower()}_net'] = sub['buy'] - sub['sell']
        except:
            df['foreign_net'] = 0
            df['investment_net'] = 0

        df = df.fillna(0)
        
        # C. 外資成本線演算法 (VWAP - 永不消失補丁)
        costs = []
        last_valid = np.nan
        for i in range(len(df)):
            win = df.iloc[max(0, i-19) : i+1]
            buys = win[win['foreign_net'] > 0]
            if not buys.empty:
                last_valid = (buys['Close'] * buys['foreign_net']).sum() / buys['foreign_net'].sum()
            costs.append(last_valid)
        
        df['Foreign_Cost'] = costs
        df['Foreign_Cost'] = df['Foreign_Cost'].ffill().bfill()
        
        # D. 籌碼集中度與連買天數
        df['Concentration'] = (df['foreign_net'] + df['investment_net']) / df['Volume'] * 100
        f_net_list = df['foreign_net'].tolist()
        consecutive = 0
        for val in reversed(f_net_list):
            if val > 0: consecutive += 1
            elif val < 0: break
            
        return df, consecutive

# --- 3. UI 介面與標的清單 ---
st.title("🏹 2026 AI 雙核全功能戰略系統")

# 補齊標的清單
monitored_targets = {
    "市場型 (市值型) Top 3": {
        "元大台灣50 (0050)": "0050",
        "富邦台50 (006208)": "006208",
        "國泰領袖50 (00922)": "00922"
    },
    "主動型成長 Top 3": {
        "統一台股主動 (00981A)": "00981A",
        "群益精選主動 (00982A)": "00982A",
        "復華台灣主動 (00980A)": "00980A"
    },
    "2nm 供應鏈核心": {
        "台積電 (2330)": "2330",
        "弘塑科技 (3131)": "3131",
        "辛耘企業 (3583)": "3583",
        "萬潤 (6187)": "6187"
    }
}

st.sidebar.header("📊 監控清單")
cat = st.sidebar.selectbox("標的分類", list(monitored_targets.keys()))
stock_name = st.sidebar.selectbox("選擇標的", list(monitored_targets[cat].keys()))
stock_id = monitored_targets[cat][stock_name]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)
df, con_buy = monitor.get_comprehensive_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    f_cost = latest['Foreign_Cost']
    bias = (latest['Close'] / f_cost - 1) * 100 if f_cost > 0 else 0
    
    # --- 4. 三階段操作看板 ---
    tab1, tab2, tab3 = st.tabs(["🌙 晚上：分析期", "🌌 深夜：校正期", "☀️ 開盤：執行期"])

    with tab1:
        st.subheader("篩選指標：外資連買與集中度")
        c1, c2 = st.columns(2)
        c1.metric("外資連買天數", f"{con_buy} 天")
        c2.metric("最新籌碼集中度", f"{latest['Concentration']:.2f}%")
        
        fig_con = go.Figure(go.Bar(x=df.index[-20:], y=df['Concentration'].iloc[-20:], marker_color='lightblue'))
        fig_con.update_layout(title="近 20 日籌碼集中度趨勢", template="plotly_dark", height=300)
        st.plotly_chart(fig_con, use_container_width=True)

    with tab2:
        st.subheader("美股校正：ADR 與 AI 動能")
        c1, c2 = st.columns(2)
        c1.metric("ADR 溢價率", f"{latest['ADR_Premium']:.2f}%")
        c2.metric("NVDA 當前動能", f"{latest['AI_Momentum']:.2f}%")
        st.info("💡 提醒：若深夜 ADR 轉為負溢價，隔日執行期應轉為保守。")

    with tab3:
        st.subheader("🚦 自動化執行燈號")
        
        # 獲取開盤即時指標
        info = yf.Ticker(f"{stock_id}.TW").fast_info
        last_price = info.last_price
        open_price = info.open
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        
        # 🟢🟡🔴 紅綠燈邏輯整合
        if last_price < f_cost * 1.02 and last_price > open_price and info.last_volume > (avg_vol/4):
            st.success("🟢 **綠燈 (佈局期)**：符合預期，買盤強勁且貼近成本。建議：分批加碼市值型 ETF。")
        elif bias > 10 or latest['ADR_Premium'] < 0:
            st.warning(f"🟡 **黃燈 (觀望期)**：乖離過大({bias:.1f}%)或 ADR 負溢價。建議：主動型獲利了結。")
        elif last_price < f_cost and latest['investment_net'] < 0:
            st.error("🔴 **紅燈 (警戒期)**：跌破外資防線且投信同步賣超。建議：強制減碼，避開修正。")
        else:
            st.info("⚪ **盤整期**：目前數據處於常態，不進行大動作調整。")

        # 核心數據
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("當前股價", f"${last_price:.2f}")
        cc2.metric("外資成本線", f"${f_cost:.1f}")
        cc3.metric("外資乖離 %", f"{bias:.2f}%")

        # 戰略趨勢圖
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="價格", line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Foreign_Cost'].iloc[-60:], name="外資成本防線", line=dict(color='#d62728', dash='dot')))
        fig.add_hrect(y0=f_cost*0.98, y1=f_cost*1.02, line_width=0, fillcolor="green", opacity=0.1)
        fig.update_layout(template="plotly_dark", height=450, title=f"{stock_name} 執行期參考圖表")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("2026 戰略提醒：晚上分析籌碼集中度，深夜對齊 ADR 溢價，開盤觀測量價執行。")
