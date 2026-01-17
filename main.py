import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統配置 ---
st.set_page_config(page_title="2026 AI 雙核自動警示系統", layout="wide")

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
            self.api.login(token=token.strip())
        except:
            self.api.token = token.strip()

    @st.cache_data(ttl=3600)
    def get_comprehensive_data(_self, stock_id, days=120):
        # A. 抓取價格與 ADR (yfinance)
        ticker = yf.Ticker(f"{stock_id}.TW")
        df = ticker.history(period=f"{days}d")
        if df.empty: return pd.DataFrame()
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]
        
        # 抓取 ADR (TSM) 用於黃燈判斷
        adr = yf.Ticker("TSM").history(period=f"{days}d")
        adr.index = adr.index.tz_localize(None).normalize()
        # 簡易溢價估算 (假設 1 ADR = 5 股，匯率 32)
        df['ADR_Close'] = adr['Close']
        df['ADR_Premium'] = ((adr['Close'] / 5 * 32) / df['Close'] - 1) * 100

        # B. 抓取籌碼 (FinMind v1.9.3)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            # 1. 外資
            df_f = df_chip[df_chip['name'].str.contains('Foreign', case=False)].copy()
            df_f['date'] = pd.to_datetime(df_f['date'])
            df_f = df_f.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
            df_f['f_net'] = df_f['buy'] - df_f['sell']
            
            # 2. 投信 (用於紅燈判斷)
            df_it = df_chip[df_chip['name'].str.contains('Investment', case=False)].copy()
            df_it['date'] = pd.to_datetime(df_it['date'])
            df_it = df_it.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
            df_it['it_net'] = df_it['buy'] - df_it['sell']
        except:
            return df

        # C. 合併數據與計算成本線
        combined = pd.concat([df, df_f[['f_net']], df_it[['it_net']]], axis=1)
        combined = combined.dropna(subset=['Close']).fillna(0)

        # 外資加權成本 (20日)
        costs = []
        for i in range(len(combined)):
            if i < 20: costs.append(np.nan)
            else:
                win = combined.iloc[i-19 : i+1]
                buys = win[win['f_net'] > 0]
                cost = (buys['Close'] * buys['f_net']).sum() / buys['f_net'].sum() if not buys.empty else np.nan
                costs.append(cost)
        
        combined['Foreign_Cost'] = costs
        combined['Foreign_Cost'] = combined['Foreign_Cost'].ffill()
        return combined

# --- 3. UI 介面 ---
st.title("🏹 2026 AI 雙核：自動警示與策略過濾系統")

# 標的選擇
targets = {
    "權值型": {"台積電": "2330", "元大台灣50": "0050", "富邦台50": "006208"},
    "主動型": {"統一台股主動": "00981A", "群益精選主動": "00982A"},
    "設備商": {"弘塑": "3131", "辛耘": "3583", "萬潤": "6187"}
}
category = st.sidebar.selectbox("標的類別", list(targets.keys()))
stock_name = st.sidebar.selectbox("監控個股", list(targets[category].keys()))
stock_id = targets[category][stock_name]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)
df = monitor.get_comprehensive_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    prev_20_vol = df['Volume'].rolling(20).mean().iloc[-1]
    
    # 核心數據提取
    price = latest['Close']
    f_cost = latest['Foreign_Cost']
    bias = (price / f_cost - 1) * 100 if f_cost > 0 else 0
    adr_pre = latest['ADR_Premium']
    it_net = latest['it_net']
    vol = latest['Volume']

    # --- 4. 自動警示燈號邏輯 ---
    st.subheader("🚦 2026 戰略執行燈號")
    
    if price < f_cost * 1.02 and vol > prev_20_vol:
        st.success("🟢 綠燈 (佈局期)：價格極近成本線且爆量。建議：分批加碼市值型 ETF。")
        signal_color = "green"
    elif bias > 10 or adr_pre < 0:
        st.warning(f"🟡 黃燈 (觀望期)：乖離率({bias:.1f}%)過高或 ADR 負溢價({adr_pre:.1f}%)。建議：停止追高，主動型獲利了結。")
        signal_color = "yellow"
    elif price < f_cost and it_net < 0:
        st.error("🔴 紅燈 (警戒期)：跌破外資成本線且投信同步賣超。建議：系統強制發出減碼通知。")
        signal_color = "red"
    else:
        st.info("⚪ 盤整期：數據未達警示標準，維持既有配置。")
        signal_color = "white"

    # 數據看板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目前股價", f"${price:.2f}")
    c2.metric("外資成本", f"${f_cost:.2f}")
    c3.metric("外資乖離", f"{bias:.2f}%")
    c4.metric("ADR 溢價", f"{adr_pre:.2f}%")

    # --- 5. 圖表可視化 ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="股價", line=dict(color='#1f77b4', width=3)))
    fig.add_trace(go.Scatter
