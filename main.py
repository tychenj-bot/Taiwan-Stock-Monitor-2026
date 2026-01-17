import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統設定與頁面配置 ---
st.set_page_config(page_title="2026 AI 雙核監控系統", layout="wide")

# 從 Streamlit Secrets 讀取 Token
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
            # 兼容性登入
            if hasattr(self.api, 'login'):
                self.api.login(token=token.strip())
            else:
                self.api.token = token.strip()
        except:
            pass

    @st.cache_data(ttl=3600)
    def get_comprehensive_data(_self, stock_id, days=150):
        # A. 價格與 ADR 數據 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df_price = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df_price.empty: return pd.DataFrame()
        df_price.index = df_price.index.tz_localize(None).normalize()
        df_price = df_price[~df_price.index.duplicated(keep='last')]
        
        # 抓取 ADR (TSM) 參考資訊
        adr = yf.Ticker("TSM").history(period=f"{days}d")
        adr.index = adr.index.tz_localize(None).normalize()
        df_price['ADR_Premium'] = ((adr['Close'] / 5 * 32) / df_price['Close'] - 1) * 100

        # B. 籌碼數據 (FinMind 1.9.3)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            # 1. 外資 (Foreign)
            df_f = df_chip[df_chip['name'].str.contains('Foreign', case=False)].copy()
            df_f['date'] = pd.to_datetime(df_f['date'])
            df_f = df_f.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
            df_f['f_net'] = df_f['buy'] - df_f['sell']
            
            # 2. 投信 (Investment)
            df_it = df_chip[df_chip['name'].str.contains('Investment', case=False)].copy()
            df_it['date'] = pd.to_datetime(df_it['date'])
            df_it = df_it.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
            df_it['it_net'] = df_it['buy'] - df_it['sell']
        except:
            return df_price

        # C. 合併數據與計算成本線 (VWAP)
        combined = pd.concat([df_price, df_f[['f_net']], df_it[['it_net']]], axis=1)
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

# --- 3. UI 介面與標的清單 ---
st.title("🏹 2026 AI 雙核戰略系統")

# 補齊標的代號清單 (包含市場前三、主動前三、以及權值標竿)
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
    "半導體核心": {
        "台積電 (2330)": "2330",
        "弘塑科技 (3131)": "3131",
        "辛耘企業 (3583)": "3583"
    }
}

st.sidebar.header("📊 監控清單")
cat = st.sidebar.selectbox("標的分類", list(monitored_targets.keys()))
name = st.sidebar.selectbox("選擇個股/ETF", list(monitored_targets[cat].keys()))
stock_id = monitored_targets[cat][name]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)
df = monitor.get_comprehensive_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    # 計算 20 日成交均量
    avg_vol_20 = df['Volume'].rolling(20).mean().iloc[-1]
    
    # 指標提取
    price = latest['Close']
    f_cost = latest['Foreign_Cost']
    bias = (price / f_cost - 1) * 100 if f_cost > 0 else 0
    adr_pre = latest['ADR_Premium']
    it_net = latest['it_net']
    curr_vol = latest['Volume']

    # --- 4. 自動警示燈號邏輯 ---
    st.subheader("🚦 2026 戰略過濾燈號")
    
    # 🟢 綠燈 (佈局期)
    if price < f_cost * 1.02 and curr_vol > avg_vol_20:
        st.success(f"🟢 **綠燈 (佈局期)**：價格貼近成本線 (${f_cost:.1f}) 且爆量。建議：分批加碼市值型 ETF。")
    
    # 🔴 紅燈 (警戒期) - 優先權高於黃燈
    elif price < f_cost and it_net < 0:
        st.error(f"🔴 **紅燈 (警戒期)**：跌破外資防線且投信同步倒貨。建議：強制減碼，避開 Q2 可能修正。")
        
    # 🟡 黃燈 (觀望期)
    elif bias > 10 or adr_pre < 0:
        st.warning(f"🟡 **黃燈 (觀望期)**：乖離({bias:.1f}%)過大或 ADR 負溢價。建議：主動型 ETF 獲利了結。")
        
    else:
        st.info("⚪ **盤整期**：目前數據處於常態區間，維持現有部位。")

    # 數據看板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("當前股價", f"${price:.2f}")
    c2.metric("外資成本", f"${f_cost:.1f}")
    c3.metric("成本乖離 %", f"{bias:.2f}%")
    c4.metric("ADR 溢價 %", f"{adr_pre:.2f}%")

    # --- 5. 戰略可視化圖表 ---
    fig = go.Figure()
    # 股價線
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="收盤價", line=dict(color='#1f77b4', width=2.5)))
    # 外資成本線
    fig.add_trace(go.Scatter(x=df.index, y=df['Foreign_Cost'], name="外資 20 日加權成本", line=dict(color='#d62728', dash='dot')))
    
    # 畫出佈局區區間 (成本線上下 2%)
    fig.add_hrect(y0=f_cost*0.98, y1=f_cost*1.02, line_width=0, fillcolor="green", opacity=0.1)

    fig.update_layout(template="plotly_dark", height=500, title=f"{name} 戰略趨勢", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 投信籌碼條形圖
    st.subheader("🏢 投信籌碼監控 (近 30 日)")
    it_fig = go.Bar(x=df.index[-30:], y=df['it_net'].iloc[-30:], marker_color='orange', name="投信買賣超")
    st.plotly_chart(go.Figure(data=[it_fig], layout=dict(template="plotly_dark", height=250)), use_container_width=True)

else:
    st.error("無法載入數據，請檢查標的代號或 API 權限。")

st.divider()
st.caption("2026 戰略提醒：主動型 ETF 適合於綠燈轉盤整期攻擊，紅燈出現時請果斷切換回 0050/006208。")
