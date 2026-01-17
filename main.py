import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統配置 ---
st.set_page_config(page_title="2026 雙核戰略系統 v3.0", layout="wide")

if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN")
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
    def get_full_analysis(_self, stock_id, days=120):
        # A. 價格與 ADR 抓取
        ticker_yf = f"{stock_id}.TW"
        df = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df.empty: return pd.DataFrame()
        df.index = df.index.tz_localize(None).normalize()
        
        # 深夜校正指標：TSM ADR 與 NVDA
        adr = yf.Ticker("TSM").history(period=f"{days}d")
        nvda = yf.Ticker("NVDA").history(period=f"{days}d")
        adr.index = adr.index.tz_localize(None).normalize()
        nvda.index = nvda.index.tz_localize(None).normalize()
        
        df['ADR_Premium'] = ((adr['Close'] / 5 * 32) / df['Close'] - 1) * 100
        df['AI_Momentum'] = nvda['Close'].pct_change() * 100 # NVDA 當日漲跌幅

        # B. 籌碼數據 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            # 外資與投信
            for name in ['Foreign', 'Investment']:
                sub = df_chip[df_chip['name'].str.contains(name, case=False)].copy()
                sub['date'] = pd.to_datetime(sub['date'])
                sub = sub.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
                df[f'{name.lower()}_net'] = sub['buy'] - sub['sell']
        except:
            pass

        df = df.fillna(0)
        
        # C. 晚上分析指標：外資連買與籌碼集中度
        # 連買天數計算
        f_net = df['foreign_net'].tolist()
        consecutive = 0
        for val in reversed(f_net):
            if val > 0: consecutive += 1
            else: break
        
        # 籌碼集中度：(三大法人買超和 / 總成交量)
        df['Concentration'] = (df['foreign_net'] + df['investment_net']) / df['Volume'] * 100
        
        # 外資加權成本
        costs = []
        for i in range(len(df)):
            win = df.iloc[max(0, i-19) : i+1]
            buys = win[win['foreign_net'] > 0]
            costs.append((buys['Close'] * buys['foreign_net']).sum() / buys['foreign_net'].sum() if not buys.empty else np.nan)
        df['Foreign_Cost'] = pd.Series(costs, index=df.index).ffill()
        
        return df, consecutive

# --- 3. UI 介面與邏輯分頁 ---
st.title("🏹 2026 雙核交易執行看板")

monitored_targets = {
    "市場型 Top 3": {"0050": "0050", "006208": "006208", "00922": "00922"},
    "主動型 Top 3": {"00981A": "00981A", "00982A": "00982A", "00980A": "00980A"},
    "核心權值": {"2330": "2330", "3131": "3131", "3583": "3583"}
}

cat = st.sidebar.selectbox("標的分類", list(monitored_targets.keys()))
name = st.sidebar.selectbox("監控標的代號", list(monitored_targets[cat].keys()))
stock_id = monitored_targets[cat][name]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)
df, con_buy = monitor.get_full_analysis(stock_id)

# --- 操作邏輯三階段展示 ---
tab1, tab2, tab3 = st.tabs(["🌙 晚上：分析期", "🌌 深夜：校正期", "☀️ 開盤：執行期"])

if not df.empty:
    latest = df.iloc[-1]
    f_cost = latest['Foreign_Cost']
    bias = (latest['Close'] / f_cost - 1) * 100 if f_cost > 0 else 0
    
    with tab1:
        st.subheader("篩選指標：籌碼集中度與連買")
        col1, col2 = st.columns(2)
        col1.metric("外資連續買超天數", f"{con_buy} 天", delta="強勢" if con_buy >= 3 else "觀察")
        col2.metric("最新籌碼集中度", f"{latest['Concentration']:.2f}%")
        
        # 集中度趨勢圖
        fig_con = go.Figure(go.Bar(x=df.index[-20:], y=df['Concentration'].iloc[-20:], marker_color='lightblue'))
        fig_con.update_layout(title="近 20 日籌碼集中度趨勢", template="plotly_dark", height=300)
        st.plotly_chart(fig_con, use_container_width=True)

    with tab2:
        st.subheader("美股聯動：ADR 與 AI 族群")
        col1, col2 = st.columns(2)
        col1.metric("ADR 溢價率", f"{latest['ADR_Premium']:.2f}%", delta="領先訊號" if latest['ADR_Premium'] > 0 else "拖累訊號")
        col2.metric("NVDA 動能 (AI 族群)", f"{latest['AI_Momentum']:.2f}%")
        st.info("💡 深夜校正邏輯：若 ADR 出現負溢價且 NVDA 重挫，隔日開盤需嚴防跳空下殺。")

    with tab3:
        st.subheader("即時執行燈號")
        # 獲取開盤數據
        info = yf.Ticker(f"{stock_id}.TW").fast_info
        gap = (info.open / info.previous_close - 1) * 100
        vol_ratio = info.last_volume / (df['Volume'].mean()) # 簡易開盤量比
        
        # 🟢🟡🔴 邏輯判斷
        if info.last_price < f_cost * 1.02 and info.last_price > info.open:
            st.success("🟢 綠燈：符合預期，買盤強勁且貼近成本。進場市值型 ETF。")
        elif bias > 10 or latest['ADR_Premium'] < 0:
            st.warning("🟡 黃燈：開太高或 ADR 轉弱。停止追高，主動型獲利了結。")
        elif info.last_price < f_cost:
            st.error("🔴 紅燈：跌破外資防線。強制減碼，回防 0050。")
            
        st.metric("開盤跳空幅度", f"{gap:.2f}%")
        
        # 戰略趨勢圖
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="價格", line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Foreign_Cost'].iloc[-60:], name="外資成本線", line=dict(color='#d62728', dash='dot')))
        fig.update_layout(template="plotly_dark", height=400, title="執行期參考：價格 vs. 成本線")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("2026 操作備忘：晚上選股、深夜校對、開盤決斷。嚴守外資成本線，不與趨勢作對。")
