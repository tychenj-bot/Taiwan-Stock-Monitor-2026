import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統環境配置 ---
st.set_page_config(page_title="2026 ADR 優先戰情系統", layout="wide")

if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN，請檢查 Secrets 設定。")
    st.stop()
else:
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]

# --- 2. 核心運算引擎 ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        self.api = DataLoader()
        try:
            if hasattr(self.api, 'login'): self.api.login(token=token.strip())
            else: self.api.token = token.strip()
        except: pass

    @st.cache_data(ttl=600) # ADR 數據更新頻率較高，設為 10 分鐘
    def get_global_tsm_signal(_self):
        """專門抓取全域 TSM ADR 訊號"""
        try:
            tsm_adr = yf.Ticker("TSM").history(period="5d")
            tsm_tw = yf.Ticker("2330.TW").history(period="5d")
            
            # 確保時區一致
            tsm_adr.index = tsm_adr.index.tz_localize(None).normalize()
            tsm_tw.index = tsm_tw.index.tz_localize(None).normalize()
            
            # 取得最新價格
            adr_close = tsm_adr['Close'].iloc[-1]
            tw_close = tsm_tw['Close'].iloc[-1]
            
            # 簡易匯率 (可接 API，此處以 32.5 為基準，或動態調整)
            # 實戰中建議手動校正匯率，這裡示範 32.5
            fx_rate = 32.5 
            
            # 計算溢價率
            implied_price = (adr_close * fx_rate) / 5
            premium = ((implied_price / tw_close) - 1) * 100
            
            return premium, adr_close, implied_price
        except:
            return 0, 0, 0

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id, days=150):
        # A. 個股數據
        ticker_yf = f"{stock_id}.TW"
        df = yf.Ticker(ticker_yf).history(period=f"{days}d")
        
        if df.empty: return pd.DataFrame(), 0, 0
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]

        # 估算殖利率
        try:
            divs = yf.Ticker(ticker_yf).dividends
            if divs.index.tz is not None: divs.index = divs.index.tz_localize(None)
            one_year_ago = pd.Timestamp.now() - pd.DateOffset(months=12)
            est_yield = (divs[divs.index > one_year_ago].sum() / df['Close'].iloc[-1]) * 100
        except:
            est_yield = 0

        # RS 相對強度
        mkt = yf.Ticker("0050.TW").history(period=f"{days}d")
        mkt.index = mkt.index.tz_localize(None).normalize()
        df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100

        # B. 雙軌籌碼 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            for name in ['Foreign', 'Investment']:
                sub = df_chip[df_chip['name'].str.contains(name, case=False)].copy()
                sub['date'] = pd.to_datetime(sub['date'])
                sub = sub.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
                df[f'{name.lower()}_net'] = sub['buy'] - sub['sell']
        except:
            df['foreign_net'] = 0
            df['investment_net'] = 0

        df = df.fillna(0)
        
        # C. 成本線
        def calculate_vwap(net_buy_col):
            costs = []
            last_valid = np.nan
            for i in range(len(df)):
                win = df.iloc[max(0, i-19) : i+1]
                buys = win[win[net_buy_col] > 0]
                if not buys.empty:
                    last_valid = (buys['Close'] * buys[net_buy_col]).sum() / buys[net_buy_col].sum()
                costs.append(last_valid)
            return costs

        df['Foreign_Cost'] = pd.Series(calculate_vwap('foreign_net'), index=df.index).ffill().bfill()
        df['Invest_Cost'] = pd.Series(calculate_vwap('investment_net'), index=df.index).ffill().bfill()
        
        # 連買
        f_net_list = df['foreign_net'].tolist()
        consecutive = 0
        for val in reversed(f_net_list):
            if val > 0: consecutive += 1
            elif val < 0: break
            
        return df, consecutive, est_yield

# --- 3. UI 介面：ADR 優先戰情室 ---
st.title("🦅 2026 ADR 優先戰情系統")

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# --- 核心戰略：全域 ADR 儀表板 (置頂顯示) ---
st.markdown("### 🌎 全球戰略風向 (TSM ADR)")
adr_premium, adr_price, implied_tw = monitor.get_global_tsm_signal()

# ADR 儀表板設計
col_main, col_insight = st.columns([1, 2])

with col_main:
    # 根據溢價率變色
    delta_color = "normal"
    if adr_premium > 5: delta_color = "inverse" # 過熱
    elif adr_premium < 0: delta_color = "off"   # 折價
    
    st.metric(
        "TSM ADR 溢價率 (核心指標)", 
        f"{adr_premium:.2f}%", 
        f"美股收盤 ${adr_price:.2f}",
        delta_color=delta_color
    )

with col_insight:
    if adr_premium > 5:
        st.warning(f"🔥 **過熱警戒**：溢價率 > 5%，美股情緒極度亢奮。今日台股容易**開高走低**，嚴禁追價。")
    elif adr_premium > 0:
        st.success(f"🟢 **多頭順風**：溢價率為正，美股帶動台股。個股拉回成本線為安全買點。")
    elif adr_premium > -2:
        st.info(f"⚪ **整理區間**：小幅折價，市場觀望。個股表現回歸基本面與內資籌碼。")
    else:
        st.error(f"💎 **校正買點機會**：大幅負溢價 (< -2%)。若您監控的個股昨日外資是買超的，今日開低即為**錯殺買點**。")

st.divider()

# --- 次要監控：個股細節 ---
st.markdown("### 🔍 個股/ETF 驗證 (Secondary Checks)")

targets = {
    "🔥 引擎一：成長進攻": {
        "中信上游半導體 (00991A)": "00991A",
        "統一主動 (00981A)": "00981A", 
        "台積電 (2330)": "2330", 
        "弘塑 (3131)": "3131", 
        "辛耘 (3583)": "3583"
    },
    "🛡️ 引擎二：市值防禦": {
        "元大台灣50 (0050)": "0050", 
        "富邦台50 (006208)": "006208"
    },
    "💰 引擎三：高息領息": {
        "元大高股息 (0056)": "0056", 
        "國泰永續高股息 (00878)": "00878", 
        "群益台灣精選高息 (00919)": "00919",
        "復華台灣科技優息 (00929)": "00929"
    }
}

c1, c2 = st.columns(2)
with c1: cat = st.selectbox("標的分類", list(targets.keys()))
with c2: name = st.selectbox("選擇標的", list(targets[cat].keys()))
stock_id = targets[cat][name]

df, con_buy, yield_rate = monitor.get_strategic_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    
    # 根據標的屬性決定次要指標
    is_high_div = "高息" in cat
    cost_src = df['Invest_Cost'] if is_high_div else df['Foreign_Cost']
    cost_name = "投信成本" if is_high_div else "外資成本"
    bias = (latest['Close'] / cost_src.iloc[-1] - 1) * 100
    
    # 次要指標看板
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("參考：外資連買", f"{con_buy} 天")
    k2.metric(f"參考：{cost_name}", f"${cost_src.iloc[-1]:.1f}")
    k3.metric("參考：籌碼乖離", f"{bias:.2f}%")
    k4.metric("參考：RS 強度", f"{latest['RS_Index']:.2f}")

    # 整合判讀 (ADR + 次要指標)
    st.markdown("#### 📝 綜合戰略建議")
    if adr_premium < -1:
        if con_buy > 0:
            st.success(f"🎯 **執行代碼 4 (校正買點)**：ADR 雖然跌，但 {name} 昨日外資(或投信)有買。今日若開低，是**絕佳進場點**。")
        else:
            st.warning(f"⚠️ **保守觀望**：ADR 跌，且 {name} 籌碼也渙散。建議暫時避開。")
    elif adr_premium > 5:
        st.warning(f"🛑 **禁止追高**：ADR 過熱，{name} 開盤容易見高點。請等待盤中拉回成本線 (${cost_src.iloc[-1]:.1f}) 再考慮。")
    else:
        # ADR 正常，回歸個股籌碼判斷
        if bias < 2 and latest['Close'] > cost_src.iloc[-1]:
            st.info(f"✅ **順勢操作**：外部環境正常，{name} 守在成本線上，可分批佈局。")
        else:
            st.info(f"⚪ **區間盤整**：外部環境正常，但個股無明顯訊號。")

    # 簡易圖表 (只保留最核心的成本線)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="股價", line=dict(color='#1f77b4', width=3)))
    fig.add_trace(go.Scatter(x=df.index[-60:], y=cost_src.iloc[-60:], name=cost_name, line=dict(color='#d62728', dash='dot')))
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.caption("系統核心：以 TSM ADR 溢價率定調全域多空，再以個股籌碼決定進出場點位。")
