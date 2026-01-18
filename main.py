import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統環境配置 ---
st.set_page_config(page_title="2026 三引擎戰略監控 v5.1", layout="wide")

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

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id, days=150):
        # A. 價格、殖利率與國際指標 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        ticker_obj = yf.Ticker(ticker_yf)
        df = ticker_obj.history(period=f"{days}d")
        
        if df.empty: return pd.DataFrame(), 0, 0
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]

        # 估算殖利率 (近 12 個月配息 / 現價)
        try:
            divs = ticker_obj.dividends
            if divs.index.tz is not None: divs.index = divs.index.tz_localize(None)
            one_year_ago = pd.Timestamp.now() - pd.DateOffset(months=12)
            est_yield = (divs[divs.index > one_year_ago].sum() / df['Close'].iloc[-1]) * 100
        except:
            est_yield = 0

        # 國際指標
        adr = yf.Ticker("TSM").history(period=f"{days}d")
        mkt = yf.Ticker("0050.TW").history(period=f"{days}d")
        for d in [adr, mkt]: d.index = d.index.tz_localize(None).normalize()
        
        df['ADR_Premium'] = ((adr['Close'] / 5 * 32) / df['Close'] - 1) * 100
        df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100

        # B. 雙軌籌碼分析 (FinMind)
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
        
        # C. 雙軌成本線演算法
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
        
        # 連買計算
        f_net_list = df['foreign_net'].tolist()
        consecutive = 0
        for val in reversed(f_net_list):
            if val > 0: consecutive += 1
            elif val < 0: break
            
        return df, consecutive, est_yield

# --- 3. 戰情室 UI ---
st.title("🏹 2026 三引擎戰略監控系統 v5.1")

# 更新後的監控清單
targets = {
    "🔥 引擎一：成長進攻 (主動/設備)": {
        "中信上游半導體 (00991A)": "00991A",  # <--- 新增標的
        "統一主動 (00981A)": "00981A", 
        "群益精選 (00982A)": "00982A", 
        "台積電 (2330)": "2330", 
        "弘塑 (3131)": "3131", 
        "辛耘 (3583)": "3583"
    },
    "🛡️ 引擎二：市值防禦 (大盤)": {
        "元大台灣50 (0050)": "0050", 
        "富邦台50 (006208)": "006208", 
        "國泰領袖50 (00922)": "00922"
    },
    "💰 引擎三：穩健領息 (高股息)": {
        "元大高股息 (0056)": "0056", 
        "國泰永續高股息 (00878)": "00878", 
        "群益台灣精選高息 (00919)": "00919", 
        "復華台灣科技優息 (00929)": "00929"
    }
}

st.sidebar.header("🔍 戰情中心")
cat = st.sidebar.selectbox("選擇引擎", list(targets.keys()))
name = st.sidebar.selectbox("監控標的", list(targets[cat].keys()))
stock_id = targets[cat][name]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)
df, con_buy, yield_rate = monitor.get_strategic_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    
    # 智慧切換邏輯：高股息看投信，成長股看外資
    is_high_div = "高股息" in cat
    main_cost = latest['Invest_Cost'] if is_high_div else latest['Foreign_Cost']
    cost_name = "投信成本 (內資)" if is_high_div else "外資成本 (外資)"
    
    bias = (latest['Close'] / main_cost - 1) * 100
    
    # 儀表板
    st.subheader(f"{name} 戰略分析")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("當前股價", f"${latest['Close']:.2f}")
    c2.metric("主力防線", f"${main_cost:.1f}")
    c3.metric("籌碼乖離", f"{bias:.2f}%", delta_color="inverse")
    c4.metric("外資連買", f"{con_buy} 天")

    # 分頁看板
    t1, t2, t3 = st.tabs(["📊 籌碼校正", "🌌 避險監控", "📅 資金配置"])

    with t1:
        st.write(f"#### 核心防線：{cost_name}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="股價", line=dict(color='#1f77b4', width=3)))
        
        line_col = '#ff7f0e' if is_high_div else '#d62728'
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df[('Invest' if is_high_div else 'Foreign')+'_Cost'].iloc[-60:], 
                                 name=f"{cost_name}線", line=dict(color=line_col, dash='dot')))
        
        # 佈局區間提示
        fig.add_hrect(y0=main_cost*0.98, y1=main_cost*1.02, line_width=0, fillcolor="green", opacity=0.1)
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        if bias < 3 and con_buy > 0:
            st.success(f"🟢 **進場訊號**：股價貼近{cost_name}且出現連續買超。")
        elif latest['Close'] < main_cost:
            st.error(f"🔴 **警戒**：跌破{cost_name}，請注意風險。")

    with t2:
        st.subheader("深夜校正：國際盤勢連動")
        c1, c2 = st.columns(2)
        c1.metric("ADR 溢價率", f"{latest['ADR_Premium']:.2f}%")
        c2.metric("估算殖利率", f"{yield_rate:.2f}%")
        
        if latest['ADR_Premium'] < -1 and latest['foreign_net'] > 0:
             st.success("💎 **校正買點**：ADR 錯殺 + 台股外資買超。")
        elif is_high_div and yield_rate > 7:
             st.success(f"🛡️ **高息護體**：殖利率達 {yield_rate:.1f}%，具備長線保護力。")

    with t3:
        st.subheader("資金配置：相對強度 (RS)")
        rs = df['RS_Index']
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=df.index[-90:], y=rs.iloc[-90:], fill='tozeroy', name="RS vs 0050"))
        st.plotly_chart(fig_rs, use_container_width=True)
        
        if rs.iloc[-1] > 0: st.success(f"📈 **強勢**：{name} 強於大盤，建議續抱。")
        else: st.warning(f"🛡️ **弱勢**：{name} 弱於大盤，建議資金回流 0050。")

st.divider()
st.caption("2026 三引擎監控 v5.1 | 新增：中信上游半導體 (00991A)")
