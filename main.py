import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統環境配置 ---
st.set_page_config(page_title="2026 三引擎戰略監控", layout="wide")

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
        # A. 價格、國際指標與殖利率 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        ticker_obj = yf.Ticker(ticker_yf)
        df = ticker_obj.history(period=f"{days}d")
        
        if df.empty: return pd.DataFrame(), 0, 0
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]

        # 估算殖利率 (近 12 個月配息總和 / 現價)
        try:
            divs = ticker_obj.dividends
            # 篩選近一年的配息
            one_year_ago = pd.Timestamp.now() - pd.DateOffset(months=12)
            # 修正：確保時區一致或無時區
            if divs.index.tz is not None:
                divs.index = divs.index.tz_localize(None)
            
            recent_divs = divs[divs.index > one_year_ago]
            total_div = recent_divs.sum()
            current_price = df['Close'].iloc[-1]
            est_yield = (total_div / current_price) * 100 if current_price > 0 else 0
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
        
        # C. 雙軌成本線演算法 (外資 Foreign + 投信 Investment)
        # 定義通用計算函式
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
        
        # D. 外資連買計算
        f_net_list = df['foreign_net'].tolist()
        consecutive = 0
        for val in reversed(f_net_list):
            if val > 0: consecutive += 1
            elif val < 0: break
            
        return df, consecutive, est_yield

# --- 3. 戰情室 UI ---
st.title("🏹 2026 三引擎戰略監控系統")

# 三大引擎標的清單
targets = {
    "🔥 引擎一：成長進攻 (主動/半導體)": {
        "統一主動 (00981A)": "00981A", "群益精選 (00982A)": "00982A", 
        "台積電 (2330)": "2330", "弘塑 (3131)": "3131", "辛耘 (3583)": "3583"
    },
    "🛡️ 引擎二：市值防禦 (大盤)": {
        "元大台灣50 (0050)": "0050", "富邦台50 (006208)": "006208", "國泰領袖50 (00922)": "00922"
    },
    "💰 引擎三：穩健領息 (高股息)": {
        "元大高股息 (0056)": "0056", "國泰永續高股息 (00878)": "00878", 
        "群益台灣精選高息 (00919)": "00919", "復華台灣科技優息 (00929)": "00929"
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
    
    # --- 智慧切換邏輯 ---
    # 若是高股息引擎，主要參考「投信成本」；其他參考「外資成本」
    is_high_div = "高股息" in cat
    main_cost = latest['Invest_Cost'] if is_high_div else latest['Foreign_Cost']
    cost_name = "投信成本 (內資主力)" if is_high_div else "外資成本 (國際主力)"
    
    bias = (latest['Close'] / main_cost - 1) * 100
    
    # --- 儀表板顯示 ---
    st.subheader(f"{name} 戰略分析")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("當前股價", f"${latest['Close']:.2f}")
    col2.metric("主力防線", f"${main_cost:.1f}", help=f"依據 {cost_name} 計算之 20 日加權平均")
    col3.metric("籌碼乖離", f"{bias:.2f}%", delta_color="inverse")
    col4.metric("估算殖利率", f"{yield_rate:.2f}%", help="近 4 季配息總和 / 現價")

    # --- 三大時段戰略看板 ---
    tab1, tab2, tab3 = st.tabs(["📊 15:30 籌碼校正", "🌌 22:30 避險監控", "📅 季度資金配置"])

    with tab1:
        st.write(f"#### 核心指標：{cost_name}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="收盤價", line=dict(color='#1f77b4', width=3)))
        
        # 根據引擎繪製不同的成本線
        if is_high_div:
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Invest_Cost'].iloc[-60:], name="投信成本線", line=dict(color='#ff7f0e', dash='dot', width=2)))
            st.info("💡 **高息股戰略**：主要觀察**投信**動向。若跌破投信成本線且殖利率低於 5%，建議減碼。")
        else:
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Foreign_Cost'].iloc[-60:], name="外資成本線", line=dict(color='#d62728', dash='dot', width=2)))
            st.info("💡 **成長股戰略**：主要觀察**外資**動向。若連續買超且貼近成本線，為佈局良機。")

        # 佈局區間
        fig.add_hrect(y0=main_cost*0.98, y1=main_cost*1.02, line_width=0, fillcolor="green", opacity=0.1, annotation_text="主力護盤區")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("深夜校正：國際盤勢連動")
        c1, c2 = st.columns(2)
        c1.metric("ADR 溢價率", f"{latest['ADR_Premium']:.2f}%")
        
        # 避險邏輯
        if latest['ADR_Premium'] < -1 and latest['foreign_net'] > 0:
             st.success("💎 **校正買點**：ADR 錯殺 + 台股外資買超。明日開盤可留意。")
        elif is_high_div and yield_rate > 6:
             st.success("🛡️ **防禦優勢**：殖利率 > 6%，具備下檔保護力，受 ADR 波動影響較小。")
        else:
             st.info("觀察美股動向，目前無顯著避險訊號。")

    with tab3:
        st.subheader("資金配置：相對強度 (RS)")
        rs = df['RS_Index']
        curr_rs = rs.iloc[-1]
        
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=df.index[-90:], y=rs.iloc[-90:], fill='tozeroy', name="RS vs 0050"))
        st.plotly_chart(fig_rs, use_container_width=True)
        
        if is_high_div:
            if curr_rs > 0: st.success("📈 **趨勢**：高股息目前強於大盤，適合在市場震盪時增加配置。")
            else: st.warning("📉 **趨勢**：高股息目前弱於大盤，建議將資金轉往成長型引擎。")
        else:
            if curr_rs > 0: st.success("🚀 **進攻**：成長股強於大盤，建議加碼攻擊。")
            else: st.warning("🛡️ **防守**：成長股轉弱，建議回防 0050。")

st.divider()
st.caption("2026 三引擎監控 v5.0 | 成長看外資・高息看投信・避險看 ADR")
