import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統環境配置 ---
st.set_page_config(page_title="2026 雙核全功能戰旗版", layout="wide")

# 安全讀取 Token
if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN，請檢查 Secrets 設定。")
    st.stop()
else:
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]

# --- 2. 核心戰略運算引擎 ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        self.api = DataLoader()
        # 終極登入補丁：自動適配所有 FinMind 版本
        try:
            if hasattr(self.api, 'login'): self.api.login(token=token.strip())
            elif hasattr(self.api, 'login_token'): self.api.login_token(token=token.strip())
            else: self.api.token = token.strip()
        except: pass

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id, days=150):
        # A. 價格與國際避險指標 (yfinance)
        # 1. 台股標的
        ticker_yf = f"{stock_id}.TW"
        df = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df.empty: return pd.DataFrame(), 0
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')] # 去除重複索引

        # 2. 國際指標 (ADR & EWT)
        adr = yf.Ticker("TSM").history(period=f"{days}d")
        ewt = yf.Ticker("EWT").history(period=f"{days}d") # MSCI 台灣 ETF (夜盤代理)
        mkt = yf.Ticker("0050.TW").history(period=f"{days}d") # 大盤基準
        
        # 統一索引時區
        for d in [adr, ewt, mkt]:
            d.index = d.index.tz_localize(None).normalize()
        
        # 計算指標
        # ADR 溢價率：(ADR / 5 * 32) / 台積電現價 - 1
        # 註：此處以 32 為固定匯率估算，實戰可接匯率 API
        df['ADR_Premium'] = ((adr['Close'] / 5 * 32) / df['Close'] - 1) * 100
        df['Night_Proxy'] = ewt['Close'].pct_change() * 100 
        
        # 相對強度 (RS)：標的 20 日漲幅 - 0050 20 日漲幅
        df['RS_Index'] = (df['Close'].pct_change(20) - mkt['Close'].pct_change(20)) * 100

        # B. 籌碼深度分析 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            # 分別提取外資與投信
            for name in ['Foreign', 'Investment']:
                sub = df_chip[df_chip['name'].str.contains(name, case=False)].copy()
                sub['date'] = pd.to_datetime(sub['date'])
                # 解決 InvalidIndexError：加總同一天的數據
                sub = sub.set_index('date').groupby(level=0).agg({'buy':'sum', 'sell':'sum'})
                df[f'{name.lower()}_net'] = sub['buy'] - sub['sell']
        except:
            df['foreign_net'] = 0
            df['investment_net'] = 0

        df = df.fillna(0)
        
        # C. 腳本 2：外資成本線 (VWAP - 永不中斷版)
        costs = []
        last_valid_cost = np.nan # 記憶變數
        
        for i in range(len(df)):
            win = df.iloc[max(0, i-19) : i+1]
            buys = win[win['foreign_net'] > 0] # 只看買進的日子
            
            if not buys.empty:
                # 加權平均公式
                current_cost = (buys['Close'] * buys['foreign_net']).sum() / buys['foreign_net'].sum()
                last_valid_cost = current_cost
            
            # 若無買進，沿用上一次成本 (或用當日均價填充)
            costs.append(last_valid_cost)
        
        df['Foreign_Cost'] = costs
        df['Foreign_Cost'] = df['Foreign_Cost'].ffill().bfill() # 雙向填充確保無斷線
        
        # D. 腳本 1：外資連買計數器
        f_net_list = df['foreign_net'].tolist()
        consecutive = 0
        for val in reversed(f_net_list):
            if val > 0: consecutive += 1
            elif val < 0: break
            
        return df, consecutive

# --- 3. 戰情室 UI 介面 ---
st.title("🏹 2026 雙核戰略：全功能戰旗版")

# 標的清單 (含代號)
targets = {
    "核心戰略 (權值)": {"台積電 (2330)": "2330", "元大台灣50 (0050)": "0050", "富邦台50 (006208)": "006208"},
    "主動攻擊 (成長)": {"統一主動 (00981A)": "00981A", "群益精選 (00982A)": "00982A", "復華主動 (00980A)": "00980A"},
    "供應鏈 (設備)": {"弘塑 (3131)": "3131", "辛耘 (3583)": "3583", "萬潤 (6187)": "6187"}
}

st.sidebar.header("🔍 戰情監控中心")
cat = st.sidebar.selectbox("戰略位置", list(targets.keys()))
name = st.sidebar.selectbox("監控標的", list(targets[cat].keys()))
stock_id = targets[cat][name]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)
df, con_buy = monitor.get_strategic_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    f_cost = latest['Foreign_Cost']
    bias = (latest['Close'] / f_cost - 1) * 100
    
    # --- 三大時段戰略看板 ---
    tab1, tab2, tab3 = st.tabs(["📊 15:30 盤後分析 (籌碼校正)", "🌌 22:30 深夜監控 (即時避險)", "📅 每季策略審視 (資金配置)"])

    # --- 1. 盤後分析期 ---
    with tab1:
        st.subheader("執行腳本 1 & 2：籌碼過濾與成本確認")
        c1, c2, c3 = st.columns(3)
        c1.metric("外資連續買超", f"{con_buy} 天", delta="強勢" if con_buy >=3 else "觀察")
        c2.metric("股價 / 外資成本", f"{latest['Close']:.1f} / {f_cost:.1f}")
        c3.metric("籌碼乖離率", f"{bias:.2f}%", delta="安全區" if bias < 5 else "過熱", delta_color="inverse")
        
        # 繪圖：價格 vs 成本線
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="收盤價", line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Foreign_Cost'].iloc[-60:], name="外資成本線", line=dict(color='#d62728', dash='dot', width=2)))
        
        # 標示綠燈佈局區
        fig.add_hrect(y0=f_cost*0.98, y1=f_cost*1.02, line_width=0, fillcolor="green", opacity=0.1, annotation_text="佈局區")
        
        fig.update_layout(template="plotly_dark", height=450, title=f"{name} 外資成本防線圖")
        st.plotly_chart(fig, use_container_width=True)
        
        if latest['Close'] < f_cost:
            st.error(f"🔴 **警戒**：股價已跌破外資成本線 ${f_cost:.1f}。若投信同步賣超，請執行減碼。")
        elif bias < 3 and con_buy > 0:
            st.success("🟢 **機會**：外資買超且股價貼近成本，符合「連續性」與「低乖離」條件。")

    # --- 2. 深夜監控期 ---
    with tab2:
        st.subheader("執行腳本 4：ADR 與 夜盤避險")
        cc1, cc2 = st.columns(2)
        cc1.metric("ADR 溢價率 (TSM)", f"{latest['ADR_Premium']:.2f}%")
        cc2.metric("夜盤代理 (EWT)", f"{latest['Night_Proxy']:.2f}%")
        
        st.markdown("#### ⚖️ 校正回檔買點偵測")
        # 邏輯：ADR 跌 (市場情緒差) 但 台股外資買 (實質籌碼好) = 錯殺機會
        if latest['ADR_Premium'] < -1 and latest['foreign_net'] > 0:
            st.success(f"💎 **校正買點觸發**：ADR 負溢價，但今日台股外資買超 {int(latest['foreign_net']/1000)} 張。明早開盤若不破平盤，為絕佳防禦性買點。")
        elif latest['ADR_Premium'] > 2:
            st.warning("⚠️ **追高風險**：ADR 溢價過高，明早台股容易開高走低，不宜追價。")
        else:
            st.info("⚪ 目前國際盤勢正常，無特殊訊號。")

    # --- 3. 每季審視期 ---
    with tab3:
        st.subheader("執行腳本 3：市值型 vs. 主動型 績效追蹤")
        st.write("利用 **RS (相對強度) 指標** 決定資金流向：")
        
        # RS 指標圖
        rs = df['RS_Index']
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=df.index[-90:], y=rs.iloc[-90:], fill='tozeroy', 
                                    line=dict(color='orange'), name="RS Index"))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="white")
        fig_rs.update_layout(template="plotly_dark", height=350, title="相對強度 (正值 = 強於 0050)")
        st.plotly_chart(fig_rs, use_container_width=True)
        
        curr_rs = rs.iloc[-1]
        c_str = f"{curr_rs:.2f}"
        
        col_a, col_b = st.columns([1, 2])
        col_a.metric("當前 RS 值", c_str, delta="強於大盤" if curr_rs > 0 else "弱於大盤")
        
        with col_b:
            if curr_rs > 0:
                st.success(f"📈 **進攻模式**：{name} 目前強於 0050。建議：**增持主動型 ETF / 個股**，放大 2026 超額報酬。")
            else:
                st.warning(f"🛡️ **防禦模式**：{name} 目前弱於 0050。建議：**資金回流市值型 ETF**，規避風險。")

else:
    st.info("數據載入中... 請稍候或檢查 Token。")

st.divider()
st.caption("2026 戰旗系統 v4.0 | 核心原則：晚上篩選真外資，深夜校正錯殺點，季末汰弱留強。")
