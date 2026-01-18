import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統環境配置 ---
st.set_page_config(page_title="2026 三引擎戰略系統 v6.5 (開盤決斷版)", layout="wide")

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

    @st.cache_data(ttl=600)
    def get_global_tsm_signal(_self):
        """全域 TSM ADR 訊號"""
        try:
            tsm_adr = yf.Ticker("TSM").history(period="5d")
            tsm_tw = yf.Ticker("2330.TW").history(period="5d")
            tsm_adr.index = tsm_adr.index.tz_localize(None).normalize()
            tsm_tw.index = tsm_tw.index.tz_localize(None).normalize()
            
            adr_close = tsm_adr['Close'].iloc[-1]
            tw_close = tsm_tw['Close'].iloc[-1]
            fx_rate = 32.5 
            
            implied_price = (adr_close * fx_rate) / 5
            premium = ((implied_price / tw_close) - 1) * 100
            return premium, adr_close
        except:
            return 0, 0

    @st.cache_data(ttl=3600)
    def get_strategic_data(_self, stock_id, days=150):
        # A. 歷史數據 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df = yf.Ticker(ticker_yf).history(period=f"{days}d")
        
        if df.empty: return pd.DataFrame(), 0, 0, "無數據", 0, 0, 0, 0
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]

        # 技術指標
        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        
        # 20日均量 (用於計算開盤量比)
        vol_ma20 = df['Volume'].rolling(20).mean()
        avg_vol = vol_ma20.iloc[-1]

        # 殖利率
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

        # B. 籌碼數據 (FinMind)
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
        df['Concentration'] = (df['foreign_net'] + df['investment_net']) / df['Volume'] * 100

        # C. 智慧成本線
        def calculate_vwap_safe(net_buy_col):
            costs = []
            last_valid = np.nan
            has_data = False
            for i in range(len(df)):
                win = df.iloc[max(0, i-19) : i+1]
                buys = win[win[net_buy_col] > 0]
                if not buys.empty:
                    val = (buys['Close'] * buys[net_buy_col]).sum() / buys[net_buy_col].sum()
                    last_valid = val
                    has_data = True
                costs.append(last_valid)
            return pd.Series(costs, index=df.index).ffill().bfill(), has_data

        f_cost, f_has = calculate_vwap_safe('foreign_net')
        i_cost, i_has = calculate_vwap_safe('investment_net')
        
        df['Foreign_Cost'] = f_cost
        df['Invest_Cost'] = i_cost
        
        if i_has and not f_has: used_source = "投信成本"; main_net = df['investment_net']
        elif not i_has and f_has: used_source = "外資成本"; main_net = df['foreign_net']
        else: used_source = "外資成本"; main_net = df['foreign_net']
        
        # 連買天數
        net_list = main_net.tolist()
        consecutive = 0
        if net_list:
            last_val = net_list[-1]
            if last_val > 0:
                for val in reversed(net_list):
                    if val > 0: consecutive += 1
                    else: break
            elif last_val < 0:
                for val in reversed(net_list):
                    if val < 0: consecutive -= 1
                    else: break
        
        return df, consecutive, est_yield, used_source, df['K'].iloc[-1], df['Concentration'].iloc[-1], avg_vol

    # 新增：即時開盤數據抓取 (不透過 Cache，確保 09:05 是最新的)
    def get_realtime_open(self, stock_id):
        try:
            info = yf.Ticker(f"{stock_id}.TW").fast_info
            # 嘗試獲取今日開盤價，若盤前可能為 None
            open_p = info.open if info.open else info.last_price
            last_p = info.last_price
            # 簡易估算當日量 (若盤中則直接用 last_volume，開盤初期需自行判斷)
            curr_vol = info.last_volume 
            return open_p, last_p, curr_vol
        except:
            return 0, 0, 0

# --- 3. UI 介面 ---
st.title("🦅 2026 三引擎戰略系統 v6.5 (開盤決斷版)")

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# ADR 儀表板
st.markdown("### 🌎 全球戰略風向 (TSM ADR)")
adr_premium, adr_price = monitor.get_global_tsm_signal()
c_m, c_i = st.columns([1, 2])
with c_m:
    d_c = "inverse" if adr_premium > 5 else ("off" if adr_premium < 0 else "normal")
    st.metric("TSM ADR 溢價率", f"{adr_premium:.2f}%", f"美股 ${adr_price:.2f}", delta_color=d_c)
with c_i:
    if adr_premium > 5: st.warning("🔥 **過熱**：美股過熱，台股易開高走低。")
    elif adr_premium < -2: st.error("💎 **校正**：負溢價錯殺，留意開低買點。")
    else: st.info("🟢 **正常**：回歸個股籌碼與技術面判斷。")

st.divider()

# 標的選擇
targets = {
    "🔥 引擎一：成長進攻": {"台積電 (2330)": "2330", "中信上游半導體 (00991A)": "00991A", "統一主動 (00981A)": "00981A", "群益精選 (00982A)": "00982A", "復華好收益 (00980A)": "00980A"},
    "🛡️ 引擎二：市值防禦": {"元大台灣50 (0050)": "0050", "富邦台50 (006208)": "006208", "國泰領袖50 (00922)": "00922"},
    "💰 引擎三：穩健領息": {"元大高股息 (0056)": "0056", "國泰永續高股息 (00878)": "00878", "群益台灣精選高息 (00919)": "00919", "復華台灣科技優息 (00929)": "00929"}
}

c1, c2 = st.columns(2)
with c1: cat = st.selectbox("引擎分類", list(targets.keys()))
with c2: name = st.selectbox("監控標的", list(targets[cat].keys()))
stock_id = targets[cat][name]

df, con_days, yield_rate, source_name, k_val, conc_val, avg_vol_20 = monitor.get_strategic_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    
    # 決定成本線
    is_high_div = "高股息" in cat or "穩健領息" in cat
    if is_high_div and "投信" in source_name: main_cost = latest['Invest_Cost']; cost_label = "投信成本"
    elif is_high_div: main_cost = latest['Foreign_Cost']; cost_label = "外資成本 (備援)"
    else: main_cost = latest['Foreign_Cost']; cost_label = "外資成本"

    bias = (latest['Close'] / main_cost - 1) * 100
    
    # 獲取即時開盤數據 (09:05 用)
    real_open, real_last, real_vol = monitor.get_realtime_open(stock_id)
    # 預估量比 (簡單推估：若現在是 09:30，量已達均量 30%，全天可能爆量。此處使用即時量/20日均量作為參考)
    # 實戰中：開盤 15 分鐘量 > 20日均量 * 0.15 視為有量
    real_vol_ratio = real_vol / avg_vol_20 
    
    # --- 三大時段戰略看板 (晚 -> 夜 -> 早) ---
    tab1, tab2, tab3 = st.tabs(["📊 15:30 盤後篩選", "🌌 22:30 深夜校正", "☀️ 09:05 開盤執行"])

    with tab1:
        st.subheader("分析期：尋找「準買入名單」")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"防線: {cost_label}", f"${main_cost:.1f}")
        c2.metric("籌碼乖離", f"{bias:.2f}%", delta="安全" if bias < 5 else "過熱", delta_color="inverse")
        
        con_label = f"連買 {con_days} 天" if con_days > 0 else f"連賣 {abs(con_days)} 天"
        con_color = "normal" if con_days > 0 else "inverse"
        c3.metric("主力連續動向", con_label, delta="主力進場" if con_days>=3 else "主力撤退", delta_color=con_color)
        c4.metric("籌碼集中度", f"{conc_val:.2f}%", delta="大戶收貨" if conc_val > 0 else "散戶接刀")
        
        # 核心圖表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="股價", line=dict(color='#1f77b4', width=3)))
        line_col = '#ff7f0e' if is_high_div else '#d62728'
        cost_series = df['Invest_Cost'] if is_high_div else df['Foreign_Cost']
        fig.add_trace(go.Scatter(x=df.index[-60:], y=cost_series.iloc[-60:], name=cost_label, line=dict(color=line_col, dash='dot')))
        fig.update_layout(template="plotly_dark", height=350, margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("校正期：ADR 避險監控")
        k1, k2, k3 = st.columns(3)
        k1.metric("ADR 溢價率", f"{adr_premium:.2f}%")
        k2.metric("KD 位階", f"{k_val:.0f}", help="<20 超賣, >80 過熱")
        k3.metric("RS 強度", f"{latest['RS_Index']:.2f}", help=">0 強於 0050")
        
        if adr_premium < -1 and con_days > 0:
            st.success("💎 **校正訊號**：ADR 跌 + 台股主力買。明日開低為「黃金買點」。")
        elif adr_premium > 5:
            st.warning("🔥 **過熱訊號**：ADR 溢價過大，明日開高容易拉回，切勿追價。")
        else:
            st.info("⚪ **中性訊號**：無特殊國際盤影響，回歸 Tab 3 開盤條件判斷。")

    with tab3:
        st.subheader("決斷期：09:05 執行指令")
        
        # 顯示即時數據
        m1, m2, m3 = st.columns(3)
        m1.metric("1. 今日開盤價", f"${real_open:.2f}")
        m2.metric("2. 主力防線 (成本)", f"${main_cost:.1f}")
        m3.metric("3. 即時量比 (Vol Ratio)", f"{real_vol_ratio:.2f}", help="數值持續上升代表有量")

        st.markdown("---")
        st.markdown("#### ⚔️ 交易執行腳本")
        
        # 條件判斷邏輯
        cond_price = real_open > main_cost # 條件1: 開在成本之上
        cond_vol = real_vol_ratio > 0.1 # 條件2: 開盤有量 (09:05 若達日均量 10% 算不錯)
        # 註：這裡的 0.1 是開盤瞬間的經驗值，若盤中請調整為 > 1.0
        
        if cond_price:
            st.success(f"✅ **條件 A (價) 符合**：開盤價 (${real_open}) 守在 {cost_label} 之上。")
        else:
            st.error(f"❌ **條件 A (價) 破局**：開盤跌破 {cost_label}，防線失守。")

        if cond_vol: # 僅作提示，非絕對
            st.info(f"ℹ️ **條件 B (量) 觀察**：開盤量能正常。請持續觀察量比是否放大至 > 1.0。")
        
        # 最終指令輸出 (LaTeX 格式)
        st.markdown("##### 📢 系統建議：")
        
        if cond_price:
            st.markdown(r"""
            > **符合進場條件** $\rightarrow$ **果斷買進 (分批 3 筆)**
            > * 第一筆：開盤確認支撐後。
            > * 第二筆：盤中突破早盤高點時。
            > * 第三筆：尾盤確認收紅時。
            """)
        else:
            st.markdown(r"""
            > **跌破成本線或量能急凍** $\rightarrow$ **取消交易，觀望**
            > * 目前股價由主力防線之下開出，多方棄守。
            > * 建議：等待股價重新站回成本線且量能放大後再重新評估。
            """)

st.caption("v6.5 最終版：依據 09:05 即時開盤價與成本線乖離，自動生成進出場指令。")
