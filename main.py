import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統環境配置 ---
st.set_page_config(page_title="2026 三引擎戰略系統 v6.4 (全配版)", layout="wide")

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
        # A. 價格與技術指標 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df = yf.Ticker(ticker_yf).history(period=f"{days}d")
        
        if df.empty: return pd.DataFrame(), 0, 0, "無數據", 0, 0, 0
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep='last')]

        # 指標 1: KD (9,3,3)
        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        
        # 指標 2: 量比 (Vol Ratio)
        vol_ma20 = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / vol_ma20

        # 指標 3: 殖利率
        try:
            divs = yf.Ticker(ticker_yf).dividends
            if divs.index.tz is not None: divs.index = divs.index.tz_localize(None)
            one_year_ago = pd.Timestamp.now() - pd.DateOffset(months=12)
            est_yield = (divs[divs.index > one_year_ago].sum() / df['Close'].iloc[-1]) * 100
        except:
            est_yield = 0

        # 指標 4: RS 相對強度
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
        
        # 指標 5: 籌碼集中度 (Concentration)
        # 公式：(外資買賣超 + 投信買賣超) / 當日成交量 * 100
        df['Concentration'] = (df['foreign_net'] + df['investment_net']) / df['Volume'] * 100

        # C. 智慧成本線 (VWAP)
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
        
        # 決定主要防守線與主要法人
        if i_has and not f_has: # 僅有投信 (高股息)
            used_source = "投信成本"
            main_net = df['investment_net']
        elif not i_has and f_has: # 僅有外資
            used_source = "外資成本"
            main_net = df['foreign_net']
        else: # 兩者皆有或皆無，預設外資 (除非是高股息ETF在外部邏輯會覆蓋)
            used_source = "外資成本" 
            main_net = df['foreign_net']
        
        # 指標 6: 連續買賣超天數 (Consecutive Days)
        # 正值=連買, 負值=連賣
        net_list = main_net.tolist()
        consecutive = 0
        if net_list:
            last_val = net_list[-1]
            if last_val > 0: # 檢查連買
                for val in reversed(net_list):
                    if val > 0: consecutive += 1
                    else: break
            elif last_val < 0: # 檢查連賣
                for val in reversed(net_list):
                    if val < 0: consecutive -= 1
                    else: break
        
        # 回傳最新數據
        k_val = df['K'].iloc[-1]
        vol_r = df['Vol_Ratio'].iloc[-1]
        conc_val = df['Concentration'].iloc[-1]
            
        return df, consecutive, est_yield, used_source, k_val, vol_r, conc_val

# --- 3. UI 介面 ---
st.title("🦅 2026 三引擎戰略系統 v6.4 (全配版)")

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# 1. ADR 儀表板
st.markdown("### 🌎 全球戰略風向 (TSM ADR)")
adr_premium, adr_price = monitor.get_global_tsm_signal()
c_m, c_i = st.columns([1, 2])
with c_m:
    d_c = "inverse" if adr_premium > 5 else ("off" if adr_premium < 0 else "normal")
    st.metric("TSM ADR 溢價率", f"{adr_premium:.2f}%", f"美股 ${adr_price:.2f}", delta_color=d_c)
with c_i:
    if adr_premium > 5: st.warning("🔥 **過熱**：嚴禁追價，留意開高走低。")
    elif adr_premium < -2: st.error("💎 **校正**：負溢價錯殺，留意開低買點。")
    else: st.info("🟢 **正常**：回歸個股籌碼與技術面判斷。")

st.divider()

# 2. 標的選擇
st.markdown("### 🔍 標的驗證 (價・量・籌・勢)")
targets = {
    "🔥 引擎一：成長進攻": {
        "台積電 (2330)": "2330",
        "中信上游半導體 (00991A)": "00991A",
        "統一主動 (00981A)": "00981A", 
        "群益精選 (00982A)": "00982A",
        "復華好收益 (00980A)": "00980A"
    },
    "🛡️ 引擎二：市值防禦": {
        "元大台灣50 (0050)": "0050", 
        "富邦台50 (006208)": "006208",
        "國泰領袖50 (00922)": "00922"
    },
    "💰 引擎三：穩健領息": {
        "元大高股息 (0056)": "0056", 
        "國泰永續高股息 (00878)": "00878", 
        "群益台灣精選高息 (00919)": "00919",
        "復華台灣科技優息 (00929)": "00929"
    }
}

c1, c2 = st.columns(2)
with c1: cat = st.selectbox("引擎分類", list(targets.keys()))
with c2: name = st.selectbox("監控標的", list(targets[cat].keys()))
stock_id = targets[cat][name]

df, con_days, yield_rate, source_name, k_val, vol_r, conc_val = monitor.get_strategic_data(stock_id)

if not df.empty:
    latest = df.iloc[-1]
    
    # 決定成本線
    is_high_div = "高股息" in cat or "穩健領息" in cat
    # 若為高息股且投信有數據，優先用投信
    if is_high_div and "投信" in source_name:
        main_cost = latest['Invest_Cost']
        cost_label = "投信成本"
    elif is_high_div: # 高息但無投信數據，用外資備援
        main_cost = latest['Foreign_Cost']
        cost_label = "外資成本 (備援)"
    else: # 成長/市值，優先用外資
        main_cost = latest['Foreign_Cost']
        cost_label = "外資成本"

    bias = (latest['Close'] / main_cost - 1) * 100
    
    # --- 關鍵指標儀表板 (重新排列) ---
    # Row 1: 價格與技術面
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("當前股價", f"${latest['Close']:.2f}")
    k2.metric("量比 (攻擊力)", f"{vol_r:.2f}倍", delta="攻擊" if vol_r > 1.2 else "溫和")
    k3.metric("KD 值 (位階)", f"{k_val:.0f}", delta="過熱" if k_val > 80 else "低檔", delta_color="inverse")
    k4.metric("RS 強度 (vs 0050)", f"{latest['RS_Index']:.2f}")

    # Row 2: 籌碼面 (補回關鍵指標)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{cost_label}", f"${main_cost:.1f}", help="主力 20 日平均持倉成本")
    c2.metric("籌碼乖離", f"{bias:.2f}%", delta="安全" if bias < 5 else "風險", delta_color="inverse")
    
    # 連續買賣超：正數為連買，負數為連賣
    con_label = f"連買 {con_days} 天" if con_days > 0 else f"連賣 {abs(con_days)} 天"
    con_delta = "主力進場" if con_days >= 3 else ("主力出貨" if con_days <= -3 else "中性")
    con_color = "normal" if con_days > 0 else "inverse"
    c3.metric("主力連續動向", con_label, delta=con_delta, delta_color=con_color)
    
    # 籌碼集中度
    conc_delta = "大戶收集" if conc_val > 5 else ("籌碼渙散" if conc_val < 0 else None)
    c4.metric("籌碼集中度", f"{conc_val:.2f}%", delta=conc_delta, help="(外資+投信買賣超)/成交量。正值越高代表籌碼越集中。")

    # 綜合戰略判讀
    st.markdown("#### 📝 最終戰略判讀")
    
    # 基準標的
    if stock_id in ["0050", "006208"]:
        st.info("ℹ️ **基準標的**：大盤觀測基準。")
    
    # 1. 賣出訊號 (連賣 + 破線 + 集中度負)
    elif con_days <= -3 and latest['Close'] < main_cost:
        st.error(f"🔴 **主力出貨警報**：股價跌破成本線，且主力已{con_label}。籌碼集中度 ({conc_val:.2f}%) 不佳，建議離場。")
    
    # 2. 假突破過濾 (漲但沒量/沒籌碼)
    elif latest['Close'] > main_cost and conc_val < 0 and vol_r < 0.8:
        st.warning(f"⚠️ **虛漲背離**：股價上漲但籌碼集中度為負，且量能不足。小心假突破。")
        
    # 3. 買進訊號 (連買 + 守線 + 集中度正)
    elif con_days >= 3 and bias < 5 and conc_val > 0:
        st.success(f"🚀 **真金白銀**：主力{con_label}，且籌碼集中度翻正。股價貼近成本線，為穩健買點。")
    
    # 4. 校正買點
    elif adr_premium < -1 and con_days > 0:
        st.success(f"💎 **校正買點**：ADR 錯殺，但台股主力仍在買方。留意開低後的機會。")
        
    else:
        st.info(f"⚪ **區間震盪**：多空力道均衡，等待進一步訊號。")

    # 核心圖表 (雙軸：價格+成本 / 成交量)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="股價", line=dict(color='#1f77b4', width=3)))
    
    line_col = '#ff7f0e' if is_high_div else '#d62728'
    cost_series = df['Invest_Cost'] if is_high_div else df['Foreign_Cost']
    fig.add_trace(go.Scatter(x=df.index[-60:], y=cost_series.iloc[-60:], name=cost_label, line=dict(color=line_col, dash='dot')))
    
    fig.add_trace(go.Bar(x=df.index[-60:], y=df['Volume'].iloc[-60:], name="成交量", marker_color='rgba(255, 255, 255, 0.3)', yaxis='y2'))
    
    fig.update_layout(
        template="plotly_dark", height=400, margin=dict(t=30, b=20),
        yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False)
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption("v6.4 終極全配版：補回「籌碼集中度」與「主力連賣」偵測，徹底過濾假突破。")
