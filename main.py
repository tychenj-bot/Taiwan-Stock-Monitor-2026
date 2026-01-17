import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 系統設定與頁面配置 ---
st.set_page_config(page_title="2026 台股雙核監控系統", layout="wide")

# 從 Streamlit Secrets 安全讀取 Token
if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 找不到 FINMIND_TOKEN，請前往 Streamlit Cloud 的 Settings -> Secrets 進行設定。")
    st.stop()
else:
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]

# --- 2. 核心分析類別 ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        self.api = DataLoader()
        self.login_status = False
        
        # 側邊欄診斷區：找出「登入失敗」的真實原因
        with st.sidebar.expander("🛠️ 系統診斷資訊", expanded=True):
            clean_token = token.strip() # 去除隱藏空白
            st.write(f"🔑 Token 前綴: `{clean_token[:15]}...`")
            
            import FinMind
            st.write(f"📦 FinMind 版本: `{FinMind.__version__}`")
            
            try:
                # 優先嘗試最新版指令
                self.api.login(token=clean_token)
                st.success("✅ 成功呼叫 login 指令")
                self.login_status = True
            except Exception as e:
                # 捕獲並顯示原始錯誤訊息
                error_msg = str(e)
                st.error(f"❌ 伺服器拒絕登入。原因：{error_msg}")
                
                # 自動判斷常見錯誤
                if "Unauthorized" in error_msg:
                    st.info("💡 提示：Token 可能已過期，請至 FinMind 官網重新產生。")
                elif "Invalid" in error_msg:
                    st.info("💡 提示：Token 格式不正確，請檢查 Secrets 是否包含多餘引號。")
                
                # 嘗試舊版指令作為最後防線
                try:
                    self.api.login_token(token=clean_token)
                    st.success("✅ 成功使用 login_token 指令")
                    self.login_status = True
                except:
                    pass

    @st.cache_data(ttl=3600)
    def get_full_analysis_data(_self, stock_id, days=60):
        # A. 價格數據 (yfinance)
        ticker_yf = f"{stock_id}.TW"
        df_price = yf.Ticker(ticker_yf).history(period=f"{days}d")
        if df_price.empty:
            return pd.DataFrame()
        df_price.index = df_price.index.tz_localize(None).normalize()

        # B. 籌碼數據 (FinMind)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df_chip = _self.api.taiwan_stock_institutional_investors(
                data_id=stock_id,
                start_date=start_date
            )
            # 過濾外資數據 (包含大小寫相容處理)
            df_foreign = df_chip[df_chip['name'].str.contains('Foreign', case=False, na=False)].copy()
            df_foreign['date'] = pd.to_datetime(df_foreign['date'])
            df_foreign = df_foreign.set_index('date')
        except Exception:
            return df_price # 失敗則回傳僅有價格的數據

        # C. 計算外資加權成本線
        combined = pd.concat([df_price, df_foreign[['net_buy']]], axis=1).dropna(subset=['Close'])
        
        def get_weighted_cost(window_df):
            buys = window_df[window_df['net_buy'] > 0]
            if buys.empty: return np.nan
            return (buys['Close'] * buys['net_buy']).sum() / buys['net_buy'].sum()

        costs = []
        window = 20
        for i in range(len(combined)):
            if i < window: costs.append(np.nan)
            else:
                win = combined.iloc[i-window+1 : i+1]
                costs.append(get_weighted_cost(win))
        
        combined['Foreign_Cost_Line'] = costs
        combined['Foreign_Cost_Line'] = combined['Foreign_Cost_Line'].ffill()
        return combined

    def get_realtime_signal(self, stock_id):
        try:
            ticker = yf.Ticker(f"{stock_id}.TW")
            fast = ticker.fast_info
            last, open_p, prev_c = fast.last_price, fast.open, fast.previous_close
            if last > open_p and open_p > prev_c: signal = "🟢 強勢多頭 (開高走高)"
            elif last < open_p: signal = "🟡 留意回檔 (開高走低)"
            else: signal = "⚪ 震盪整理"
            return last, round((open_p/prev_c-1)*100, 2), signal
        except:
            return 0.0, 0.0, "數據讀取中..."

# --- 3. UI 介面 ---
st.title("🚀 2026 台股雙核監控系統")
st.markdown("---")

stock_options = {
    "台積電 (2330)": "2330",
    "元大台灣50 (0050)": "0050",
    "富邦台50 (006208)": "006208",
    "國泰領袖50 (00922)": "00922",
    "統一台股 (主動型)": "00981A",
    "群益精選 (主動型)": "00982A"
}
target_name = st.sidebar.selectbox("🎯 選擇監控標的", list(stock_options.keys()))
target_id = stock_options[target_name]

# 初始化監控器
monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)

# A. 即時行情 (yfinance)
last, gap, sig = monitor.get_realtime_signal(target_id)
c1, c2, c3 = st.columns(3)
c1.metric("當前成交價", f"${last:.2f}")
c2.metric("開盤漲幅 %", f"{gap}%")
c3.info(f"盤中訊號：{sig}")

# B. 成本線圖表 (FinMind + Plotly)
st.divider()
st.subheader("📊 外資加權成本分析")

with st.spinner("正在對接 FinMind 獲取籌碼數據..."):
    df = monitor.get_full_analysis_data(target_id)
    if not df.empty and 'Foreign_Cost_Line' in df.columns:
        latest = df.iloc[-1]
        f_cost = latest['Foreign_Cost_Line']
        bias = (latest['Close'] / f_cost - 1) * 100 if f_cost > 0 else 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="日 K 收盤價", line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Foreign_Cost_Line'], name="外資 20 日成本線", line=dict(color="#d62728", dash='dot', width=2)))
        
        fig.update_layout(template="plotly_dark", height=500, xaxis_title="日期", yaxis_title="價格", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # 根據乖離率給予顏色建議
        if bias < 3:
            st.success(f"✅ 股價距離外資成本僅 **{bias:.2f}%** (成本價: {f_cost:.2f})。法人防守區，適合佈局。")
        elif bias > 10:
            st.warning(f"⚠️ 乖離率達 **{bias:.2f}%**。短線過熱，建議等回測成本線再進場。")
        else:
            st.info(f"💡 目前乖離率為 **{bias:.2f}%**。趨勢觀察中。")
    else:
        st.warning("⚠️ 籌碼數據載入中或 Token 無法獲取完整權限，僅顯示價格趨勢。")

# C. 2026 戰略指引
st.divider()
month = datetime.now().month
q = (month-1)//3 + 1
st.success(f"📅 2026 Q{q} 戰略：當前月份建議檢視供應鏈資本支出，並關注外資成本線的支撐力道。")
