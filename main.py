# --- 2. 核心分析類別 (加入診斷邏輯) ---
class TaiwanStockMonitor2026:
    def __init__(self, token):
        # 建立側邊欄診斷區
        with st.sidebar.expander("🛠️ 系統診斷資訊", expanded=True):
            # A. 檢查 Token 是否存在
            if not token:
                st.error("❌ Secrets 中未偵測到 FINMIND_TOKEN")
                self.login_status = False
            else:
                # 顯示 Token 前 6 碼以資識別 (其餘遮蔽)
                st.write(f"🔑 Token 前綴: `{token[:6]}...`")
                self.login_status = True

            # B. 檢查套件版本
            import FinMind
            st.write(f"📦 FinMind 版本: `{FinMind.__version__}`")
            
            self.api = DataLoader()
            
            # C. 嘗試登入並補獲錯誤
            try:
                if hasattr(self.api, 'login'):
                    self.api.login(token=token)
                    st.success("✅ 成功呼叫 login 指令")
                elif hasattr(self.api, 'login_token'):
                    self.api.login_token(token=token)
                    st.success("✅ 成功呼叫 login_token 指令")
                else:
                    st.warning("⚠️ 找不到任何登入指令")
            except Exception as e:
                st.error(f"❌ 登入過程報錯: {e}")

    @st.cache_data(ttl=3600)
    def get_full_analysis_data(_self, stock_id, days=60):
        # 原有的數據抓取邏輯...
        # [此處保留之前完整代碼中的內容]
        pass

# --- 3. 系統初始化與執行 ---
# 增加一個更嚴格的 Secrets 檢查
if "FINMIND_TOKEN" not in st.secrets:
    st.error("⚠️ 系統偵測不到 Secrets 設定。請確認您已在 Streamlit Cloud 的 'Advanced settings' -> 'Secrets' 中貼入 `FINMIND_TOKEN = '您的金鑰'`")
    st.stop()
else:
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]

monitor = TaiwanStockMonitor2026(FINMIND_TOKEN)
