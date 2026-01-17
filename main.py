class TaiwanStockMonitor2026:
    def __init__(self, token):
        self.api = DataLoader()
        # --- 偵錯區：確認 Token 是否有值 ---
        if not token or len(token) < 10:
            st.error("❌ Token 讀取失敗，請檢查 Secrets 設定。")
            return

        try:
            # 顯示 Token 前 10 碼（安全偵錯）
            # st.write(f"系統嘗試登入中... (Token 前綴: {token[:10]})") 
            
            if hasattr(self.api, 'login'):
                self.api.login(token=token)
            elif hasattr(self.api, 'login_token'):
                self.api.login_token(token=token)
            
            st.toast("✅ FinMind 登入成功！", icon="🚀")
        except Exception as e:
            st.sidebar.error(f"登入異常：{e}")
