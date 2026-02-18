import os
import requests
import json

class HantuTrader:
    def __init__(self):
        self.app_key = os.environ.get('HANTU_APP_KEY')
        self.secret_key = os.environ.get('HANTU_SECRET_KEY')
        self.acc_no = os.environ.get('HANTU_ACCOUNT_NO')
        self.acc_proc = os.environ.get('HANTU_ACCOUNT_PROC')
        self.base_url = "https://openapivts.koreainvestment.com:29443" # 모의투자용 URL
        self.token = self.get_access_token()

    def get_access_token(self):
        """접근 토큰 발급"""
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "secretkey": self.secret_key
        }
        res = requests.post(url, headers=headers, data=json.dumps(body))
        return res.json().get('access_token')

    def buy_market_order(self, symbol, amount_usd):
        """미국 주식 시장가 매수 (금액 단위 주문)"""
        url = f"{self.base_url}/uapi/google-nasdaq/v1/trading/order-down" # 모의투자용 주문 API
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appkey": self.app_key,
            "secretkey": self.secret_key,
            "tr_id": "VTTT1002U", # 모의투자 나스닥 매수 ID
            "custtype": "P"
        }
        # 실제 한투 API 명세에 따른 상세 Body 구성 필요
        # 1주 단위가 아닌 소수점/금액 주문 시 별도 tr_id 사용
        print(f"🚀 [매수 실행] {symbol}을(를) ${amount_usd} 만큼 매수 시도합니다.")
