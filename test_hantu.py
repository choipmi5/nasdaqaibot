import requests
import json
import os

# 1. 환경 변수 로드 (Secrets에 등록한 이름과 동일해야 함)
APP_KEY = os.environ.get('HANTU_APP_KEY')
SECRET_KEY = os.environ.get('HANTU_SECRET_KEY')
ACC_NO = os.environ.get('HANTU_ACCOUNT_NO') # 계좌번호 앞 8자리

# 모의투자용 주소 (실전은 도메인이 다름)
BASE_URL = "https://openapivts.koreainvestment.com:29443"

def get_hantu_token():
    print("--- [1] 토큰 발급 테스트 시작 ---")
    url = f"{BASE_URL}/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": SECRET_KEY
    }
    res = requests.post(url, data=json.dumps(body))
    if res.status_code == 200:
        token = res.json().get('access_token')
        print(f"✅ 토큰 발급 성공! (Token: {token[:10]}...)")
        return token
    else:
        print(f"❌ 토큰 발급 실패: {res.text}")
        return None

def check_balance(token):
    print("\n--- [2] 계좌 잔고 조회 테스트 시작 ---")
    # 해외주식(미국) 모의투자 잔고 조회 URL
    url = f"{BASE_URL}/uapi/google-nasdaq/v1/trading/inquire-psbl-order"
    
    # 헤더 설정 (한투 API 필수 규격)
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "secretkey": SECRET_KEY,
        "tr_id": "VTTT1007U", # 모의투자 미국 매수 가능 조회 ID
        "custtype": "P"
    }
    
    # 파라미터 (계좌번호 등)
    params = {
        "CANO": ACC_NO,
        "ACNT_PRDT_CD": "01", # 보통 01
        "WCRC_FRCR_DVSN_CD": "02", # 외화 기준
        "ITEM_CD": "TQQQ", # 테스트용 종목
        "ORD_UNPR": "0",
        "ORD_DVSN": "00"
    }

    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        data = res.json()
        print(f"✅ 잔고 조회 성공!")
        print(f"💰 주문 가능 외화: ${data.get('output', {}).get('frcr_ord_psbl_amt1', '0')}")
    else:
        print(f"❌ 잔고 조회 실패: {res.text}")

if __name__ == "__main__":
    if not APP_KEY or not SECRET_KEY:
        print("⚠️ 에러: API Key 설정이 안 되어 있습니다. GitHub Secrets를 확인하세요.")
    else:
        token = get_hantu_token()
        if token:
            check_balance(token)
