import os
import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime
import pytz
import google.generativeai as genai

# ==========================================
# 1. 환경 설정 및 종목 리스트 (100개 전수 포함)
# ==========================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

KR_STOCKS = [
    ("삼성전자", "005930.KS"), ("SK하이닉스", "000660.KS"), ("LG엔솔", "373220.KS"), ("삼성바이오", "207940.KS"), ("현대차", "005380.KS"),
    ("기아", "000270.KS"), ("셀트리온", "068270.KS"), ("KB금융", "105560.KS"), ("POSCO홀딩스", "005490.KS"), ("NAVER", "035420.KS"),
    ("신한지주", "055550.KS"), ("삼성물산", "028260.KS"), ("현대모비스", "012330.KS"), ("LG화학", "051910.KS"), ("하나금융지주", "086790.KS"),
    ("삼성생명", "032830.KS"), ("카카오", "035720.KS"), ("메리츠금융", "138040.KS"), ("삼성SDI", "006400.KS"), ("LG전자", "066570.KS"),
    ("카카오뱅크", "323410.KS"), ("삼성화재", "000810.KS"), ("KT&G", "033780.KS"), ("한국전력", "015760.KS"), ("HMM", "011200.KS"),
    ("SK이노베이션", "096770.KS"), ("삼성전기", "009150.KS"), ("크래프톤", "259960.KS"), ("두산에너빌리티", "034020.KS"), ("HD현대중공업", "329180.KS"),
    ("에코프로비엠", "247540.KQ"), ("에코프로", "086520.KQ"), ("HLB", "028300.KQ"), ("알테오젠", "191150.KQ"), ("엔켐", "348370.KQ"),
    ("리노공업", "058470.KQ"), ("레인보우로보틱스", "272410.KQ"), ("HPSP", "403870.KQ"), ("신성델타테크", "065350.KQ"), ("제주반도체", "080220.KQ"),
    ("포스코퓨처엠", "003670.KS"), ("SK", "034730.KS"), ("S-Oil", "010950.KS"), ("고려아연", "010130.KS"), ("삼성에스디에스", "018260.KS"),
    ("한화에어로스페이스", "012450.KS"), ("대한항공", "003490.KS"), ("KT", "030200.KS"), ("기업은행", "024110.KS"), ("HD현대", "267250.KS"),
    ("LG", "003550.KS"), ("한국금융지주", "071050.KS"), ("아모레퍼시픽", "090430.KS"), ("코웨이", "021240.KS"), ("금양", "001570.KS"),
    ("한온시스템", "018880.KS"), ("현대글로비스", "086280.KS"), ("삼성중공업", "010140.KS"), ("넷마블", "251270.KS"), ("카카오페이", "377300.KS"),
    ("엔씨소프트", "036570.KS"), ("유한양행", "000100.KS"), ("한미사이언스", "008930.KS"), ("한미약품", "128940.KS"), ("오리온", "271560.KS"),
    ("미래에셋증권", "006800.KS"), ("하이브", "352820.KS"), ("팬오션", "028670.KS"), ("두산밥캣", "241560.KS"), ("롯데케미칼", "011170.KS"),
    ("현대건설", "000720.KS"), ("LG생활건강", "051900.KS"), ("SK바이오사이언스", "302440.KS"), ("호텔신라", "008770.KS"), ("GS", "078930.KS"),
    ("포스코인터내셔널", "047050.KS"), ("에스디바이오센서", "137310.KS"), ("씨젠", "096530.KQ"), ("펄어비스", "263750.KQ"), ("셀트리온제약", "068760.KQ"),
    ("휴젤", "145020.KQ"), ("클래시스", "214150.KQ"), ("에스엠", "041510.KQ"), ("JYP Ent.", "035900.KQ"), ("루닛", "328130.KQ"),
    ("가온칩스", "399720.KQ"), ("오픈엣지테크놀로지", "394280.KQ"), ("소울브레인", "357780.KQ"), ("동진쎄미켐", "005290.KQ"), ("원익IPS", "030530.KQ"),
    ("이오테크닉스", "039030.KQ"), ("솔브레인홀딩스", "036830.KQ"), ("파두", "440110.KQ"), ("위메이드", "112040.KQ"), ("컴투스", "078340.KQ"),
    ("바이오니아", "064550.KQ"), ("STX", "011810.KS"), ("한화오션", "042660.KS"), ("LS", "006260.KS"), ("LS ELECTRIC", "010120.KS")
]

# --- 기술 분석 헬퍼 함수 ---
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + (gain / loss)))

def calculate_mfi(df, period=14):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_f = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_f = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    return 100 - (100 / (1 + (pos_f / neg_f)))

def get_ai_analysis(s_name, t_obj):
    if not GEMINI_API_KEY: return "중립", 0
    try:
        news_list = t_obj.news
        if not news_list: return "정보부족", 0
        titles = [n['title'] for n in news_list[:5]]
        prompt = f"Stock: {s_name}. News: {titles}. 긍정이면 Positive, 부정이면 Negative, 판단불가면 Neutral. 한 단어만 답해."
        response = model.generate_content(prompt)
        res = response.text.strip().capitalize()
        if "Positive" in res: return "호재", 20
        if "Negative" in res: return "악재", -20
        return "중립", 0
    except: return "중립", 0

# --- 메인 로직 ---
def run_analysis_kr():
    print(f"🚀 국장 예민한 수급 엔진 가동 (100개 종목)...")
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul'); now = datetime.now(kst)
    
    # 시장 지수 확인 (반등장인지 확인)
    market_df = flatten_df(yf.download("^KS11", period="5d", progress=False))
    market_recovery = (market_df['Close'].iloc[-1] > market_df['Close'].iloc[-2]) if not market_df.empty else False
    
    super_buys, strong_buys, normal_buys = [], [], []
    total_analyzed = 0

    for s_name, s_code in KR_STOCKS:
        try:
            t_obj = yf.Ticker(s_code)
            df = flatten_df(t_obj.history(period="60d"))
            if len(df) < 20: continue
            
            recent = t_obj.history(period="1d", interval="1m")
            curr_p = float(recent['Close'].iloc[-1]) if not recent.empty else float(df['Close'].iloc[-1])
            total_analyzed += 1
            
            # 1. 예민한 수급 및 기술 지표
            rsi = calculate_rsi(df['Close']).iloc[-1]
            mfi = calculate_mfi(df).iloc[-1]
            vol_ma = df['Volume'].rolling(5).mean().iloc[-1]
            is_high_volume = df['Volume'].iloc[-1] > vol_ma * 1.3 # 거래량 30% 급증 여부
            
            # 수급 상태 세분화 (MFI 기준 상향으로 더 예민하게 포착)
            supply_status = "보통"; supply_score = 0
            if mfi < 35: # 자금 유입 초기 신호 (매우 예민)
                supply_status = "🔥강력매수"; supply_score = 25
            elif mfi < 50 and is_high_volume:
                supply_status = "수급개선"; supply_score = 15
            elif mfi > 70:
                supply_status = "차익경계"; supply_score = -10
            
            # 2. 선별적 AI 분석 (저점이거나 수급이 들어올 때만)
            sentiment, ai_score = "중립", 0
            if rsi < 45 or supply_score > 0:
                sentiment, ai_score = get_ai_analysis(s_name, t_obj)
                time.sleep(0.4) # API 속도 조절

            # 3. 실적 리스크
            earnings_status = "안정"
            try:
                cal = t_obj.calendar
                e_date = cal['Earnings Date'][0] if isinstance(cal, dict) else cal.iloc[0][0]
                days = (pd.to_datetime(e_date).replace(tzinfo=None) - datetime.now().replace(tzinfo=None)).days
                if 0 <= days <= 7: earnings_status = f"⚠️D-{days}"
            except: pass

            # 4. 최종 점수 합산 알고리즘
            total_score = ai_score + supply_score + (20 if rsi < 35 else 0)
            
            # 5. ATR 기반 목표가/손절가
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            t1_p, t2_p, stop_p = curr_p + (atr * 1.5), curr_p + (atr * 3.0), curr_p - (atr * 1.2)
            
            toss_link = f"https://tossinvest.com/stocks/{s_code.split('.')[0]}"
            
            # [출력 포맷 통일]
            t_info = (f"🔥 **{s_name}** (점수:{total_score})\n"
                      f"📍 Buy: {int(curr_p):,}원 (RSI:{rsi:.1f})\n"
                      f"🎯 Target: {int(t1_p):,} / {int(t2_p):,}원\n"
                      f"🛑 Stop: {int(stop_p):,}원\n"
                      f"📊 뉴스:{sentiment} | 실적:{earnings_status} | 수급:{supply_status}\n"
                      f"🔗 [주문하기]({toss_link})")

            # 등급 판정
            if total_score >= 45 and market_recovery: super_buys.append(t_info)
            elif total_score >= 25: strong_buys.append(t_info)
            elif rsi < 33: normal_buys.append(t_info)
            
            time.sleep(0.05)
        except Exception as e:
            continue

    # 6. 분할 전송 리포트
    header = f"🇰🇷 *KOREA STOCK QUANT AI*\n📅 {now.strftime('%m-%d %H:%M')}\n━━━━━━━━━━━━━━"
    def send(msg): requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True})

    send(header)
    
    if super_buys: 
        send("🎯 **[SUPER BUY]**\n\n" + "\n\n".join(super_buys[:5]))
    
    if strong_buys:
        for i in range(0, len(strong_buys), 5):
            send("💎 **[STRONG BUY]**\n\n" + "\n\n".join(strong_buys[i:i+5]))
            time.sleep(1) # 전송 안정성
            
    if normal_buys:
        for i in range(0, len(normal_buys), 5):
            send("🔍 **[NORMAL BUY]**\n\n" + "\n\n".join(normal_buys[i:i+5]))
            time.sleep(1)

if __name__ == "__main__":
    run_analysis_kr()

