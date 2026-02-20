import os
import yfinance as yf
import pandas as pd
import requests
import json
import time
from datetime import datetime
import pytz
import google.generativeai as genai

# ==========================================
# 1. 환경 설정 및 종목 리스트 (100개 튜플)
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

# --- 헬퍼 함수 ---
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + (gain / loss)))

def calculate_mfi(df, period=14):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_f = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_f = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    return 100 - (100 / (1 + (pos_f / neg_f)))

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    return macd, macd.ewm(span=9, adjust=False).mean()

def get_ai_sentiment(s_name, t_obj):
    """지표가 좋은 종목에 대해서만 선별적으로 AI 뉴스 분석 실행"""
    if not GEMINI_API_KEY: return "중립", 0
    try:
        news = t_obj.news[:3]
        if not news: return "데이터없음", 0
        titles = [n['title'] for n in news]
        prompt = f"Analyze Korean stock '{s_name}' based on news titles: {titles}. Respond ONLY with: Positive, Negative, or Neutral."
        res = model.generate_content(prompt).text.strip().capitalize()
        if "Positive" in res: return "호재", 20
        if "Negative" in res: return "악재", -20
        return "중립", 0
    except: return "에러", 0

# --- 실행 함수 ---
def run_analysis_kr():
    print(f"🚀 KOREA Stock Analysis (100개) 시작...")
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul'); now = datetime.now(kst)
    
    # 지수 확인 (코스피)
    market_df = flatten_df(yf.download("^KS11", period="5d", progress=False))
    market_recovery = (market_df['Close'].iloc[-1] > market_df['Close'].iloc[-2]) if not market_df.empty else False
    
    review_reports, super_buys, strong_buys, normal_buys = [], [], [], []
    total_analyzed, down_count = 0, 0

    for idx, (s_name, s_code) in enumerate(KR_STOCKS):
        print(f"[{idx+1}/{len(KR_STOCKS)}] {s_name} 분석 중...", end='\r')
        try:
            t_obj = yf.Ticker(s_code)
            df = flatten_df(t_obj.history(period="60d"))
            if len(df) < 30: continue
            
            # 실시간 가격 (1분봉 시도, 안되면 일봉)
            recent = t_obj.history(period="1d", interval="1m")
            curr_p = float(recent['Close'].iloc[-1]) if not recent.empty else float(df['Close'].iloc[-1])
            total_analyzed += 1
            
            # 1. 전일 복기 (RSI 36 미만 종목이 오늘 2% 올랐는가)
            rsi_series = calculate_rsi(df['Close'])
            if len(rsi_series) > 1 and rsi_series.iloc[-2] < 36:
                hit = float(df['High'].iloc[-1]) >= float(df['Close'].iloc[-2]) * 1.02
                review_reports.append(f"{s_name}:{'🎯' if hit else '⏳'}")

            # 2. 기술적 지표 계산
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            if curr_p < ma20: down_count += 1
            mfi = float(calculate_mfi(df).iloc[-1])
            macd, signal = calculate_macd(df['Close'])
            
            is_oversold = rsi_series.iloc[-1] < 36 or curr_p <= (ma20 - (df['Close'].rolling(20).std().iloc[-1] * 2))
            is_turning = float(macd.iloc[-1]) > float(signal.iloc[-1])
            is_vol_spike = float(df['Volume'].iloc[-1]) > float(df['Volume'].rolling(5).mean().iloc[-1]) * 1.2
            
            # 3. 선별적 AI 분석 (지표가 좋을 때만 실행하여 속도와 API 보호)
            sentiment, ai_score = "중립", 0
            if is_oversold or is_turning or is_vol_spike:
                sentiment, ai_score = get_ai_sentiment(s_name, t_obj)
                time.sleep(0.5) # Gemini 분당 제한 방지
            
            total_score = ai_score + (20 if is_oversold else 0) + (10 if is_turning else 0) + (10 if is_vol_spike else 0)
            
            # ATR 목표가 (국장 맞춤형 1.2/2.5배)
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            t1_p, t2_p = curr_p + (atr * 1.2), curr_p + (atr * 2.5)
            
            toss_link = f"https://tossinvest.com/stocks/{s_code.split('.')[0]}"
            t_info = (f"🇰🇷 **`{s_name}`** ({total_score}점)\n📍 {int(curr_p):,}원 (RSI:{rsi_series.iloc[-1]:.1f})\n"
                      f"🎯 목표: {int(t1_p):,} / {int(t2_p):,}원\n📊 뉴스:{sentiment}\n🔗 [토스주문]({toss_link})")

            # 4. 등급 판정
            if is_oversold and mfi < 40 and is_turning and is_vol_spike and market_recovery:
                super_buys.append(t_info)
            elif is_oversold and (mfi < 40 or is_turning) and (is_vol_spike or total_score >= 20):
                strong_buys.append(t_info)
            elif is_oversold or total_score >= 40:
                normal_buys.append(t_info)
            
            time.sleep(0.1) 
        except Exception as e:
            continue

    # 5. 결과 리포트 전송
    ratio = down_count / total_analyzed if total_analyzed > 0 else 0.5
    mode_str = "🚀 강세" if ratio < 0.3 else "📈 보통" if ratio < 0.6 else "⚠️ 약세"
    
    header = f"🇰🇷 *KOREA STOCK PRO AI*\n📅 {now.strftime('%m-%d %H:%M')} | {mode_str}\n━━━━━━━━━━━━━━\n📊 **[전일 복기]**\n" + (", ".join(review_reports[:12]) if review_reports else "-") + "\n━━━━━━━━━━━━━━"
    
    # 메시지 전송 로직 (슈퍼/스트롱/노멀 순서로 분할 전송)
    def send_msg(text):
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True})

    send_msg(header)
    
    if super_buys: send_msg("🎯 **[SUPER BUY]**\n\n" + "\n\n".join(super_buys))
    if strong_buys: 
        for i in range(0, len(strong_buys), 5):
            send_msg("💎 **[STRONG BUY]**\n\n" + "\n\n".join(strong_buys[i:i+5]))
    if normal_buys:
        for i in range(0, len(normal_buys), 5):
            send_msg("🔍 **[NORMAL BUY]**\n\n" + "\n\n".join(normal_buys[i:i+5]))

    print("\n✅ 분석 완료 및 전송 성공.")

if __name__ == "__main__":
    run_analysis_kr()



