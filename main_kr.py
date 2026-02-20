import os
import yfinance as yf
import pandas as pd
import requests
import time
import json
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai

# ==========================================
# 1. 환경 설정 및 종목 리스트 (100개 유지)
# ==========================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

# (SECTORS 및 KR_STOCKS 리스트는 기존과 동일하게 유지됩니다)
SECTORS = {
    "반도체": ["005930.KS", "000660.KS", "058470.KQ", "403870.KQ", "399720.KQ", "394280.KQ", "080220.KQ"],
    "바이오": ["207940.KS", "068270.KS", "191150.KQ", "028300.KQ", "068760.KQ", "145020.KQ", "000100.KS"],
    "2차전지": ["373220.KS", "051910.KS", "006400.KS", "247540.KQ", "086520.KQ", "348370.KQ", "003670.KS"],
    "자동차": ["005380.KS", "000270.KS", "012330.KS", "086280.KS", "018880.KS"],
    "금융/지주": ["105560.KS", "055550.KS", "086790.KS", "138040.KS", "000810.KS", "032830.KS", "003550.KS", "034730.KS"],
    "엔터/게임": ["352820.KS", "259960.KS", "041510.KQ", "035900.KQ", "251270.KS", "036570.KS", "112040.KQ", "078340.KQ"]
}

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

# --- 기술 분석 및 리포트 유틸리티 ---
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

def get_analyst_consensus(t_obj):
    """증권사 리포트 연동: 목표가 및 투자의견 추출"""
    try:
        info = t_obj.info
        target_p = info.get('targetMeanPrice', 0)
        recommend = info.get('recommendationKey', 'none').replace('_', ' ').capitalize()
        return target_p, recommend
    except:
        return 0, "N/A"

def get_ai_analysis(s_name, t_obj):
    if not GEMINI_API_KEY: return "중립", 0
    try:
        news_list = t_obj.news
        if not news_list: return "정보부족", 0
        titles = [n['title'] for n in news_list[:5]]
        prompt = f"Stock: {s_name}. News: {titles}. Positive, Negative, or Neutral? Reply with ONE word."
        response = model.generate_content(prompt)
        res = response.text.strip().capitalize()
        if "Positive" in res: return "호재", 20
        if "Negative" in res: return "악재", -20
        return "중립", 0
    except: return "중립", 0

def get_yesterday_backtest():
    try:
        m_df = flatten_df(yf.download("^KS11", period="5d", progress=False))
        if len(m_df) < 2: return 0.0
        change = ((m_df['Close'].iloc[-1] / m_df['Close'].iloc[-2]) - 1) * 100
        return change
    except: return 0.0

# --- 메인 실행 엔진 ---
def run_full_pro_system():
    print("🚀 국장 PRO 퀀트 시스템(리포트 연동형) 가동 중...")
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul'); now = datetime.now(kst)
    
    y_perf = get_yesterday_backtest()
    risk_mode = "⚠️방어운전" if y_perf < -1.0 else "✅안정적"
    score_threshold = 45 if y_perf < -0.5 else 30
    
    analysis_results = []
    sector_momentum = {name: 0 for name in SECTORS.keys()}

    for s_name, s_code in KR_STOCKS:
        try:
            t_obj = yf.Ticker(s_code)
            df = flatten_df(t_obj.history(period="100d"))
            if len(df) < 20: continue
            
            curr_p = float(df['Close'].iloc[-1])
            rsi = calculate_rsi(df['Close']).iloc[-1]
            mfi = calculate_mfi(df).iloc[-1]
            high_52 = df['High'].max()
            drop_rate = (1 - (curr_p / high_52)) * 100
            
            # 수급 엔진
            vol_spike = df['Volume'].iloc[-1] > df['Volume'].rolling(10).mean().iloc[-1] * 1.8
            price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]
            
            supply_tag = "보통"; s_score = 0
            if vol_spike and price_up and mfi < 50:
                supply_tag = "💎양매수포착"; s_score = 35
            elif mfi < 30:
                supply_tag = "🔥저점매집"; s_score = 25
            
            if s_score > 0:
                for s_tile, codes in SECTORS.items():
                    if s_code in codes: sector_momentum[s_tile] += 1

            # [증권사 리포트 연동 추가]
            broker_target, broker_opinion = get_analyst_consensus(t_obj)
            broker_upside = ((broker_target / curr_p) - 1) * 100 if broker_target > 0 else 0
            
            # 리포트 가점: 목표가가 현재가보다 20% 이상 높고 투자의견이 좋을 때
            broker_bonus = 15 if broker_upside > 20 and "Buy" in broker_opinion else 0

            # 실적 정보
            e_status = "안정"
            try:
                cal = t_obj.calendar
                e_date = cal['Earnings Date'][0] if isinstance(cal, dict) else cal.iloc[0][0]
                days = (pd.to_datetime(e_date).replace(tzinfo=None) - datetime.now().replace(tzinfo=None)).days
                if 0 <= days <= 7: e_status = f"⚠️D-{days}"
            except: pass

            analysis_results.append({
                "name": s_name, "code": s_code, "price": curr_p, "rsi": rsi, "mfi": mfi,
                "supply": supply_tag, "s_score": s_score, "e_status": e_status, 
                "drop": drop_rate, "broker_target": broker_target, "broker_opinion": broker_opinion,
                "broker_upside": broker_upside, "broker_bonus": broker_bonus,
                "df": df, "t_obj": t_obj
            })
            time.sleep(0.01)
        except: continue

    hot_sectors = [k for k, v in sector_momentum.items() if v >= 2]
    final_cards = []

    for item in analysis_results:
        theme_bonus = 15 if any(item['code'] in SECTORS[hs] for hs in hot_sectors) else 0
        
        sentiment, ai_score = "중립", 0
        if item['rsi'] < 42 or item['s_score'] > 0 or theme_bonus > 0:
            sentiment, ai_score = get_ai_analysis(item['name'], item['t_obj'])
            time.sleep(0.4)

        # 최종 점수 합산 (리포트 가점 포함)
        total_score = item['s_score'] + ai_score + theme_bonus + item['broker_bonus'] + \
                      (20 if item['rsi'] < 33 else 0) + (10 if item['drop'] > 35 else 0)
        
        atr = (item['df']['High'] - item['df']['Low']).rolling(14).mean().iloc[-1]
        t1, t2, stop = item['price'] + (atr * 1.5), item['price'] + (atr * 3.0), item['price'] - (atr * 1.2)
        
        if total_score >= score_threshold or item['rsi'] < 30:
            t_link = f"https://tossinvest.com/stocks/{item['code'].split('.')[0]}"
            hot_tag = " [Hot테마]" if theme_bonus > 0 else ""
            
            # 리포트 요약 텍스트
            broker_info = f"{int(item['broker_target']):,}원({item['broker_upside']:.1f}%)" if item['broker_target'] > 0 else "정보없음"
            
            card = (f"🔥 **{item['name']}**{hot_tag} (점수:{total_score})\n"
                    f"📍 Buy: {int(item['price']):,}원 (RSI:{item['rsi']:.1f})\n"
                    f"🎯 Target: {int(t1):,} / {int(t2):,}원\n"
                    f"🛑 Stop: {int(stop):,}원\n"
                    f"📊 뉴스:{sentiment} | 수급:{item['supply']}\n"
                    f"🏛 리포트:{item['broker_opinion']} | 목표:{broker_info}\n"
                    f"🔗 [주문하기]({t_link})")
            final_cards.append((total_score, card))

    final_cards.sort(key=lambda x: x[0], reverse=True)
    
    header = f"🇰🇷 *KOREA STOCK QUANT PRO*\n📅 {now.strftime('%m-%d %H:%M')} | {risk_mode}\n"
    if hot_sectors: header += f"🚩 주도섹터: {', '.join(hot_sectors)}\n"
    header += f"📈 어제 시장변동: {y_perf:+.2f}%\n━━━━━━━━━━━━━━\n\n"
    
    body = "\n\n".join([c[1] for c in final_cards[:15]])
    full_message = header + body

    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": full_message, "parse_mode": "Markdown", "disable_web_page_preview": True})

if __name__ == "__main__":
    run_full_pro_system()


