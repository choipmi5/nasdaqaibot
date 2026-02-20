import os
import yfinance as yf
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai

# [1. 환경 설정]
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

# [2. 섹터 및 종목 리스트]
SECTORS = {
    "SEMICON": ["NVDA", "AMD", "AVGO", "TSM", "MU", "ARM", "LRCX", "AMAT", "SOXL"],
    "BIGTECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "QQQ"],
    "AI/SW/FIN": ["PLTR", "SNOW", "ADBE", "ORCL", "CRM", "PANW", "COIN", "MSTR"]
}

STOCKS = list(set(["QQQ", "TQQQ", "SQQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "SOXL", "SOXS", "AVGO", "NFLX", "TSM", "ADBE", "COST", "PEP", "AZN", "LIN", "CSCO", "TMUS", "INTC", "TXN", "QCOM", "AMAT", "ADP", "ISRG", "SBUX", "MDLZ", "GILD", "INTU", "VRTX", "AMGN", "REGN", "PYPL", "FISV", "BKNG", "CSX", "MU", "PANW", "SNPS", "CDNS", "ORLY", "MNST", "MAR", "KDP", "CHTR", "KLAC", "AEP", "LRCX", "ADSK", "DXCM", "MELI", "IDXX", "PAYX", "CTAS", "LULU", "MCHP", "MRVL", "CPRT", "ODFL", "TEAM", "ALGN", "WDAY", "FAST", "PCAR", "ROST", "DLTR", "EBAY", "SIRI", "ZM", "JD", "LCID", "DDOG", "RIVN", "ENPH", "CEG", "ZS", "ABNB", "PDD", "OKTA", "CONL", "NVDL", "TSLL", "SOXX", "SCHD", "JEPI", "VOO", "IVV", "VTI", "UPRO", "TMF", "ARM", "PLTR", "SNOW", "U", "COIN", "MSTR"]))

# [3. 핵심 유틸리티 함수]
def flatten_df(df):
    """멀티인덱스 칼럼을 단일 칼럼으로 변환"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # MFI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_f = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_f = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_f / neg_f)))
    
    # MACD
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # 볼린저 밴드
    df['MA20'] = df['Close'].rolling(20).mean()
    df['STD'] = df['Close'].rolling(20).std()
    df['BB_Low'] = df['MA20'] - (df['STD'] * 2)
    return df

def get_market_status():
    """VIX 및 QQQ 변동성 체크 (에러 방지 로직 강화)"""
    try:
        # VIX 지수 가져오기
        vix_data = flatten_df(yf.download("^VIX", period="1d", progress=False))
        vix = float(vix_data['Close'].iloc[-1])
        
        # QQQ 지수 수익률 계산
        qqq_data = flatten_df(yf.download("QQQ", period="2d", progress=False))
        if len(qqq_data) >= 2:
            change = float(((qqq_data['Close'].iloc[-1] / qqq_data['Close'].iloc[-2]) - 1) * 100)
        else:
            change = 0.0
        return vix, change
    except Exception as e:
        print(f"Market Status Error: {e}")
        return 20.0, 0.0

def get_external_data(s, t_obj, curr_p):
    data = {"sentiment": "중립", "earnings": "안정", "target": 0.0, "upside": 0.0, "score": 0}
    try:
        # 1. AI 뉴스 분석
        news = t_obj.news[:3]
        if news and GEMINI_API_KEY:
            titles = [n['title'] for n in news]
            prompt = f"Stock {s}: {titles}. Respond Positive/Negative/Neutral only."
            res = model.generate_content(prompt).text.strip().capitalize()
            data["sentiment"] = "호재" if "Positive" in res else "악재" if "Negative" in res else "중립"
            if data["sentiment"] == "호재": data["score"] += 20
        
        # 2. 애널리스트 목표가
        info = t_obj.info
        target = info.get('targetMeanPrice', 0.0)
        if target and target > 0:
            data["target"] = float(target)
            data["upside"] = float(((target / curr_p) - 1) * 100)
            if data["upside"] > 15: data["score"] += 15

        # 3. 실적 발표일
        cal = t_obj.calendar
        # calendar 구조 변경 대응
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            e_date = cal.iloc[0, 0]
        elif isinstance(cal, dict) and 'Earnings Date' in cal:
            e_date = cal['Earnings Date'][0]
        else:
            e_date = None

        if e_date:
            days = (pd.to_datetime(e_date).replace(tzinfo=None) - datetime.now().replace(tzinfo=None)).days
            if 0 <= days <= 7: 
                data["earnings"] = f"⚠️D-{days}"
                data["score"] -= 40
    except: pass
    return data

# [4. 메인 분석 엔진]
def run_full_scan():
    print("🚀 NASDAQ Master-Quant System Starting...")
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul'); now = datetime.now(kst)
    
    # 지표 값들을 확실히 float로 받음 (에러 발생 지점 수정)
    vix, m_perf = get_market_status()
    
    # 명시적 float 비교로 ValueError 방지
    is_risky = float(vix) > 24.0 or float(m_perf) < -1.5
    risk_mode = "⚠️방어운전" if is_risky else "✅안정적"
    score_min = 45 if risk_mode == "⚠️방어운전" else 30
    
    review_list, super_buys, strong_buys, normal_buys = [], [], [], []
    sector_momentum = {k: 0 for k in SECTORS.keys()}
    results = []

    for idx, s in enumerate(STOCKS):
        print(f"[{idx+1}/{len(STOCKS)}] {s}", end='\r')
        try:
            t_obj = yf.Ticker(s)
            df = flatten_df(t_obj.history(period="100d"))
            if len(df) < 30: continue
            
            df = calculate_indicators(df)
            curr_p = float(df['Close'].iloc[-1])
            
            # 복기 로직
            if df['RSI'].iloc[-2] < 35:
                hit = float(df['High'].iloc[-1]) >= float(df['Close'].iloc[-2]) * 1.025
                review_list.append(f"{s}:{'🎯' if hit else '⏳'}")

            # 기술적 판단
            high_52 = float(df['High'].max())
            drop_rate = float((1 - (curr_p / high_52)) * 100)
            is_turning = bool(df['MACD'].iloc[-1] > df['Signal'].iloc[-1])
            is_vol = bool(df['Volume'].iloc[-1] > df['Volume'].rolling(5).mean().iloc[-1] * 1.5)
            is_bb_support = bool(curr_p <= df['BB_Low'].iloc[-1] * 1.02)
            
            external = get_external_data(s, t_obj, curr_p)
            
            # 주도 섹터 수급 체크
            if is_vol and curr_p > float(df['Close'].iloc[-2]):
                for s_name, stocks in SECTORS.items():
                    if s in stocks: sector_momentum[s_name] += 1

            results.append({
                "symbol": s, "price": curr_p, "rsi": float(df['RSI'].iloc[-1]), "mfi": float(df['MFI'].iloc[-1]),
                "drop": drop_rate, "is_turning": is_turning, "is_vol": is_vol, "is_bb": is_bb_support,
                "external": external, "df": df
            })
            time.sleep(0.1)
        except Exception as e:
            print(f"\nError analyzing {s}: {e}")
            continue

    # 가점 및 최종 분류
    hot_sectors = [k for k, v in sector_momentum.items() if v >= 2]
    
    for item in results:
        s = item['symbol']
        theme_bonus = 15 if any(s in SECTORS[hs] for hs in hot_sectors) else 0
        
        total_score = item['external']['score'] + theme_bonus + \
                      (25 if item['rsi'] < 35 else 0) + \
                      (10 if item['is_turning'] else 0) + \
                      (15 if item['is_vol'] else 0) + \
                      (10 if item['drop'] > 30 else 0) + \
                      (10 if item['is_bb'] else 0)

        atr = (item['df']['High'] - item['df']['Low']).rolling(14).mean().iloc[-1]
        t1, t2, stop = item['price'] + (atr * 2), item['price'] + (atr * 4), item['price'] - (atr * 1.5)
        
        t_link = f"https://tossinvest.com/stocks/{s}"
        msg = (f"🔥 **`{s}`** (점수:{total_score})\n"
               f"📍 Buy: ${item['price']:.2f} (RSI:{item['rsi']:.1f})\n"
               f"🎯 Target: ${t1:.2f} / ${t2:.2f} | 🛑 Stop: ${stop:.2f}\n"
               f"📊 뉴스:{item['external']['sentiment']} | 낙폭:{item['drop']:.1f}% | 업사이드:{item['external']['upside']:.1f}%\n"
               f"🏛 실적:{item['external']['earnings']} | [주문하기]({t_link})")

        if "⚠️" in item['external']['earnings']: continue
        
        if total_score >= 70 and item['is_vol'] and risk_mode == "✅안정적":
            super_buys.append(msg)
        elif total_score >= 55:
            strong_buys.append(msg)
        elif total_score >= score_min:
            normal_buys.append(msg)

    # 전송 레이아웃
    header = [
        f"🇺🇸 *NASDAQ PRO MASTER REPORT*",
        f"📅 {now.strftime('%m-%d %H:%M')} | {risk_mode}",
        f"📉 VIX: {vix:.2f} | Mkt: {m_perf:+.2f}%",
        f"🚩 Hot Sectors: {', '.join(hot_sectors) if hot_sectors else '중립'}",
        "━━━━━━━━━━━━━━",
        f"📊 **[전일 복기]**\n" + (", ".join(review_list[:12]) if review_list else "-"),
        "━━━━━━━━━━━━━━"
    ]
    
    full_text = "\n".join(header + 
                ([f"🎯 **[SUPER BUY]**\n" + "\n\n".join(super_buys[:3])] if super_buys else []) +
                ([f"\n💎 **[STRONG BUY]**\n" + "\n\n".join(strong_buys[:5])] if strong_buys else []) +
                ([f"\n🔍 **[NORMAL BUY]**\n" + "\n\n".join(normal_buys[:8])] if normal_buys else []) +
                ["━━━━━━━━━━━━━━", f"✅ {len(results)}개 전수조사 완료"])

    for part in [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      json={"chat_id": CHAT_ID, "text": part, "parse_mode": "Markdown", "disable_web_page_preview": True})

if __name__ == "__main__":
    run_full_scan()





