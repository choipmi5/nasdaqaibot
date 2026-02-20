import os
import yfinance as yf
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai
import re

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
    df['BB_High'] = df['MA20'] + (df['STD'] * 2) # 상단 밴드 추가
    return df

def get_market_status():
    """VIX 및 QQQ 변동성 체크"""
    try:
        vix_data = flatten_df(yf.download("^VIX", period="1d", progress=False))
        vix = float(vix_data['Close'].iloc[-1])
        
        qqq_data = flatten_df(yf.download("QQQ", period="2d", progress=False))
        if len(qqq_data) >= 2:
            change = float(((qqq_data['Close'].iloc[-1] / qqq_data['Close'].iloc[-2]) - 1) * 100)
        else:
            change = 0.0
        return vix, change
    except Exception as e:
        print(f"Market Status Error: {e}")
        return 20.0, 0.0

def get_target_price_fallback(ticker, current_price, history_df):
    """
    Upside N/A 방지를 위한 Fallback 로직
    1. Gemini에게 물어보기
    2. 실패시 기술적 저항선(BB상단 or 52주 고가) 사용
    """
    target = None
    source = "N/A"
    
    # [시도 1] Gemini에게 최신 컨센서스 문의
    try:
        if GEMINI_API_KEY:
            prompt = f"What is the average analyst target price for stock {ticker} in numbers only? If unsure, return 0."
            response = model.generate_content(prompt).text
            # 숫자만 추출
            numbers = re.findall(r"\d+\.\d+|\d+", response)
            if numbers:
                ai_target = float(numbers[0])
                if ai_target > current_price: # 현재가보다 높을 때만 유효하다고 판단
                    return ai_target, "🤖AI"
    except:
        pass

    # [시도 2] 기술적 목표가 (최근 120일 고가와 볼린저밴드 상단 중 큰 값)
    try:
        recent_high = history_df['High'].iloc[-120:].max()
        bb_high = history_df['BB_High'].iloc[-1]
        tech_target = max(recent_high, bb_high * 1.05)
        return tech_target, "📈Tech"
    except:
        return current_price * 1.1, "Est."

def get_external_data(s, t_obj, curr_p, df_hist):
    data = {"sentiment": "중립", "earnings": "안정", "target": None, "upside": "N/A", "upside_tag": "", "score": 0}
    try:
        # 1. AI 뉴스 분석
        try:
            news = t_obj.news[:3]
            if news and GEMINI_API_KEY:
                titles = [n['title'] for n in news]
                prompt = f"Stock {s}: {titles}. Respond exactly one word: Positive, Negative, or Neutral."
                res = model.generate_content(prompt).text.strip()
                if "Positive" in res:
                    data["sentiment"] = "호재"
                    data["score"] += 20
                elif "Negative" in res:
                    data["sentiment"] = "악재"
                else:
                    data["sentiment"] = "중립"
        except:
            pass
        
        # 2. 애널리스트 목표가 (다중 백업 + Fallback 로직 적용)
        info = {}
        try:
            info = t_obj.info
        except:
            pass # info 가져오기 실패해도 계속 진행
            
        target = info.get('targetMeanPrice') or info.get('targetMedianPrice') or info.get('targetHighPrice')
        
        source_label = "🏦Analyst"
        
        # yfinance 데이터가 없거나 0이면 Fallback 실행
        if not target or float(target) == 0:
            target, source_label = get_target_price_fallback(s, curr_p, df_hist)
        
        if target and float(target) > 0:
            data["target"] = float(target)
            upside_val = ((target / curr_p) - 1) * 100
            data["upside"] = f"{upside_val:.1f}"
            data["upside_tag"] = f"({source_label})"
            
            if upside_val > 15: data["score"] += 15
        else:
            data["upside"] = "N/A"

        # 3. 실적 발표일
        try:
            cal = t_obj.calendar
            e_date = None
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                # yfinance 버전 차이 대응
                if 0 in cal.index: e_date = cal.iloc[0, 0]
                elif 'Earnings Date' in cal.columns: e_date = cal['Earnings Date'].iloc[0]
            elif isinstance(cal, dict) and 'Earnings Date' in cal:
                e_date = cal['Earnings Date'][0]

            if e_date:
                e_date_obj = pd.to_datetime(e_date).replace(tzinfo=None)
                days = (e_date_obj - datetime.now().replace(tzinfo=None)).days
                if 0 <= days <= 7: 
                    data["earnings"] = f"⚠️D-{days}"
                    data["score"] -= 40
        except:
            pass
            
    except Exception as e:
        print(f"External Data Error ({s}): {e}")
        
    return data

# [4. 메인 분석 엔진]
def run_full_scan():
    print("🚀 NASDAQ Master-Quant System Starting...")
    if not TELEGRAM_TOKEN or not CHAT_ID: 
        print("토큰 설정 확인 필요")
        return
        
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    
    vix, m_perf = get_market_status()
    
    is_risky = float(vix) > 24.0 or float(m_perf) < -1.5
    risk_mode = "⚠️방어운전" if is_risky else "✅안정적"
    score_min = 45 if risk_mode == "⚠️방어운전" else 30
    
    review_list, super_buys, strong_buys, normal_buys = [], [], [], []
    sector_momentum = {k: 0 for k in SECTORS.keys()}
    results = []

    for idx, s in enumerate(STOCKS):
        print(f"[{idx+1}/{len(STOCKS)}] {s} Analyzing...", end='\r')
        try:
            t_obj = yf.Ticker(s)
            df = flatten_df(t_obj.history(period="150d")) # 기간 조금 늘림
            if len(df) < 30: continue
            
            df = calculate_indicators(df)
            curr_p = float(df['Close'].iloc[-1])
            
            # RSI 과매도 복기
            if df['RSI'].iloc[-2] < 35:
                hit = float(df['High'].iloc[-1]) >= float(df['Close'].iloc[-2]) * 1.025
                review_list.append(f"{s}:{'🎯' if hit else '⏳'}")

            high_52 = float(df['High'].max())
            drop_rate = float((1 - (curr_p / high_52)) * 100)
            is_turning = bool(df['MACD'].iloc[-1] > df['Signal'].iloc[-1])
            is_vol = bool(df['Volume'].iloc[-1] > df['Volume'].rolling(5).mean().iloc[-1] * 1.5)
            is_bb_support = bool(curr_p <= df['BB_Low'].iloc[-1] * 1.02)
            
            # 여기서 df를 넘겨서 기술적 목표가 계산 가능하게 함
            external = get_external_data(s, t_obj, curr_p, df)
            
            if is_vol and curr_p > float(df['Close'].iloc[-2]):
                for s_name, stocks in SECTORS.items():
                    if s in stocks: sector_momentum[s_name] += 1

            results.append({
                "symbol": s, "price": curr_p, "rsi": float(df['RSI'].iloc[-1]), 
                "mfi": float(df['MFI'].iloc[-1]), "drop": drop_rate, 
                "is_turning": is_turning, "is_vol": is_vol, "is_bb": is_bb_support,
                "external": external, "df": df
            })
            time.sleep(0.1) # API 부하 방지
        except Exception as e:
            # print(f"\nError analyzing {s}: {e}")
            continue

    hot_sectors = [k for k, v in sector_momentum.items() if v >= 2]
    
    for item in results:
        s = item['symbol']
        theme_bonus = 15 if any(s in SECTORS[hs] for hs in hot_sectors) else 0
        
        # Upside가 확실히 있으면 점수 부여
        upside_bonus = 0
        if item['external']['upside'] != "N/A" and float(item['external']['upside']) > 20:
            upside_bonus = 10
        
        total_score = item['external']['score'] + theme_bonus + upside_bonus + \
                      (25 if item['rsi'] < 35 else 0) + \
                      (10 if item['is_turning'] else 0) + \
                      (15 if item['is_vol'] else 0) + \
                      (10 if item['drop'] > 30 else 0) + \
                      (10 if item['is_bb'] else 0)

        # 트레이딩 가이드라인 계산
        atr = (item['df']['High'] - item['df']['Low']).rolling(14).mean().iloc[-1]
        t1 = item['price'] + (atr * 2)
        t2 = item['price'] + (atr * 4)
        stop = item['price'] - (atr * 1.5)
        
        upside_str = f"{item['external']['upside']}%" if item['external']['upside'] != "N/A" else "N/A"
        upside_tag = item['external']['upside_tag']
        
        bb_status = "🌕하단" if item['is_bb'] else "🌑정상"
        vol_status = "🔥폭발" if item['is_vol'] else "💤보통"
        
        t_link = f"https://tossinvest.com/stocks/{s}"
        
        # 메시지 포맷 개선
        msg = (f"🔥 **`{s}`** (점수:{total_score})\n"
               f"📍 Price: ${item['price']:.2f} (RSI:{item['rsi']:.1f})\n"
               f"🎯 TP: ${t1:.2f} | 🆙 Upside: {upside_str} {upside_tag}\n"
               f"📊 뉴스:{item['external']['sentiment']} | 낙폭:{item['drop']:.1f}%\n"
               f"🚩 수급:{vol_status} | BB:{bb_status} | 🏛 실적:{item['external']['earnings']}\n"
               f"🛑 Stop: ${stop:.2f}\n"
               f"🔗 [토스증권 바로가기]({t_link})")

        if "⚠️" in item['external']['earnings']: continue
        
        if total_score >= 70 and item['is_vol'] and risk_mode == "✅안정적":
            super_buys.append(msg)
        elif total_score >= 55:
            strong_buys.append(msg)
        elif total_score >= score_min:
            normal_buys.append(msg)

    # 보고서 전송
    header = [
        f"🇺🇸 *NASDAQ PRO MASTER REPORT*",
        f"📅 {now.strftime('%Y-%m-%d %H:%M')} | {risk_mode}",
        f"📉 VIX: {vix:.2f} | Market: {m_perf:+.2f}%",
        f"🚩 Hot Sectors: {', '.join(hot_sectors) if hot_sectors else '없음'}",
        "━━━━━━━━━━━━━━",
        f"📊 **[전일 RSI 과매도 적중률]**\n" + (", ".join(review_list[:8]) if review_list else "데이터 부족"),
        "━━━━━━━━━━━━━━"
    ]
    
    full_text = "\n".join(header + 
                ([f"🚀 **[SUPER BUY]** - 강력 추천\n" + "\n\n".join(super_buys[:3])] if super_buys else []) +
                ([f"\n💎 **[STRONG BUY]** - 매수 유효\n" + "\n\n".join(strong_buys[:5])] if strong_buys else []) +
                ([f"\n🔍 **[NORMAL BUY]** - 관망/소액\n" + "\n\n".join(normal_buys[:8])] if normal_buys else []) +
                ["━━━━━━━━━━━━━━", f"✅ {len(results)}개 종목 분석 완료"])

    print("\nSending Telegram Report...")
    # 긴 메시지 분할 전송
    for part in [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      json={"chat_id": CHAT_ID, "text": part, "parse_mode": "Markdown", "disable_web_page_preview": True})
    print("Done.")

if __name__ == "__main__":
    run_full_scan()






