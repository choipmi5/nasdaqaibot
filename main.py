import os
import yfinance as yf
import pandas as pd
import numpy as np # [NEW] 벡터 연산을 위한 numpy 추가
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

# [2. 섹터 및 종목 리스트 (약 250개, ETF 완전 배제)]
SECTORS = {
    "SEMICON": ["NVDA", "AMD", "AVGO", "TSM", "MU", "ARM", "LRCX", "AMAT", "TXN", "QCOM", "INTC", "KLAC", "SNPS", "CDNS", "MRVL"],
    "BIGTECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX"],
    "AI/SW/FIN": ["PLTR", "SNOW", "ADBE", "ORCL", "CRM", "PANW", "COIN", "MSTR", "INTU", "CRWD", "DDOG", "NOW"]
}

# ETF를 제외한 미국 주요 우량주/성장주 약 250개 리스트
STOCKS = list(set([
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "AVGO", "NFLX", "TSM", "ADBE", "COST", "PEP", "AZN", "LIN", "CSCO", 
    "TMUS", "INTC", "TXN", "QCOM", "AMAT", "ADP", "ISRG", "SBUX", "MDLZ", "GILD", "INTU", "VRTX", "AMGN", "REGN", "PYPL", "FISV", "BKNG", 
    "CSX", "MU", "PANW", "SNPS", "CDNS", "ORLY", "MNST", "MAR", "KDP", "CHTR", "KLAC", "AEP", "LRCX", "ADSK", "DXCM", "MELI", "IDXX", 
    "PAYX", "CTAS", "LULU", "MCHP", "MRVL", "CPRT", "ODFL", "TEAM", "ALGN", "WDAY", "FAST", "PCAR", "ROST", "DLTR", "EBAY", "SIRI", 
    "ZM", "JD", "LCID", "DDOG", "RIVN", "ENPH", "CEG", "ZS", "ABNB", "PDD", "OKTA", "ARM", "PLTR", "SNOW", "U", "COIN", "MSTR", 
    "BRK-B", "UNH", "JNJ", "JPM", "V", "PG", "HD", "CVX", "MA", "ABBV", "MRK", "KO", "PFE", "TMO", "MCD", "DIS", "ABT", "WMT", "CRM", 
    "DHR", "NEE", "PM", "BMY", "UNP", "NKE", "RTX", "LOW", "HON", "SPGI", "ORCL", "BA", "IBM", "GS", "ELV", "CAT", "GE", "MDT", "AXP", 
    "DE", "LMT", "BLK", "ADI", "TJX", "SYK", "C", "NOW", "CVS", "ZTS", "CIT", "MMC", "CB", "SO", "DUK", "PGR", "BDX", "BSX", "T", "CI", 
    "EQIX", "SLB", "EOG", "AON", "NOC", "SHW", "WM", "FCX", "ICE", "MCO", "EMR", "EW", "MCK", "CMCSA", "GPN", "PXD", "MPC", "NXPI", 
    "FDX", "VLO", "PH", "KMB", "PSX", "SRE", "ROP", "TEL", "TRV", "MSI", "O", "AIG", "WELL", "AZO", "PSA", "D", "EXC", "TT", "CTVA", 
    "CNC", "AFL", "STZ", "SPG", "WMB", "HLT", "BIIB", "PAYC", "YUM", "FTNT", "DHI", "IQV", "PRU", "SYY", "MTD", "A", "NEM", "CTSH", 
    "GWW", "WBA", "KMI", "BKR", "K", "TGT", "HOOD", "AFRM", "PATH", "MNDY", "DOCN", "NET", "CRWD", "SE", "SQ", "ROKU", "PINS", "TWLO", 
    "SPOT", "UBER", "LYFT", "DASH", "CVNA", "CHWY", "Z", "W", "ETSY", "DKNG", "PENN", "WYNN", "LVS", "MGM", "RCL", "CCL", "NCLH", 
    "DAL", "UAL", "AAL", "LUV", "EXPE", "TRIP", "SHOP", "GLW", "FSLR", "SEDG", "RUN", "PLUG", "FCEL", "QS", "CHPT", "BLNK"
]))

# [3. 핵심 유틸리티 함수 (기존 코드 완벽 유지)]
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_f = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_f = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_f / neg_f)))
    
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    df['MA20'] = df['Close'].rolling(20).mean()
    df['STD'] = df['Close'].rolling(20).std()
    df['BB_Low'] = df['MA20'] - (df['STD'] * 2)
    df['BB_High'] = df['MA20'] + (df['STD'] * 2)
    return df

def get_market_status():
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
    target = None
    source = "N/A"
    try:
        if GEMINI_API_KEY:
            prompt = f"What is the average analyst target price for stock {ticker} in numbers only? If unsure, return 0."
            response = model.generate_content(prompt).text
            numbers = re.findall(r"\d+\.\d+|\d+", response)
            if numbers:
                ai_target = float(numbers[0])
                if ai_target > current_price:
                    return ai_target, "🤖AI"
    except:
        pass

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
        
        info = {}
        try:
            info = t_obj.info
        except:
            pass
            
        target = info.get('targetMeanPrice') or info.get('targetMedianPrice') or info.get('targetHighPrice')
        source_label = "🏦Analyst"
        
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

        try:
            cal = t_obj.calendar
            e_date = None
            if isinstance(cal, pd.DataFrame) and not cal.empty:
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

# [4. 메인 분석 엔진 (최적화 적용)]
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

    # [NEW 1] MSFT 가격 상한선 설정 (내 자금력에 맞춘 필터링)
    try:
        msft_data = flatten_df(yf.Ticker("MSFT").history(period="1d"))
        msft_price_limit = float(msft_data['Close'].iloc[-1])
        print(f"📌 매수 상한선 (MSFT 현재가 기준 적용): ${msft_price_limit:.2f}")
    except:
        msft_price_limit = 450.0 # 에러 시 기본값

    # [NEW 2] 가중치(Weight) 벡터 정의 (순서: RSI과매도, MACD교차기울기, 거래량폭발, 낙폭과대, BB하단)
    WEIGHTS = np.array([25, 10, 15, 10, 10])

    for idx, s in enumerate(STOCKS):
        print(f"[{idx+1}/{len(STOCKS)}] {s} Analyzing...", end='\r')
        try:
            t_obj = yf.Ticker(s)
            df = flatten_df(t_obj.history(period="150d"))
            if len(df) < 30: continue
            
            df = calculate_indicators(df)
            curr_p = float(df['Close'].iloc[-1])

            # [조건 검사 1] 마이크로소프트(MSFT) 주가보다 비싼 종목 패스
            if curr_p > msft_price_limit:
                continue
            
            # RSI 과매도 복기
            if df['RSI'].iloc[-2] < 35:
                hit = float(df['High'].iloc[-1]) >= float(df['Close'].iloc[-2]) * 1.025
                review_list.append(f"{s}:{'🎯' if hit else '⏳'}")

            high_52 = float(df['High'].max())
            drop_rate = float((1 - (curr_p / high_52)) * 100)
            
            # [NEW 3] 모멘텀 기울기 분석 (최근 5일간 MACD 기울기)
            macd_slope = (df['MACD'].iloc[-1] - df['MACD'].iloc[-5]) / 5
            is_turning = bool(df['MACD'].iloc[-1] > df['Signal'].iloc[-1])
            
            # 기울기가 양수(상승 가속)일 때만 추가 가중치(Multiplier) 적용 (최대 1.5배)
            slope_multiplier = min(max(macd_slope, 0) * 10, 1.5) if is_turning else 0
            is_turning_score = 1.0 + slope_multiplier if is_turning else 0.0

            is_vol = bool(df['Volume'].iloc[-1] > df['Volume'].rolling(5).mean().iloc[-1] * 1.5)
            is_bb_support = bool(curr_p <= df['BB_Low'].iloc[-1] * 1.02)
            
            if is_vol and curr_p > float(df['Close'].iloc[-2]):
                for s_name, stocks in SECTORS.items():
                    if s in stocks: sector_momentum[s_name] += 1

            # [NEW 4] Feature 벡터를 통한 행렬(내적) 연산
            features = np.array([
                1.0 if df['RSI'].iloc[-1] < 35 else 0.0,
                is_turning_score,
                1.0 if is_vol else 0.0,
                1.0 if drop_rate > 30 else 0.0,
                1.0 if is_bb_support else 0.0
            ])
            tech_score = float(np.dot(features, WEIGHTS)) # 기술적 베이스 점수

            # [NEW 5] API 지연 호출 (Lazy Evaluation)
            # 기술적 점수가 20점 미만이면 가망이 없으므로 무거운 API 호출 생략 (실행 속도 수십 배 향상)
            if tech_score >= 20:
                external = get_external_data(s, t_obj, curr_p, df)
            else:
                external = {"sentiment": "➖생략", "earnings": "➖", "upside": "N/A", "upside_tag": "", "score": 0}

            results.append({
                "symbol": s, "price": curr_p, "rsi": float(df['RSI'].iloc[-1]), 
                "drop": drop_rate, "is_turning": is_turning, "is_vol": is_vol, "is_bb": is_bb_support,
                "external": external, "df": df, "tech_score": tech_score
            })
            time.sleep(0.05) # 속도가 빨라졌으므로 딜레이 약간 단축
        except Exception as e:
            continue

    hot_sectors = [k for k, v in sector_momentum.items() if v >= 2]
    
    for item in results:
        s = item['symbol']
        theme_bonus = 15 if any(s in SECTORS[hs] for hs in hot_sectors) else 0
        
        upside_bonus = 0
        if item['external']['upside'] != "N/A" and float(item['external']['upside']) > 20:
            upside_bonus = 10
        
        # 합산: 백테스팅 시 조율할 수 있도록 분리된 구조
        total_score = item['tech_score'] + item['external']['score'] + theme_bonus + upside_bonus

        atr = (item['df']['High'] - item['df']['Low']).rolling(14).mean().iloc[-1]
        t1 = item['price'] + (atr * 2)
        t2 = item['price'] + (atr * 4)
        stop = item['price'] - (atr * 1.5)
        
        upside_str = f"{item['external']['upside']}%" if item['external']['upside'] != "N/A" else "N/A"
        upside_tag = item['external']['upside_tag']
        
        bb_status = "🌕하단" if item['is_bb'] else "🌑정상"
        vol_status = "🔥폭발" if item['is_vol'] else "💤보통"
        
        t_link = f"https://tossinvest.com/stocks/{s}"
        
        msg = (f"🔥 **`{s}`** (점수:{total_score:.1f})\n"
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
    for part in [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      json={"chat_id": CHAT_ID, "text": part, "parse_mode": "Markdown", "disable_web_page_preview": True})
    print("Done.")

if __name__ == "__main__":
    run_full_scan()






