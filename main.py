import os
import yfinance as yf
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai

# [설정] 환경 변수
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

STOCKS = ["QQQ", "TQQQ", "SQQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "SOXL", "SOXS", "AVGO", "NFLX", "TSM", "ADBE", "COST", "PEP", "AZN", "LIN", "CSCO", "TMUS", "INTC", "TXN", "QCOM", "AMAT", "ADP", "ISRG", "SBUX", "MDLZ", "GILD", "INTU", "VRTX", "AMGN", "REGN", "PYPL", "FISV", "BKNG", "CSX", "MU", "PANW", "SNPS", "CDNS", "ORLY", "MNST", "MAR", "KDP", "CHTR", "KLAC", "AEP", "LRCX", "ADSK", "DXCM", "MELI", "IDXX", "PAYX", "CTAS", "LULU", "MCHP", "MRVL", "CPRT", "ODFL", "TEAM", "ALGN", "WDAY", "FAST", "PCAR", "ROST", "DLTR", "EBAY", "SIRI", "ZM", "JD", "LCID", "DDOG", "RIVN", "ENPH", "CEG", "ZS", "ABNB", "PDD", "OKTA", "CONL", "NVDL", "TSLL", "SOXX", "SCHD", "JEPI", "VOO", "IVV", "VTI", "UPRO", "TMF", "ARM", "PLTR", "SNOW", "U", "COIN", "MSTR"]

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

def get_comprehensive_data(s, t_obj):
    analysis = {"sentiment": "중립", "earnings": "안정", "option": "중립", "score": 0}
    try:
        news = t_obj.news[:3]
        if news and GEMINI_API_KEY:
            titles = [n['title'] for n in news]
            prompt = f"Analyze stock {s}: {titles}. Respond ONLY with one word: Positive, Negative, or Neutral."
            res = model.generate_content(prompt).text.strip().capitalize()
            analysis["sentiment"] = "호재" if "Positive" in res else "악재" if "Negative" in res else "중립"
            if analysis["sentiment"] == "호재": analysis["score"] += 20
    except: pass
    try:
        cal = t_obj.calendar
        e_date = cal['Earnings Date'][0] if isinstance(cal, dict) else cal.iloc[0][0]
        # 시간대 에러 방지 (둘 다 naive로 변환)
        days = (pd.to_datetime(e_date).replace(tzinfo=None) - datetime.now().replace(tzinfo=None)).days
        if 0 <= days <= 7: 
            analysis["earnings"] = f"⚠️D-{days}"
            analysis["score"] -= 40
    except: pass
    try:
        opt_info = t_obj.option_chain(t_obj.options[0])
        pc_ratio = opt_info.puts['volume'].sum() / opt_info.calls['volume'].sum()
        analysis["option"] = "상승베팅" if pc_ratio < 0.7 else "하락베팅" if pc_ratio > 1.3 else "중립"
        if pc_ratio < 0.7: analysis["score"] += 15
    except: pass
    return analysis

def run_analysis():
    print("🚀 Full-Scan Analysis started...")
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul'); now = datetime.now(kst)
    
    market_df = flatten_df(yf.download("QQQ", period="5d", progress=False))
    market_recovery = (market_df['Close'].iloc[-1] > market_df['Close'].iloc[-2]) if not market_df.empty else False
    
    review_reports, super_buys, strong_buys, normal_buys, trade_logs = [], [], [], [], []
    total_analyzed, down_count = 0, 0

    for idx, s in enumerate(STOCKS):
        print(f"[{idx+1}/{len(STOCKS)}] {s}", end='\r')
        try:
            t_obj = yf.Ticker(s)
            df = flatten_df(t_obj.history(period="60d"))
            if len(df) < 30: continue
            
            recent = t_obj.history(period="1d", interval="1m")
            curr_p = float(recent['Close'].iloc[-1]) if not recent.empty else float(df['Close'].iloc[-1])
            total_analyzed += 1
            
            # [수정] 복기 로직 통합 (어제 RSI 36 미만 종목의 성적 체크)
            rsi_series = calculate_rsi(df['Close'])
            if len(rsi_series) > 1 and rsi_series.iloc[-2] < 36:
                high_today = float(df['High'].iloc[-1])
                prev_close = float(df['Close'].iloc[-2])
                hit = high_today >= prev_close * 1.025 # 어제 종가 대비 2.5% 이상 상승 시 성공
                review_reports.append(f"{s}:{'🎯' if hit else '⏳'}")

            # 기술 지표
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            if curr_p < ma20: down_count += 1
            mfi = float(calculate_mfi(df).iloc[-1])
            macd, signal = calculate_macd(df['Close'])
            is_oversold = rsi_series.iloc[-1] < 36 or curr_p <= (ma20 - (df['Close'].rolling(20).std().iloc[-1] * 2))
            is_turning = float(macd.iloc[-1]) > float(signal.iloc[-1])
            is_vol_spike = float(df['Volume'].iloc[-1]) > float(df['Volume'].rolling(5).mean().iloc[-1]) * 1.2
            
            extra = get_comprehensive_data(s, t_obj)
            total_score = extra['score'] + (20 if is_oversold else 0) + (10 if is_turning else 0) + (10 if is_vol_spike else 0)
            
            # ATR 기반 목표가
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            t1_p, t2_p, stop_p = curr_p + (atr * 1.5), curr_p + (atr * 3.0), curr_p - (atr * 1.0)
            
            toss_link = f"https://tossinvest.com/stocks/{s}"
            t_info = (f"🔥 **`{s}`** (점수:{total_score})\n📍 Buy: ${curr_p:.2f} (RSI:{rsi_series.iloc[-1]:.1f})\n"
                      f"🎯 Target: ${t1_p:.2f} / ${t2_p:.2f}\n🛑 Stop: ${stop_p:.2f}\n"
                      f"📊 뉴스:{extra['sentiment']} | 실적:{extra['earnings']} | 옵션:{extra['option']}\n"
                      f"🔗 [주문하기]({toss_link})")

            if "⚠️" in extra['earnings']: continue
            
            if is_oversold and mfi < 40 and is_turning and is_vol_spike and market_recovery and total_score > 30:
                super_buys.append(t_info)
                trade_logs.append({"날짜": now.strftime('%Y-%m-%d'), "종목": s, "목표가달성": "ING"})
            elif is_oversold and (mfi < 40 or is_turning) and (is_vol_spike or total_score > 20):
                strong_buys.append(t_info)
            elif is_oversold or total_score > 50:
                normal_buys.append(t_info)
            
            time.sleep(0.4) # API 속도 최적화
        except: continue

    # 결과 전송
    ratio = down_count / total_analyzed if total_analyzed > 0 else 0.5
    mode_str = "🚀 불장" if ratio < 0.3 else "📈 보통" if ratio < 0.6 else "⚠️ 하락"
    
    report = [
        f"🇺🇸 *NASDAQ PRO AI (Full-Scan)*", f"📅 {now.strftime('%m-%d %H:%M')} | {mode_str}", "━━━━━━━━━━━━━━",
        f"📊 **[전일 복기]**\n" + (", ".join(review_reports[:15]) if review_reports else "- 분석 데이터 없음"), "━━━━━━━━━━━━━━",
        f"🎯 **[SUPER BUY]**\n" + ("\n\n".join(super_buys[:5]) if super_buys else "- 없음"),
        f"\n💎 **[STRONG BUY]**\n" + ("\n\n".join(strong_buys[:7]) if strong_buys else "- 없음"),
        f"\n🔍 **[NORMAL BUY]**\n" + ("\n\n".join(normal_buys[:10]) if normal_buys else "- 없음"), "━━━━━━━━━━━━━━",
        f"✅ {total_analyzed}개 분석 (시장점수: {int((1-ratio)*100)}점)"
    ]
    
    full_text = "\n".join(report)
    for part in [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": part, "parse_mode": "Markdown", "disable_web_page_preview": True})

if __name__ == "__main__":
    run_analysis()








