import os
import yfinance as yf
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai

# 환경 설정
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

STOCKS = [
    "QQQ", "TQQQ", "SQQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "AMD", "SOXL", "SOXS", "AVGO", "NFLX", "TSM", "ADBE", "COST", "PEP", "AZN", 
    "LIN", "CSCO", "TMUS", "INTC", "TXN", "QCOM", "AMAT", "ADP", "ISRG", "SBUX", 
    "MDLZ", "GILD", "INTU", "VRTX", "AMGN", "REGN", "PYPL", "FISV", "BKNG", "CSX", 
    "MU", "PANW", "SNPS", "CDNS", "ORLY", "MNST", "MAR", "KDP", "CHTR", "KLAC", 
    "AEP", "LRCX", "ADSK", "DXCM", "MELI", "IDXX", "PAYX", "CTAS", "LULU", "MCHP", 
    "MRVL", "CPRT", "ODFL", "TEAM", "ALGN", "WDAY", "FAST", "PCAR", "ROST", "DLTR", 
    "EBAY", "SIRI", "ZM", "JD", "LCID", "DDOG", "RIVN", "ENPH", "CEG", "ZS", 
    "ABNB", "PDD", "OKTA", "CONL", "NVDL", "TSLL", "SOXX", "SCHD", "JEPI", "VOO", 
    "IVV", "VTI", "UPRO", "TMF", "ARM", "PLTR", "SNOW", "U", "COIN", "MSTR"
]

def get_optimized_stocks(log_file, blacklist_file, original_stocks):
    market_recovery = False
    try:
        # 시장 지표는 지연이 적은 1분봉 최신 데이터로 확인
        market_df = yf.download("QQQ", period="1d", interval="1m", progress=False)
        if not market_df.empty:
            if isinstance(market_df.columns, pd.MultiIndex): market_df.columns = market_df.columns.get_level_values(0)
            current_qqq = market_df['Close'].iloc[-1]
            # 20분 이동평균 대용으로 당일 시가와 비교
            market_recovery = current_qqq > market_df['Open'].iloc[0]
    except: pass
    
    if not os.path.exists(log_file): return original_stocks, market_recovery
    try:
        df = pd.read_csv(log_file)
        perf = df.groupby('종목')['목표가달성'].apply(lambda x: (x == 'YES').mean())
        count = df.groupby('종목').size()
        eval_stocks = count[count >= 5].index.tolist()
        bad_stocks = [s for s in eval_stocks if perf[s] < 0.3]
        if bad_stocks:
            with open(blacklist_file, 'w') as f: json.dump(bad_stocks, f)
        return [s for s in original_stocks if s not in bad_stocks], market_recovery
    except: return original_stocks, market_recovery

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
    exp1 = series.ewm(span=12, adjust=False).mean(); exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2; signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def get_comprehensive_data(s, t_obj):
    analysis = {"sentiment": "중립", "earnings": "안정", "option": "중립", "score": 0}
    try:
        news = t_obj.news[:5]
        if news and GEMINI_API_KEY:
            titles = [n['title'] for n in news]
            prompt = f"Analyze {s}: {titles}. One word: Positive, Negative, Neutral."
            response = model.generate_content(prompt)
            res = response.text.strip().capitalize()
            analysis["sentiment"] = "호재" if "Positive" in res else "악재" if "Negative" in res else "중립"
            if analysis["sentiment"] == "호재": analysis["score"] += 20
    except: pass
    try:
        cal = t_obj.calendar
        if cal is not None and 'Earnings Date' in cal:
            days = (cal['Earnings Date'][0].replace(tzinfo=None) - datetime.now()).days
            if 0 <= days <= 7:
                analysis["earnings"] = f"⚠️D-{days}"
                analysis["score"] -= 30
    except: pass
    return analysis

def run_analysis():
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul'); now = datetime.now(kst)
    optimized_stocks, market_recovery = get_optimized_stocks('trade_log_nasdaq.csv', 'blacklist_nasdaq.json', STOCKS)
    review_reports, super_buys, strong_buys, normal_buys, trade_logs, total_analyzed, down_count, temp_data = [], [], [], [], [], 0, 0, []

    for s in optimized_stocks:
        try:
            t_obj = yf.Ticker(s)
            # [시정 사항] 50일치 일봉 데이터와 별도로, 가장 최신 1분봉 데이터를 추가로 가져옴
            df = t_obj.history(period="50d")
            recent_tick = t_obj.history(period="1d", interval="1m")
            
            if len(df) < 30 or recent_tick.empty: continue
            
            # 현재가를 1분봉의 가장 마지막 가격으로 교체 (지연 최소화)
            realtime_price = float(recent_tick['Close'].iloc[-1])
            total_analyzed += 1
            if realtime_price < float(df['Close'].rolling(20).mean().iloc[-1]): down_count += 1
            temp_data.append((s, df, t_obj, realtime_price))
        except: continue

    ratio = down_count / total_analyzed if total_analyzed > 0 else 0.5
    t1, t2 = (1.025, 1.050) if ratio < 0.3 else (1.015, 1.030) if ratio < 0.6 else (1.008, 1.015)
    mode_str = "🚀 불장" if ratio < 0.3 else "📈 보통" if ratio < 0.6 else "⚠️ 하락"

    for s, df, t_obj, curr_p in temp_data:
        try:
            close = df['Close']; prev_p = float(close.iloc[-1]) # 어제 종가
            high_p = float(df['High'].iloc[-1]); vol = df['Volume']
            
            # 복기 (어제 추천주가 오늘 고가 기준으로 도달했는지)
            if calculate_rsi(close).iloc[-2] < 36:
                hit1, hit2 = high_p >= prev_p * t1, high_p >= prev_p * t2
                status = "🎯" if hit2 else ("🌗" if hit1 else "⏳")
                review_reports.append(f"{s}:{status}")
                trade_logs.append({"날짜": now.strftime('%Y-%m-%d'), "종목": s, "목표가달성": "YES" if hit2 else "NO"})

            extra = get_comprehensive_data(s, t_obj)
            rsi = float(calculate_rsi(close).iloc[-1])
            mfi = float(calculate_mfi(df).iloc[-1])
            macd, signal = calculate_macd(close)
            
            is_vol_spike = vol.iloc[-1] > vol.rolling(5).mean().iloc[-1] * 1.2
            is_oversold = rsi < 36 or curr_p <= float((close.rolling(20).mean() - (close.rolling(20).std() * 2)).iloc[-1])
            is_money_in = mfi < 40
            is_turning = float(macd.iloc[-1]) > float(signal.iloc[-1])
            
            stop_loss = curr_p * 0.98 
            toss_link = f"https://tossinvest.com/stocks/{s}"
            
            t_info = (f"🔥 **`{s}`** (복사)\n"
                      f"📍 Buy: ${curr_p:.2f}\n🎯 Target: ${curr_p * t1:.2f} / ${curr_p * t2:.2f}\n"
                      f"🛑 Stop: ${stop_loss:.2f}\n"
                      f"📊 뉴스:{extra['sentiment']} | 실적:{extra['earnings']}\n"
                      f"🔗 [크롬/앱으로 주문하기]({toss_link})")
            
            if "⚠️" in extra['earnings']: continue 
            if is_oversold and is_money_in and is_turning and is_vol_spike and market_recovery:
                super_buys.append(t_info)
            elif is_oversold and is_money_in and (is_vol_spike or market_recovery or extra['score'] > 20):
                strong_buys.append(t_info)
            elif is_oversold:
                normal_buys.append(t_info)
        except: continue

    if trade_logs: pd.DataFrame(trade_logs).to_csv('trade_log_nasdaq.csv', mode='a', index=False, header=not os.path.exists('trade_log_nasdaq.csv'), encoding='utf-8-sig')
    
    report = [
        f"🇺🇸 *NASDAQ PRO AI*", f"📅 {now.strftime('%m-%d %H:%M')} | {mode_str}", "━━━━━━━━━━━━━━",
        f"📊 **[전일 복기]**\n" + (", ".join(review_reports[:15]) if review_reports else "-"), "━━━━━━━━━━━━━━",
        f"🎯 **[SUPER BUY]**\n" + ("\n\n".join(super_buys[:5]) if super_buys else "- 없음"),
        f"\n💎 **[STRONG BUY]**\n" + ("\n\n".join(strong_buys[:10]) if strong_buys else "- 없음"),
        f"\n🔍 **[NORMAL BUY]**\n" + ("\n\n".join(normal_buys[:15]) if normal_buys else "- 없음"), "━━━━━━━━━━━━━━",
        f"✅ {total_analyzed}분석 (시장점수: {int((1-ratio)*100)}점)"
    ]
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": "\n".join(report), "parse_mode": "Markdown", "disable_web_page_preview": True})

if __name__ == "__main__":
    run_analysis()







