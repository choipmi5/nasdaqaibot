import os
import yfinance as yf
import pandas as pd
import requests
import json
from datetime import datetime
import pytz
import google.generativeai as genai

# --- [1. 환경 설정] ---
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

HANTU_APP_KEY = os.environ.get('HANTU_APP_KEY')
HANTU_SECRET_KEY = os.environ.get('HANTU_SECRET_KEY')
HANTU_ACC_NO = os.environ.get('HANTU_ACCOUNT_NO')
HANTU_BASE_URL = "https://openapivts.koreainvestment.com:29443"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

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

# --- [2. 한투 및 지표 함수들 (생략 없이 유지)] ---
def get_hantu_token():
    try:
        url = f"{HANTU_BASE_URL}/oauth2/tokenP"
        body = {"grant_type": "client_credentials", "appkey": HANTU_APP_KEY, "secretkey": HANTU_SECRET_KEY}
        res = requests.post(url, data=json.dumps(body))
        return res.json().get('access_token')
    except: return None

def buy_stock(symbol, token):
    try:
        url = f"{HANTU_BASE_URL}/uapi/google-nasdaq/v1/trading/order"
        headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":HANTU_APP_KEY, "secretkey":HANTU_SECRET_KEY, "tr_id":"VTTT1002U", "custtype":"P"}
        body = {"CANO": HANTU_ACC_NO, "ACNT_PRDT_CD": "01", "OVRS_EXCG_CD": "NASD", "PDNO": symbol, "ORD_QTY": "1", "OVRS_ORD_UNPR": "0", "ORD_DVSN": "00"}
        res = requests.post(url, headers=headers, data=json.dumps(body))
        return res.json()
    except: return {"rt_cd": "1"}

def calculate_rsi(series, period=14):
    delta = series.diff(); gain = (delta.where(delta > 0, 0)).rolling(period).mean(); loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + (gain / loss)))

def calculate_mfi(df, period=14):
    tp = (df['High'] + df['Low'] + df['Close']) / 3; mf = tp * df['Volume']
    pos_f = mf.where(tp > tp.shift(1), 0).rolling(period).sum(); neg_f = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    return 100 - (100 / (1 + (pos_f / neg_f)))

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean(); exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2; signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def get_optimized_stocks(log_file, blacklist_file, original_stocks):
    market_recovery = False
    try:
        market_df = yf.download("QQQ", period="50d", progress=False)
        if isinstance(market_df.columns, pd.MultiIndex): market_df.columns = market_df.columns.get_level_values(0)
        market_recovery = market_df['Close'].iloc[-1] > market_df['Close'].rolling(20).mean().iloc[-1]
    except: pass
    return original_stocks, market_recovery

def get_advanced_data(s, ticker_obj):
    analysis = {"sentiment": "중립", "earnings": "안정", "option": "중립"}
    try:
        news = ticker_obj.news[:3]
        if news and GEMINI_API_KEY:
            titles = [n['title'] for n in news]
            prompt = f"Sentiment for {s}: {titles}. ONE word: Positive, Negative, Neutral."
            res = gemini_model.generate_content(prompt).text.strip().capitalize()
            analysis["sentiment"] = "호재" if "Positive" in res else "악재" if "Negative" in res else "중립"
    except: pass
    try:
        cal = ticker_obj.calendar
        if cal is not None and 'Earnings Date' in cal:
            days = (cal['Earnings Date'][0].replace(tzinfo=None) - datetime.now()).days
            if 0 <= days <= 5: analysis["earnings"] = f"⚠️D-{days}"
    except: pass
    try:
        exp = ticker_obj.options[0]
        opt = ticker_obj.option_chain(exp)
        pc_ratio = opt.puts['volume'].sum() / opt.calls['volume'].sum()
        analysis["option"] = "상승베팅" if pc_ratio < 0.7 else "하락베팅" if pc_ratio > 1.3 else "중립"
    except: pass
    return analysis

# --- [3. 메인 실행 및 상세 리포트 구성] ---
def run_analysis():
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul'); now = datetime.now(kst); hantu_token = get_hantu_token()
    
    optimized_stocks, market_recovery = get_optimized_stocks('trade_log_nasdaq.csv', 'blacklist_nasdaq.json', STOCKS)
    review_reports, super_buys, strong_buys, normal_buys, total_analyzed, down_count = [], [], [], [], 0, 0
    temp_data = []

    for s in optimized_stocks:
        try:
            t = yf.Ticker(s); df = t.history(period="50d")
            if len(df) < 30: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            total_analyzed += 1
            if float(df['Close'].iloc[-1]) < float(df['Close'].rolling(20).mean().iloc[-1]): down_count += 1
            temp_data.append((s, df, t))
        except: continue

    ratio = down_count / total_analyzed if total_analyzed > 0 else 0.5
    t1, t2 = (1.025, 1.050) if ratio < 0.3 else (1.015, 1.030) if ratio < 0.6 else (1.007, 1.012)
    mode_str = f"🚀 불장({(t1-1)*100:.1f}/{(t2-1)*100:.1f}%)" if ratio < 0.3 else f"📈 보통({(t1-1)*100:.1f}/{(t2-1)*100:.1f}%)" if ratio < 0.6 else f"⚠️ 하락({(t1-1)*100:.1f}/{(t2-1)*100:.1f}%)"

    for s, df, t_obj in temp_data:
        try:
            close = df['Close']; curr_p, prev_p, high_p = float(close.iloc[-1]), float(close.iloc[-2]), float(df['High'].iloc[-1])
            rsi, mfi = float(calculate_rsi(close).iloc[-1]), float(calculate_mfi(df).iloc[-1])
            macd, signal = calculate_macd(close); is_turning = macd.iloc[-1] > signal.iloc[-1]
            
            # 전일 복기
            if calculate_rsi(close).iloc[-2] < 35:
                hit1, hit2 = high_p >= prev_p * t1, high_p >= prev_p * t2
                review_reports.append(f"{s}:{'🎯' if hit2 else ('🌗' if hit1 else '⏳')}")

            # 상세 정보 생성
            extra = get_advanced_data(s, t_obj)
            stop_p = curr_p * 0.975
            detail = f"📈 *{s}*\n📍 Buy: ${curr_p:.2f}\n🎯 Target: ${curr_p*t1:.2f} / ${curr_p*t2:.2f}\n🛑 Stop: ${stop_p:.2f}\n📊 뉴스:{extra['sentiment']} | 실적:{extra['earnings']} | 옵션:{extra['option']}"

            # 등급별 분류
            is_buy = (rsi < 32 or curr_p <= (close.rolling(20).mean() - close.rolling(20).std()*2).iloc[-1]) and mfi < 35 and is_turning
            
            if is_buy and market_recovery and "⚠️" not in extra['earnings']:
                order_res = buy_stock(s, hantu_token)
                status = " [✅주문]" if order_res.get('rt_cd') == '0' else " [❌실패]"
                super_buys.append(detail + status)
            elif is_buy:
                strong_buys.append(detail)
            elif rsi < 40:
                normal_buys.append(detail)
        except: continue

    # 리포트 조립
    report = [f"🇺🇸 *NASDAQ PRO AI*", f"📅 {now.strftime('%m-%d %H:%M')} | {mode_str}", "━━━━━━━━━━━━━━"]
    report.append(f"📊 **[전일 복기]**\n" + (", ".join(review_reports[:10]) if review_reports else "-"))
    report.append("━━━━━━━━━━━━━━")
    if super_buys: report.append(f"🎯 **[AUTO BUY]**\n" + "\n\n".join(super_buys))
    if strong_buys: report.append(f"\n💎 **[STRONG BUY]**\n" + "\n\n".join(strong_buys[:5]))
    if normal_buys: report.append(f"\n🔍 **[WATCHLIST]**\n" + "\n\n".join(normal_buys[:10]))
    report.append("━━━━━━━━━━━━━━\n" + f"✅ {total_analyzed}분석 (시장점수: {int((1-ratio)*100)}점)")

    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": "\n".join(report), "parse_mode": "Markdown"})

if __name__ == "__main__":
    run_analysis()

