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

# 한투 API 설정 (모의투자용)
HANTU_APP_KEY = os.environ.get('HANTU_APP_KEY')
HANTU_SECRET_KEY = os.environ.get('HANTU_SECRET_KEY')
HANTU_ACC_NO = os.environ.get('HANTU_ACCOUNT_NO')
HANTU_BASE_URL = "https://openapivts.koreainvestment.com:29443"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# 원본 100개 종목 리스트 복구
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

# --- [2. 한투 매매 함수] ---
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

# --- [3. 보조 지표 함수] ---
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
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def get_optimized_stocks(log_file, blacklist_file, original_stocks):
    market_recovery = False
    try:
        market_df = yf.download("QQQ", period="50d", progress=False)
        if isinstance(market_df.columns, pd.MultiIndex): market_df.columns = market_df.columns.get_level_values(0)
        market_recovery = market_df['Close'].iloc[-1] > market_df['Close'].rolling(20).mean().iloc[-1]
    except: pass
    if not os.path.exists(log_file): return original_stocks, market_recovery
    try:
        df = pd.read_csv(log_file)
        perf = df.groupby('종목')['목표가달성'].apply(lambda x: (x == 'YES').mean())
        count = df.groupby('종목').size()
        eval_stocks = count[count >= 10].index.tolist()
        bad_stocks = [s for s in eval_stocks if perf[s] < 0.3]
        if not market_recovery: bad_stocks.extend([s for s in eval_stocks if 0.3 <= perf[s] < 0.5])
        with open(blacklist_file, 'w') as f: json.dump(list(set(bad_stocks)), f)
        return [s for s in original_stocks if s not in bad_stocks], market_recovery
    except: return original_stocks, market_recovery

# --- [4. AI 데이터 분석] ---
def get_advanced_data(s, ticker_obj):
    analysis = {"sentiment": "중립", "earnings": "안정"}
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
    return analysis

# --- [5. 메인 실행] ---
def run_analysis():
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    hantu_token = get_hantu_token()

    optimized_stocks, market_recovery = get_optimized_stocks('trade_log_nasdaq.csv', 'blacklist_nasdaq.json', STOCKS)
    review_reports, super_buys, strong_buys, trade_logs, total_analyzed, down_count, temp_data = [], [], [], [], 0, 0, []

    for s in optimized_stocks:
        try:
            df = yf.download(s, period="50d", progress=False)
            if len(df) < 30: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            total_analyzed += 1
            if float(df['Close'].iloc[-1]) < float(df['Close'].rolling(20).mean().iloc[-1]): down_count += 1
            temp_data.append((s, df))
        except: continue

    ratio = down_count / total_analyzed if total_analyzed > 0 else 0.5
    t1, t2 = (1.025, 1.050) if ratio < 0.3 else (1.015, 1.030) if ratio < 0.6 else (1.007, 1.012)
    mode_str = "🚀 불장" if ratio < 0.3 else "📈 보통" if ratio < 0.6 else "⚠️ 하락"

    for s, df in temp_data:
        try:
            close = df['Close']
            curr_p, prev_p, high_p = float(close.iloc[-1]), float(close.iloc[-2]), float(df['High'].iloc[-1])
            
            # 1. 복기 로직
            if calculate_rsi(close).iloc[-2] < 35:
                hit = high_p >= prev_p * t2
                review_reports.append(f"{s}:{'🎯' if hit else '⏳'}")
                trade_logs.append({"날짜": now.strftime('%Y-%m-%d'), "종목": s, "목표가달성": "YES" if hit else "NO"})

            # 2. 지표 계산 및 AI 분석
            rsi, mfi = float(calculate_rsi(close).iloc[-1]), float(calculate_mfi(df).iloc[-1])
            macd, signal = calculate_macd(close)
            is_turning = macd.iloc[-1] > signal.iloc[-1]
            is_vol_spike = df['Volume'].iloc[-1] > df['Volume'].rolling(5).mean().iloc[-1] * 1.2
            
            t_obj = yf.Ticker(s)
            extra = get_advanced_data(s, t_obj)
            
            # 3. 매수 조건 (원본 차트 지표 + AI + 시장회복)
            is_buy = (rsi < 32 or curr_p <= (close.rolling(20).mean() - close.rolling(20).std()*2).iloc[-1]) and mfi < 35 and is_turning

            if is_buy and market_recovery and "⚠️" not in extra['earnings']:
                # [자동매매 실행]
                order_res = buy_stock(s, hantu_token)
                status = "✅" if order_res.get('rt_cd') == '0' else "❌"
                super_buys.append(f"🔥 *{s}* {status}\n📍 ${curr_p:.2f} | 📊 {extra['sentiment']}")
            elif is_buy:
                strong_buys.append(f"💎 *{s}*\n📍 ${curr_p:.2f} | 📊 {extra['sentiment']}")
        except: continue

    if trade_logs: pd.DataFrame(trade_logs).to_csv('trade_log_nasdaq.csv', mode='a', index=False, header=not os.path.exists('trade_log_nasdaq.csv'), encoding='utf-8-sig')
    
        # 리포트 발송 부분 수정 (상세 정보 포함)
    report = [
        f"🇺🇸 *NASDAQ PRO AI*", 
        f"📅 {now.strftime('%m-%d %H:%M')} | {mode_str}", 
        "━━━━━━━━━━━━━━",
        f"📊 **[전일 복기]**\n" + (", ".join(review_reports[:10]) if review_reports else "-")
    ]

    if super_buys:
        report.append(f"\n🎯 **[AUTO BUY]** (자동 주문 완료)\n" + "\n".join(super_buys))
    
    if strong_buys:
        report.append(f"\n💎 **[STRONG BUY]** (강력 추천)\n" + "\n".join(strong_buys))

    # [수정] NORMAL BUY 섹션을 다시 추가하고 상세 정보를 표시합니다.
    normal_display = []
    for s, df in temp_data:
        # RSI가 40 이하인 관심 종목들 추출
        rsi = float(calculate_rsi(df['Close']).iloc[-1])
        if 32 <= rsi <= 40:
            curr_p = float(df['Close'].iloc[-1])
            # 뉴스 분석 다시 가져오기 (이미 temp_data에 포함되어 있어야 함)
            normal_display.append(f"📈 *{s}*\n📍 Buy: ${curr_p:.2f} | RSI: {rsi:.1f}")
    
    if normal_display:
        report.append(f"\n🔍 **[WATCHLIST]**\n" + "\n".join(normal_display[:10]))

    report.append("━━━━━━━━━━━━━━")
    report.append(f"✅ {total_analyzed}분석 (시장점수: {int((1-ratio)*100)}점)")

    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": "\n".join(report), "parse_mode": "Markdown"})


if __name__ == "__main__":
    run_analysis()

