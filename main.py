import os
import yfinance as yf
import pandas as pd
import requests
import json
from datetime import datetime
import pytz

# 1. 환경 설정
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

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

    if not os.path.exists(log_file): return original_stocks, []
    try:
        df = pd.read_csv(log_file)
        perf = df.groupby('종목')['목표가달성'].apply(lambda x: (x == 'YES').mean())
        count = df.groupby('종목').size()
        
        # [수정] 최소 표본 10개 기준
        evaluated_stocks = count[count >= 10].index.tolist()
        
        # [수정] 승률 30% 미만은 무조건 퇴출
        bad_stocks = [s for s in evaluated_stocks if perf[s] < 0.3]
        
        # [수정] 승률 30%~50% 사이인 종목들 처리
        grey_zone = [s for s in evaluated_stocks if 0.3 <= perf[s] < 0.5]
        
        if not market_recovery:
            bad_stocks.extend(grey_zone) # 시장 안 좋으면 회색지대도 퇴출
        # else: 시장 좋으면 회색지대(30% 이상)는 복귀(bad_stocks에 안 넣음)

        with open(blacklist_file, 'w') as f:
            json.dump(list(set(bad_stocks)), f)
        return [s for s in original_stocks if s not in bad_stocks], list(set(bad_stocks))
    except: return original_stocks, []

def run_analysis():
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)

    optimized_stocks, blacklisted = get_optimized_stocks('trade_log_nasdaq.csv', 'blacklist_nasdaq.json', STOCKS)
    
    review_reports, super_buys, strong_buys, normal_buys = [], [], [], []
    trade_logs, total_analyzed, down_count = [], 0, 0

    for s in optimized_stocks:
        try:
            df = yf.download(s, period="50d", progress=False)
            if len(df) < 30: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            close = df['Close']
            curr_p, prev_p = float(close.iloc[-1]), float(close.iloc[-2])
            ma20 = close.rolling(20).mean()
            
            total_analyzed += 1
            if curr_p < float(ma20.iloc[-1]): down_count += 1
            
            y_target = 1.015 if (down_count/total_analyzed) > 0.6 else 1.025
            
            if calculate_rsi(close).iloc[-2] < 35:
                is_hit_bool = float(df['High'].iloc[-1]) >= prev_p * y_target
                review_reports.append(f"{s}:{'🎯' if is_hit_bool else '⏳'}")
                trade_logs.append({"날짜": now.strftime('%Y-%m-%d'), "종목": s, "목표가달성": "YES" if is_hit_bool else "NO"})

            rsi, mfi = float(calculate_rsi(close).iloc[-1]), float(calculate_mfi(df).iloc[-1])
            std = close.rolling(20).std()
            lower_b = float((ma20 - (std * 2)).iloc[-1])
            macd, signal = calculate_macd(close)
            
            is_oversold = rsi < 32 or curr_p <= lower_b
            is_money_in = mfi < 35
            is_turning = float(macd.iloc[-1]) > float(signal.iloc[-1])
            target_p = f"${curr_p * y_target:.2f}"
            
            if is_oversold and is_money_in and is_turning: super_buys.append(f"🎯 *{s}* ({target_p})")
            elif is_oversold and is_money_in: strong_buys.append(f"💎 *{s}* ({target_p})")
            elif is_oversold: normal_buys.append(f"📈 *{s}* ({target_p})")
        except: continue

    if trade_logs:
        pd.DataFrame(trade_logs).to_csv('trade_log_nasdaq.csv', mode='a', index=False, header=not os.path.exists('trade_log_nasdaq.csv'), encoding='utf-8-sig')

    mode = "⚠️ 하락방어" if (down_count/total_analyzed if total_analyzed > 0 else 0) > 0.6 else "🚀 정상추세"
    report = [
        f"🇺🇸 *NASDAQ EVOLVING AI*", f"📅 {now.strftime('%m-%d %H:%M')} | {mode} (🤖제외:{len(blacklisted)})", "━━━━━━━━━━━━━━",
        f"📊 **[전일 복기]**\n" + (", ".join(review_reports[:10]) if review_reports else "- 분석 대상 없음"), "━━━━━━━━━━━━━━",
        f"🎯 **[SUPER BUY]**\n" + ("\n".join(super_buys[:5]) if super_buys else "- 해당 없음"),
        f"\n💎 **[STRONG BUY]**\n" + ("\n".join(strong_buys[:10]) if strong_buys else "- 해당 없음"),
        f"\n🔍 **[NORMAL BUY]**\n" + ("\n".join(normal_buys[:15]) if normal_buys else "- 해당 없음"), "━━━━━━━━━━━━━━",
        f"✅ {total_analyzed}종목 분석 완료"
    ]
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": "\n".join(report), "parse_mode": "Markdown"})

if __name__ == "__main__":
    run_analysis()





