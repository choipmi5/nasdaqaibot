import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# 1. 환경 설정
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

# [확장된 100개 종목 리스트] 나스닥 100 핵심 + 인기 ETF
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

# --- [지표 계산 함수] ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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

def run_analysis():
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    
    review_reports, super_buys, strong_buys, normal_buys = [], [], [], []
    down_count, total_analyzed = 0, 0

    # 데이터 수집 및 분석
    for s in STOCKS:
        try:
            df = yf.download(s, period="50d", progress=False)
            if len(df) < 30: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            close = df['Close']
            curr_p = float(close.iloc[-1])
            prev_p = float(close.iloc[-2])
            ma20 = close.rolling(20).mean()
            
            total_analyzed += 1
            if curr_p < float(ma20.iloc[-1]): down_count += 1
            
            # --- [자가 복기: 어제 목표가 도달 확인] ---
            # 하락장 비율 60% 이상이면 1.5%, 아니면 2.5% 타겟
            ratio_temp = down_count / total_analyzed
            y_target = 1.015 if ratio_temp > 0.6 else 1.025
            
            if calculate_rsi(close).iloc[-2] < 35:
                is_hit = "🎯익절" if float(df['High'].iloc[-1]) >= prev_p * y_target else "⏳보유"
                review_reports.append(f"{s}:{is_hit}")

            # --- [지표 분석] ---
            rsi = float(calculate_rsi(close).iloc[-1])
            mfi = float(calculate_mfi(df).iloc[-1])
            std = close.rolling(20).std()
            lower_b = float((ma20 - (std * 2)).iloc[-1])
            macd, signal = calculate_macd(close)
            
            is_oversold = rsi < 32 or curr_p <= lower_b
            is_money_in = mfi < 35
            is_turning = float(macd.iloc[-1]) > float(signal.iloc[-1])

            target_p = f"${curr_p * (1.015 if ratio_temp > 0.6 else 1.025):.2f}"
            
            if is_oversold and is_money_in and is_turning:
                super_buys.append(f"🎯 *{s}* (목표: {target_p})")
            elif is_oversold and is_money_in:
                strong_buys.append(f"💎 *{s}* (목표: {target_p})")
            elif is_oversold:
                normal_buys.append(f"📈 *{s}* (목표: {target_p})")
                
        except: continue

    # 리포트 구성
    ratio = down_count / total_analyzed if total_analyzed > 0 else 0
    mode = "⚠️ 하락방어" if ratio > 0.6 else "🚀 정상추세"
    
    report = [
        f"🤖 *AI SELF-DIAGNOSIS TOTAL (100+)*",
        f"📅 {now.strftime('%m-%d %H:%M')} (KST) | 📡 **{mode}**",
        f"━━━━━━━━━━━━━━",
        f"📊 **[어제 예측 복기]**",
        ", ".join(review_reports[:10]) if review_reports else "- 분석 대상 없음",
        f"━━━━━━━━━━━━━━",
        f"🎯 **[SUPER BUY]**",
        "\n".join(super_buys[:5]) if super_buys else "- 해당 없음",
        f"\n💎 **[STRONG BUY]**",
        "\n".join(strong_buys[:10]) if strong_buys else "- 해당 없음",
        f"\n🔍 **[NORMAL BUY]**",
        "\n".join(normal_buys[:15]) if normal_buys else "- 해당 없음",
        f"━━━━━━━━━━━━━━",
        f"✅ `{total_analyzed}`종목 전수 분석 완료"
    ]
    
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": "\n".join(report), "parse_mode": "Markdown"})

if __name__ == "__main__":
    run_analysis()


