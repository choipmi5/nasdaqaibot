import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# 1. 설정 및 100개 종목 리스트 (나스닥 주요주 및 인기주)
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

STOCKS = [
    "QQQ", "TQQQ", "SQQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "AMD", "SOXL", "SOXS", "AVGO", "NFLX", "ASML", "TSM", "ADBE", "COST", "PEP",
    "AZN", "LIN", "CSCO", "TMUS", "AVGO", "INTC", "TXN", "QCOM", "AMAT", "ADP",
    "ISRG", "SBUX", "MDLZ", "GILD", "INTU", "VRTX", "AMGN", "REGN", "PYPL", "FISV",
    "ATVI", "BKNG", "CSX", "MU", "PANW", "SNPS", "CDNS", "ORLY", "MNST", "MAR",
    "KDP", "CHTR", "KLAC", "AEP", "LRCX", "ADSK", "MNST", "DXCM", "MELI", "IDXX",
    "PAYX", "CTAS", "ORLY", "LULU", "MCHP", "MRVL", "CPRT", "ODFL", "AZN", "TEAM",
    "ALGN", "WDAY", "FAST", "PCAR", "ROST", "DLTR", "EBAY", "SIRI", "ZM", "JD",
    "LCID", "DDOG", "RIVN", "ENPH", "CEG", "ZS", "ABNB", "PDD", "OKTA", "SPLK",
    "CONL", "NVDL", "TSLL", "SOXX", "SCHD", "JEPI", "VOO", "IVV", "VTI", "UPRO"
]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run():
    if not TELEGRAM_TOKEN or not CHAT_ID: return

    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    buy_signals = []
    down_count = 0
    total_analyzed = 0
    
    # 100개 종목 순회 분석
    for s in STOCKS:
        try:
            # 데이터 수집 (최소한의 데이터만 가져와 속도 향상)
            df = yf.download(s, period="40d", progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            close = df['Close']
            curr_p = float(close.iloc[-1])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            rsi = float(calculate_rsi(close).iloc[-1])
            
            total_analyzed += 1
            if curr_p < ma20: down_count += 1
            
            # RSI 30 미만 - 강력 매수 구간인 종목만 선별
            if rsi < 30:
                buy_signals.append(f"🔥 *{s}* (RSI: `{rsi:.1f}` / 현재가: `${curr_p:.2f}`)")
            # RSI 30~35 - 일반 매수 구간
            elif rsi < 35:
                buy_signals.append(f"📈 *{s}* (RSI: `{rsi:.1f}`)")
        except: continue

    ratio = down_count / total_analyzed if total_analyzed > 0 else 0
    mode = "⚠️ 하락장 방어" if ratio > 0.6 else "🚀 정상 추세"
    
    report = [
        f"🤖 *AI MASSIVE REPORT (100+)*",
        f"📅 {now.strftime('%m-%d %H:%M')} (KST)",
        f"━━━━━━━━━━━━━━",
        f"📡 모드: {mode}",
        f"📊 하락추세 비율: `{ratio*100:.1f}%`",
        f"📉 분석 완료: `{total_analyzed}` 종목",
        f"━━━━━━━━━━━━━━\n",
        f"🔍 *[RSI 과매도 포착]*"
    ]
    
    if buy_signals:
        # 메시지가 너무 길어지면 텔레그램에서 잘리므로 상위 15개 정도만 노출하거나 요약
        report.extend(buy_signals[:20]) 
        if len(buy_signals) > 20:
            report.append(f"\n...외 {len(buy_signals)-20}개 종목 더 있음")
    else:
        report.append("- 현재 과매도(RSI 35미만) 종목 없음")
    
    msg = "\n".join(report)
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    run()

