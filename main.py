import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# 1. 설정 및 Secrets 로드
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')
STOCKS = ["QQQ", "TQQQ", "SQQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "SOXL"]

# 2. RSI 직접 계산 함수 (에러 방지용)
def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run():
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    
    buy_signals = []
    down_count = 0
    
    for s in STOCKS:
        try:
            df = yf.download(s, period="40d", progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            close = df['Close']
            curr_p = float(close.iloc[-1])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            rsi = float(get_rsi(close).iloc[-1])
            
            if curr_p < ma20: down_count += 1
            if rsi < 32:
                buy_signals.append(f"📈 *{s}* (RSI: `{rsi:.1f}`)\n  └ 목표: `${curr_p * 1.01:.2f}`")
        except: continue

    # 리포트 작성
    ratio = down_count / len(STOCKS)
    mode = "⚠️ 하락방어" if ratio > 0.6 else "🚀 정상추세"
    
    text = [
        f"🤖 *AI REPORT*",
        f"📅 {now.strftime('%m-%d %H:%M')} (KST)",
        f"━━━━━━━━━━━━━━",
        f"📡 모드: {mode} ({ratio*100:.0f}%)",
        f"━━━━━━━━━━━━━━\n",
        f"🔥 *[추천]*"
    ]
    text.extend(buy_signals if buy_signals else ["- 조건 만족 없음"])
    
    # 전송
    msg = "\n".join(text)
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    run()
