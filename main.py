import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz
import warnings

warnings.filterwarnings('ignore')

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

STOCKS = ["QQQ", "TQQQ", "SQQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "SOXL"]

# RSI를 직접 계산하는 함수 (외부 라이브러리 미사용)
def calculate_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def run_analysis():
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    
    # 하락장 여부 판단
    down_trend_count = 0
    temp_results = []
    
    for ticker in STOCKS:
        try:
            df = yf.download(ticker, period="40d", progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            close = df['Close']
            ma20 = close.rolling(window=20).mean()
            rsi = calculate_rsi(close)
            
            curr_price = float(close.iloc[-1])
            curr_ma20 = float(ma20.iloc[-1])
            curr_rsi = float(rsi.iloc[-1])
            
            if curr_price < curr_ma20: down_trend_count += 1
            if curr_rsi < 32:
                temp_results.append((ticker, curr_rsi, curr_price))
        except: continue

    down_trend_ratio = (down_trend_count / len(STOCKS))
    applied_profit = 1.01 if down_trend_ratio > 0.6 else 1.02
    mode_text = "⚠️ 하락장 방어" if down_trend_ratio > 0.6 else "🚀 정상 추세"

    report = [
        f"🤖 *SELF-OPTIMIZING REPORT*",
        f"📅 {now.strftime('%Y-%m-%d %H:%M')} (KST)",
        f"━━━━━━━━━━━━━━",
        f"📡 모드: {mode_text}",
        f"📊 하락추세 비율: `{down_trend_ratio*100:.1f}%`",
        f"🎯 목표 수익률: `+{(applied_profit-1)*100:.1f}%`",
        f"━━━━━━━━━━━━━━\n",
        f"🔥 *[매수 추천]*"
    ]
    
    if temp_results:
        for t, r, p in temp_results:
            report.append(f"📈 *{t}* (RSI: `{r:.1f}`)\n  └ 목표가: `${p * applied_profit:.2f}`")
    else:
        report.append("- 현재 조건 만족 종목 없음")
    
    message = "\n".join(report)
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"})

if __name__ == "__main__":
    run_analysis()

