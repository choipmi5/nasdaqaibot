import os
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
CHAT_ID = os.environ['CHAT_ID']

# 1. 라이브러리 설치 및 시간대 설정
!pip install yfinance pandas_ta requests pytz --quiet

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from datetime import datetime
import pytz
import warnings

warnings.filterwarnings('ignore')

# --- [정보 입력] ---
TELEGRAM_TOKEN = "8038442610:AAFIQ9iPM_794olGtsfpG2l9iGAcxQD6eYQ"
CHAT_ID = "6165233712"
STOCKS = ["QQQ", "TQQQ", "NVDA", "TSLA", "AAPL", "MSFT", "SOXL", "AMD", "META", "AMZN", "NFLX", "GOOGL"] # 예시로 12개, 50개로 확장 가능
# ------------------

def to_float(val):
    if isinstance(val, (pd.Series, pd.DataFrame)): return float(val.iloc[0])
    return float(val)

def run_analysis():
    print(f"🔄 AI 자기 최적화 분석 가동...")
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)

    # 1. 시장 심리 및 추세 파악
    vix_df = yf.download("^VIX", period="5d", progress=False)
    vix_val = to_float(vix_df['Close'].iloc[-1])
    
    # 2. 전 종목 역배열 비율 계산 (Self-Optimization 핵심)
    down_trend_count = 0
    total_analyzed = 0
    
    temp_data = {}
    for ticker in STOCKS:
        try:
            df = yf.download(ticker, period="40d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df['MA20'] = ta.sma(df['Close'], length=20)
            
            curr_price = to_float(df['Close'].iloc[-1])
            ma20_val = to_float(df['MA20'].iloc[-1])
            
            if curr_price < ma20_val: down_trend_count += 1
            total_analyzed += 1
            temp_data[ticker] = df # 데이터 재사용을 위해 저장
        except: continue

    # 3. AI 전략 자동 수정 로직
    down_trend_ratio = (down_trend_count / total_analyzed) if total_analyzed > 0 else 0
    
    # 기본 익절률 설정 (VIX 기준)
    base_profit = 1.015 if vix_val < 25 else 1.025
    
    # [전략 수정] 역배열 종목이 60% 이상이면 '하락장 모드' 가동
    if down_trend_ratio > 0.6:
        applied_profit = base_profit - 0.005 # 익절 타겟 0.5% 하향 (보수적)
        mode_text = "⚠️ 하락장 방어 모드 (보수적 타겟)"
        filter_out_down_trend = True # 역배열 종목 추천 제외
    else:
        applied_profit = base_profit
        mode_text = "🚀 정상 추세 모드 (공격적 타겟)"
        filter_out_down_trend = False

    report = [
        f"━━━━━━━━━━━━━━",
        f"🤖 *SELF-OPTIMIZING REPORT*",
        f"📅 {now.strftime('%Y-%m-%d %H:%M')} (KST)",
        f"━━━━━━━━━━━━━━",
        f"📡 *작동 모드:* {mode_text}",
        f"📊 *하락추세 비율:* `{down_trend_ratio*100:.1f}%`",
        f"🎯 *조정된 타겟:* `+{(applied_profit-1)*100:.1f}%`",
        f"━━━━━━━━━━━━━━\n"
    ]
    
    buy_signals = []
    
    for ticker, df in temp_data.items():
        try:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            curr_price = to_float(df['Close'].iloc[-1])
            curr_rsi = to_float(df['RSI'].iloc[-1])
            ma20_val = to_float(df['MA20'].iloc[-1])
            
            # 매수 필터 적용
            if curr_rsi < 32:
                # 하락장 모드일 때 역배열 종목은 추천에서 아예 뺌
                if filter_out_down_trend and curr_price < ma20_val:
                    continue 
                
                buy_signals.append(f"📈 *{ticker}* (RSI: `{curr_rsi:.1f}`)\n  └ 목표가: `${curr_price * applied_profit:.2f}`")
        except: continue

    report.append("🔥 *[최적화된 매수 추천]*")
    report.extend(buy_signals if buy_signals else ["- 현재 조건 만족 종목 없음"])
    report.append("\n━━━━━━━━━━━━━━")

    # 텔레그램 전송
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": "\n".join(report), "parse_mode": "Markdown"})
    print(f"✅ {mode_text}로 분석 완료!")

run_analysis()
