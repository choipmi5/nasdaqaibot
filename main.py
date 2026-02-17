import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# 1. 환경 설정
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

# 분석 종목 리스트 (핵심 60~100개 요약본)
STOCKS = [
    "QQQ", "TQQQ", "SQQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "AMD", "SOXL", "SOXS", "AVGO", "NFLX", "TSM", "ADBE", "INTC", "QCOM", "MU",
    "PANW", "SNPS", "CDNS", "MAR", "LRCX", "ADSK", "MELI", "PYPL", "ABNB", "COST",
    "CONL", "NVDL", "TSLL", "SOXX", "SCHD", "VOO", "IVV", "VTI", "UPRO", "TMF"
]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_analysis():
    if not TELEGRAM_TOKEN or not CHAT_ID: return

    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    
    review_reports = []
    buy_signals = []
    down_count = 0
    total_analyzed = 0

    # 데이터 수집 및 분석 시작
    for s in STOCKS:
        try:
            df = yf.download(s, period="40d", progress=False)
            if len(df) < 20: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            close = df['Close']
            high = df['High']
            curr_p = float(close.iloc[-1])
            prev_p = float(close.iloc[-2])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            rsi = float(calculate_rsi(close).iloc[-1])
            
            total_analyzed += 1
            if curr_p < ma20: down_count += 1
            
            # --- [자가 분석: 어제 추천했다면 오늘 익절했는가?] ---
            # (어제 RSI가 35 미만이었다고 가정할 때, 오늘의 고가가 어제 종가 대비 목표 수익률을 찍었는지 확인)
            prev_rsi = calculate_rsi(close).iloc[-2]
            if prev_rsi < 35:
                target_price = prev_p * 1.015 # 어제 설정했을 목표가 (1.5%)
                is_hit = "🎯 익절완료" if float(high.iloc[-1]) >= target_price else "⏳ 보유중"
                review_reports.append(f"{s}: {is_hit} (고가: {((high.iloc[-1]/prev_p)-1)*100:+.1f}%)")

            # --- [오늘의 신규 추천 로직] ---
            if rsi < 32:
                buy_signals.append(f"📈 *{s}* (RSI: `{rsi:.1f}`, 현재: `${curr_p:.2f}`)")
        except: continue

    # 시장 모드 판별
    ratio = down_count / total_analyzed if total_analyzed > 0 else 0
    mode = "⚠️ 하락장 방어" if ratio > 0.6 else "🚀 정상 추세"
    profit_target = "1.5%" if ratio > 0.6 else "2.0~2.5%"

    # 리포트 구성
    report = [
        f"🤖 *AI SELF-DIAGNOSIS FINAL*",
        f"📅 {now.strftime('%m-%d %H:%M')} (KST)",
        f"━━━━━━━━━━━━━━",
        f"📡 **시장 모드:** {mode} ({ratio*100:.0f}%)",
        f"🎯 **오늘의 익절 목표:** `{profit_target}`",
        f"━━━━━━━━━━━━━━",
        f"📊 **[어제 추천주 복기]**",
        "\n".join(review_reports[:7]) if review_reports else "- 복기 대상 종목 없음",
        f"━━━━━━━━━━━━━━",
        f"🔥 **[실시간 매수 추천]**",
        "\n".join(buy_signals[:15]) if buy_signals else "- 현재 매수 적정 종목 없음",
        f"━━━━━━━━━━━━━━",
        f"✅ 분석 완료: `{total_analyzed}` 종목"
    ]
    
    msg = "\n".join(report)
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    run_analysis()
