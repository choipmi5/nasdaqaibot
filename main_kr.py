import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# 1. 환경 설정
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

# [국장 핵심 80~100개 종목 리스트]
STOCKS_KR = [
    "005930.KS", "000660.KS", "373220.KS", "005380.KS", "005490.KS", "000270.KS", "035420.KS", "006400.KS", "051910.KS", "068270.KS",
    "035720.KS", "105560.KS", "012330.KS", "028260.KS", "055550.KS", "003550.KS", "032830.KS", "096770.KS", "033780.KS", "000810.KS",
    "086790.KS", "009150.KS", "010130.KS", "018260.KS", "034220.KS", "011200.KS", "015760.KS", "001500.KS", "036570.KS", "009830.KS",
    "247540.KQ", "091990.KQ", "066970.KQ", "293480.KQ", "025900.KQ", "253450.KQ", "035900.KQ", "067160.KQ", "036830.KQ", "039030.KQ",
    "041510.KQ", "051900.KS", "010950.KS", "034730.KS", "000720.KS", "047050.KS", "011070.KS", "005935.KS", "030240.KS", "271560.KS"
    # 필요시 티커 추가 가능
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

    for s in STOCKS_KR:
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
            
            # --- [국장용 자가 복기] ---
            ratio_temp = down_count / total_analyzed
            y_target = 1.012 if ratio_temp > 0.6 else 1.020 # 국장은 목표치를 살짝 낮춤
            
            if calculate_rsi(close).iloc[-2] < 35:
                is_hit = "🎯익절" if float(df['High'].iloc[-1]) >= prev_p * y_target else "⏳보유"
                review_reports.append(f"{s.split('.')[0]}:{is_hit}")

            # --- [지표 분석] ---
            rsi = float(calculate_rsi(close).iloc[-1])
            mfi = float(calculate_mfi(df).iloc[-1])
            std = close.rolling(20).std()
            lower_b = float((ma20 - (std * 2)).iloc[-1])
            macd, signal = calculate_macd(close)
            
            is_oversold = rsi < 32 or curr_p <= lower_b
            is_money_in = mfi < 35
            is_turning = float(macd.iloc[-1]) > float(signal.iloc[-1])

            # 국장은 변동성을 고려해 보수적 가격 제시 (단위: 원)
            target_p = int(curr_p * (1.012 if ratio_temp > 0.6 else 1.020))
            
            ticker_name = s.split('.')[0]
            if is_oversold and is_money_in and is_turning:
                super_buys.append(f"🎯 *{ticker_name}* (목표: {target_p:,}원)")
            elif is_oversold and is_money_in:
                strong_buys.append(f"💎 *{ticker_name}* (목표: {target_p:,}원)")
            elif is_oversold:
                normal_buys.append(f"📈 *{ticker_name}* (목표: {target_p:,}원)")
                
        except: continue

    ratio = down_count / total_analyzed if total_analyzed > 0 else 0
    mode = "⚠️ 하락방어" if ratio > 0.6 else "🚀 정상추세"
    
    report = [
        f"🇰🇷 *KOREA STOCK AI REPORT*",
        f"📅 {now.strftime('%m-%d %H:%M')} (KST) | 📡 **{mode}**",
        f"━━━━━━━━━━━━━━",
        f"📊 **[전일 국장 복기]**",
        ", ".join(review_reports[:8]) if review_reports else "- 분석 대상 없음",
        f"━━━━━━━━━━━━━━",
        f"🎯 **[SUPER BUY]**",
        "\n".join(super_buys[:5]) if super_buys else "- 해당 없음",
        f"\n💎 **[STRONG BUY]**",
        "\n".join(strong_buys[:10]) if strong_buys else "- 해당 없음",
        f"\n🔍 **[NORMAL BUY]**",
        "\n".join(normal_buys[:15]) if normal_buys else "- 해당 없음",
        f"━━━━━━━━━━━━━━",
        f"✅ 국장 `{total_analyzed}`종목 정밀 분석 완료"
    ]
    
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": "\n".join(report), "parse_mode": "Markdown"})

if __name__ == "__main__":
    run_analysis()
