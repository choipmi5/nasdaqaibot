import os
import yfinance as yf
import pandas as pd
import requests
import json
from datetime import datetime
import pytz

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

KR_STOCKS = [
    ("삼성전자", "005930.KS"), ("SK하이닉스", "000660.KS"), ("LG엔솔", "373220.KS"), ("삼성바이오", "207940.KS"), ("현대차", "005380.KS"),
    ("기아", "000270.KS"), ("셀트리온", "068270.KS"), ("KB금융", "105560.KS"), ("POSCO홀딩스", "005490.KS"), ("NAVER", "035420.KS"),
    ("신한지주", "055550.KS"), ("삼성물산", "028260.KS"), ("현대모비스", "012330.KS"), ("LG화학", "051910.KS"), ("하나금융지주", "086790.KS"),
    ("삼성생명", "032830.KS"), ("카카오", "035720.KS"), ("메리츠금융", "138040.KS"), ("삼성SDI", "006400.KS"), ("LG전자", "066570.KS"),
    ("카카오뱅크", "323410.KS"), ("삼성화재", "000810.KS"), ("KT&G", "033780.KS"), ("한국전력", "015760.KS"), ("HMM", "011200.KS"),
    ("SK이노베이션", "096770.KS"), ("삼성전기", "009150.KS"), ("크래프톤", "259960.KS"), ("두산에너빌리티", "034020.KS"), ("HD현대중공업", "329180.KS"),
    ("에코프로비엠", "247540.KQ"), ("에코프로", "086520.KQ"), ("HLB", "028300.KQ"), ("알테오젠", "191150.KQ"), ("엔켐", "348370.KQ"),
    ("리노공업", "058470.KQ"), ("레인보우로보틱스", "272410.KQ"), ("HPSP", "403870.KQ"), ("신성델타테크", "065350.KQ"), ("제주반도체", "080220.KQ"),
    ("포스코퓨처엠", "003670.KS"), ("SK", "034730.KS"), ("S-Oil", "010950.KS"), ("고려아연", "010130.KS"), ("삼성에스디에스", "018260.KS"),
    ("한화에어로스페이스", "012450.KS"), ("대한항공", "003490.KS"), ("KT", "030200.KS"), ("기업은행", "024110.KS"), ("HD현대", "267250.KS"),
    ("LG", "003550.KS"), ("한국금융지주", "071050.KS"), ("아모레퍼시픽", "090430.KS"), ("코웨이", "021240.KS"), ("금양", "001570.KS"),
    ("한온시스템", "018880.KS"), ("현대글로비스", "086280.KS"), ("삼성중공업", "010140.KS"), ("넷마블", "251270.KS"), ("카카오페이", "377300.KS"),
    ("엔씨소프트", "036570.KS"), ("유한양행", "000100.KS"), ("한미사이언스", "008930.KS"), ("한미약품", "128940.KS"), ("오리온", "271560.KS"),
    ("미래에셋증권", "006800.KS"), ("하이브", "352820.KS"), ("팬오션", "028670.KS"), ("두산밥캣", "241560.KS"), ("롯데케미칼", "011170.KS"),
    ("현대건설", "000720.KS"), ("LG생활건강", "051900.KS"), ("SK바이오사이언스", "302440.KS"), ("호텔신라", "008770.KS"), ("GS", "078930.KS"),
    ("포스코인터내셔널", "047050.KS"), ("에스디바이오센서", "137310.KS"), ("씨젠", "096530.KQ"), ("펄어비스", "263750.KQ"), ("셀트리온제약", "068760.KQ"),
    ("휴젤", "145020.KQ"), ("클래시스", "214150.KQ"), ("에스엠", "041510.KQ"), ("JYP Ent.", "035900.KQ"), ("루닛", "328130.KQ"),
    ("가온칩스", "399720.KQ"), ("오픈엣지테크놀로지", "394280.KQ"), ("소울브레인", "357780.KQ"), ("동진쎄미켐", "005290.KQ"), ("원익IPS", "030530.KQ"),
    ("이오테크닉스", "039030.KQ"), ("솔브레인홀딩스", "036830.KQ"), ("파두", "440110.KQ"), ("위메이드", "112040.KQ"), ("컴투스", "078340.KQ"),
    ("바이오니아", "064550.KQ"), ("STX", "011810.KS"), ("한화오션", "042660.KS"), ("LS", "006260.KS"), ("LS ELECTRIC", "010120.KS")
]

NAME_MAP = {ticker: name for name, ticker in KR_STOCKS}
STOCKS_KR = [ticker for name, ticker in KR_STOCKS]

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

def get_optimized_stocks(log_file, blacklist_file, original_tickers):
    market_recovery = False
    try:
        market_df = yf.download("^KS11", period="50d", progress=False)
        if isinstance(market_df.columns, pd.MultiIndex): market_df.columns = market_df.columns.get_level_values(0)
        market_recovery = market_df['Close'].iloc[-1] > market_df['Close'].rolling(20).mean().iloc[-1]
    except: pass

    if not os.path.exists(log_file): return original_tickers, []
    try:
        df = pd.read_csv(log_file)
        perf = df.groupby('종목')['목표가달성'].apply(lambda x: (x == 'YES').mean())
        count = df.groupby('종목').size()
        
        # [수정] 표본 10개 이상만 평가
        eval_names = count[count >= 10].index.tolist()
        
        # [수정] 승률 30% 미만 무조건 제외
        bad_names = [n for n in eval_names if perf[n] < 0.3]
        
        # [수정] 30~50% 구간
        grey_zone = [n for n in eval_names if 0.3 <= perf[n] < 0.5]
        
        if not market_recovery:
            bad_names.extend(grey_zone)

        with open(blacklist_file, 'w') as f:
            json.dump(list(set(bad_names)), f)
        bad_tickers = [t for name, t in KR_STOCKS if name in bad_names]
        return [t for t in original_tickers if t not in bad_tickers], list(set(bad_names))
    except: return original_tickers, []

def run_analysis():
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    optimized_stocks, blacklisted = get_optimized_stocks('trade_log_kr.csv', 'blacklist_kr.json', STOCKS_KR)
    review_reports, super_buys, strong_buys, normal_buys, trade_logs, total_analyzed, down_count = [], [], [], [], [], 0, 0

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
            y_target = 1.012 if (down_count/total_analyzed) > 0.6 else 1.020
            stock_name = NAME_MAP.get(s, s)
            
            if calculate_rsi(close).iloc[-2] < 35:
                is_hit_bool = float(df['High'].iloc[-1]) >= prev_p * y_target
                review_reports.append(f"{stock_name}:{'🎯' if is_hit_bool else '⏳'}")
                trade_logs.append({"날짜": now.strftime('%Y-%m-%d'), "종목": stock_name, "목표가달성": "YES" if is_hit_bool else "NO"})

            rsi, mfi = float(calculate_rsi(close).iloc[-1]), float(calculate_mfi(df).iloc[-1])
            std = close.rolling(20).std()
            lower_b = float((ma20 - (std * 2)).iloc[-1])
            macd, signal = calculate_macd(close)
            is_oversold = rsi < 32 or curr_p <= lower_b
            is_money_in = mfi < 35
            is_turning = float(macd.iloc[-1]) > float(signal.iloc[-1])
            target_p = int(curr_p * y_target)

            if is_oversold and is_money_in and is_turning: super_buys.append(f"🎯 *{stock_name}* ({target_p:,}원)")
            elif is_oversold and is_money_in: strong_buys.append(f"💎 *{stock_name}* ({target_p:,}원)")
            elif is_oversold: normal_buys.append(f"📈 *{stock_name}* ({target_p:,}원)")
        except: continue

    if trade_logs:
        pd.DataFrame(trade_logs).to_csv('trade_log_kr.csv', mode='a', index=False, header=not os.path.exists('trade_log_kr.csv'), encoding='utf-8-sig')

    mode = "⚠️ 하락방어" if (down_count/total_analyzed if total_analyzed > 0 else 0) > 0.6 else "🚀 정상추세"
    report = [
        f"🇰🇷 *KOREA EVOLVING AI*", f"📅 {now.strftime('%m-%d %H:%M')} | {mode} (🤖제외:{len(blacklisted)})", "━━━━━━━━━━━━━━",
        f"📊 **[전일 복기]**\n" + (", ".join(review_reports[:10]) if review_reports else "- 분석 대상 없음"), "━━━━━━━━━━━━━━",
        f"🎯 **[SUPER BUY]**\n" + ("\n".join(super_buys[:5]) if super_buys else "- 해당 없음"),
        f"\n💎 **[STRONG BUY]**\n" + ("\n".join(strong_buys[:10]) if strong_buys else "- 해당 없음"),
        f"\n🔍 **[NORMAL BUY]**\n" + ("\n".join(normal_buys[:15]) if normal_buys else "- 해당 없음"), "━━━━━━━━━━━━━━",
        f"✅ {total_analyzed}종목 분석 완료"
    ]
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": "\n".join(report), "parse_mode": "Markdown"})

if __name__ == "__main__":
    run_analysis()

