import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import pytz
from google import genai # [수정됨] 지원 종료된 generativeai 대신 최신 genai 사용
import re
import warnings
import vectorbt as vbt # 전략 승률 백테스팅용

# pandas 연산 경고 무시 (출력창 깔끔하게 유지)
warnings.filterwarnings('ignore')

# ==========================================
# [1. 시스템 환경 및 리스크 매니지먼트 설정]
# ==========================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# 포지션 사이징을 위한 총 운용 자본 및 리스크 허용치
TOTAL_CAPITAL = 100000.0  
RISK_TOLERANCE_PER_TRADE = 0.01  # 1회 매수 시 총자본의 최대 1% 리스크만 노출 (켈리/리스크 패리티)

# [수정됨] 새로운 Client 기반 API 초기화
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

# ==========================================
# [2. 섹터 및 분석 대상 종목 유니버스]
# ==========================================
SECTORS = {
    "SEMICON": ["NVDA", "AMD", "AVGO", "TSM", "MU", "ARM", "LRCX", "AMAT", "TXN", "QCOM", "INTC", "KLAC", "SNPS", "CDNS", "MRVL"],
    "BIGTECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX"],
    "AI/SW/FIN": ["PLTR", "SNOW", "ADBE", "ORCL", "CRM", "PANW", "COIN", "MSTR", "INTU", "CRWD", "DDOG", "NOW"]
}

# ETF 배제, 미국 주요 우량/성장주 약 250개 (생략 없음)
STOCKS = list(set([
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "AVGO", "NFLX", "TSM", "ADBE", "COST", "PEP", "AZN", "LIN", "CSCO", 
    "TMUS", "INTC", "TXN", "QCOM", "AMAT", "ADP", "ISRG", "SBUX", "MDLZ", "GILD", "INTU", "VRTX", "AMGN", "REGN", "PYPL", "FISV", "BKNG", 
    "CSX", "MU", "PANW", "SNPS", "CDNS", "ORLY", "MNST", "MAR", "KDP", "CHTR", "KLAC", "AEP", "LRCX", "ADSK", "DXCM", "MELI", "IDXX", 
    "PAYX", "CTAS", "LULU", "MCHP", "MRVL", "CPRT", "ODFL", "TEAM", "ALGN", "WDAY", "FAST", "PCAR", "ROST", "DLTR", "EBAY", "SIRI", 
    "ZM", "JD", "LCID", "DDOG", "RIVN", "ENPH", "CEG", "ZS", "ABNB", "PDD", "OKTA", "ARM", "PLTR", "SNOW", "U", "COIN", "MSTR", 
    "BRK-B", "UNH", "JNJ", "JPM", "V", "PG", "HD", "CVX", "MA", "ABBV", "MRK", "KO", "PFE", "TMO", "MCD", "DIS", "ABT", "WMT", "CRM", 
    "DHR", "NEE", "PM", "BMY", "UNP", "NKE", "RTX", "LOW", "HON", "SPGI", "ORCL", "BA", "IBM", "GS", "ELV", "CAT", "GE", "MDT", "AXP", 
    "DE", "LMT", "BLK", "ADI", "TJX", "SYK", "C", "NOW", "CVS", "ZTS", "CIT", "MMC", "CB", "SO", "DUK", "PGR", "BDX", "BSX", "T", "CI", 
    "EQIX", "SLB", "EOG", "AON", "NOC", "SHW", "WM", "FCX", "ICE", "MCO", "EMR", "EW", "MCK", "CMCSA", "GPN", "PXD", "MPC", "NXPI", 
    "FDX", "VLO", "PH", "KMB", "PSX", "SRE", "ROP", "TEL", "TRV", "MSI", "O", "AIG", "WELL", "AZO", "PSA", "D", "EXC", "TT", "CTVA", 
    "CNC", "AFL", "STZ", "SPG", "WMB", "HLT", "BIIB", "PAYC", "YUM", "FTNT", "DHI", "IQV", "PRU", "SYY", "MTD", "A", "NEM", "CTSH", 
    "GWW", "WBA", "KMI", "BKR", "K", "TGT", "HOOD", "AFRM", "PATH", "MNDY", "DOCN", "NET", "CRWD", "SE", "SQ", "ROKU", "PINS", "TWLO", 
    "SPOT", "UBER", "LYFT", "DASH", "CVNA", "CHWY", "Z", "W", "ETSY", "DKNG", "PENN", "WYNN", "LVS", "MGM", "RCL", "CCL", "NCLH", 
    "DAL", "UAL", "AAL", "LUV", "EXPE", "TRIP", "SHOP", "GLW", "FSLR", "SEDG", "RUN", "PLUG", "FCEL", "QS", "CHPT", "BLNK"
]))

# ==========================================
# [3. 백테스팅 및 지표 계산 (수학적 통계 검증)]
# ==========================================
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def run_strategy_backtest(symbol, df):
    """지정된 기술적 패턴(RSI 과매도, MACD 크로스)의 과거 승률을 벡터 연산으로 도출"""
    try:
        close = df['Close']
        rsi = vbt.RSI.run(close).rsi
        macd = vbt.MACD.run(close)
        
        entries = (rsi < 35) | (macd.macd_crossed_above(macd.signal))
        exits = (rsi > 70) | (macd.macd_crossed_below(macd.signal))
        
        pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=10000)
        win_rate = pf.stats().get('Win Rate [%]')
        return float(win_rate) if pd.notna(win_rate) else 0.0
    except:
        return 0.0

def calculate_indicators(df):
    """모든 기술적 지표를 누락 없이 계산"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss) + 1e-6))
    
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_f = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_f = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_f / neg_f) + 1e-6))
    
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    df['MA20'] = df['Close'].rolling(20).mean()
    df['STD'] = df['Close'].rolling(20).std()
    df['BB_Low'] = df['MA20'] - (df['STD'] * 2)
    df['BB_High'] = df['MA20'] + (df['STD'] * 2)

    df['MA10'] = df['Close'].rolling(10).mean()
    df['Disparity'] = (df['Close'] / df['MA10']) * 100
    df['OBV'] = (np.sign(delta) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = (df['OBV'].iloc[-1] - df['OBV'].iloc[-5]) / 5
    df['ROC3'] = df['Close'].pct_change(3) * 100
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    # CMF (세력 매집)
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-6)
    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-6)

    # ADX (추세 강도)
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=14).mean() / (df['ATR'] + 1e-6))
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=14).mean() / (df['ATR'] + 1e-6))
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di + 1e-6))
    df['ADX'] = dx.ewm(span=14).mean()
    
    # 달러 거래대금(유동성 필터)
    df['DollarVolume'] = df['Close'] * df['Volume']

    return df

# ==========================================
# [4. 외부 데이터 수집 (시장 지수, 뉴스, 실적)]
# ==========================================
def get_market_status():
    """VIX 및 나스닥 지수(^IXIC)를 통한 실시간 시장 상태 파악"""
    try:
        vix_data = yf.download("^VIX", period="1d", progress=False)
        vix = float(vix_data['Close'].iloc[-1])
        
        # 나스닥 종합지수 추적
        ndx_data = yf.download("^IXIC", period="2d", progress=False)
        if len(ndx_data) >= 2:
            change = float(((ndx_data['Close'].iloc[-1] / ndx_data['Close'].iloc[-2]) - 1) * 100)
        else:
            change = 0.0
            
        return vix, change
    except:
        return 20.0, 0.0

def get_target_price_fallback(ticker, curr_p, df_hist):
    try:
        recent_high = df_hist['High'].iloc[-120:].max()
        bb_high = df_hist['BB_High'].iloc[-1]
        target = max(recent_high, bb_high * 1.05)
        return target if target > curr_p else curr_p * 1.1, "📈Tech"
    except:
        return curr_p * 1.1, "Est."

def get_external_data(s, t_obj, curr_p, df_hist):
    data = {"sentiment": "중립", "earnings": "안정", "target": None, "upside": "N/A", "upside_tag": "", "score": 0}
    try:
        # [수정됨] 새로운 gemini_client 규격에 맞춘 API 호출
        try:
            news = t_obj.news[:3]
            if news and gemini_client:
                titles = [n['title'] for n in news]
                prompt = f"Stock {s}: {titles}. Respond exactly one word: Positive, Negative, or Neutral."
                
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash', # 권장되는 최신 모델
                    contents=prompt
                )
                res = response.text.strip()
                
                if "Positive" in res: data["sentiment"], data["score"] = "호재", data["score"] + 20
                elif "Negative" in res: data["sentiment"] = "악재"
        except Exception as e: 
            pass
        
        info = {}
        try: info = t_obj.info
        except: pass
            
        target = info.get('targetMeanPrice') or info.get('targetMedianPrice')
        source_label = "🏦Analyst"
        if not target or float(target) <= curr_p:
            target, source_label = get_target_price_fallback(s, curr_p, df_hist)
        
        if target and float(target) > 0:
            upside_val = ((target / curr_p) - 1) * 100
            data["upside"], data["upside_tag"] = f"{upside_val:.1f}", f"({source_label})"
            # 20% 이상 괴리 시 15점 가점
            if upside_val > 20: data["score"] += 15
        
        try:
            cal = t_obj.calendar
            e_date = None
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                e_date = cal.iloc[0, 0] if 0 in cal.columns else cal.iloc[0, cal.columns.get_loc('Earnings Date')]
            elif isinstance(cal, dict):
                e_date = cal.get('Earnings Date', [None])[0]

            if e_date:
                days = (pd.to_datetime(e_date).replace(tzinfo=None) - datetime.now().replace(tzinfo=None)).days
                if 0 <= days <= 7: data["earnings"], data["score"] = f"⚠️D-{days}", data["score"] - 40
        except: pass
    except Exception as e:
        print(f"External Data Error ({s}): {e}")
    return data

# ==========================================
# [5. 메인 퀀트 엔진 프로세스]
# ==========================================
def run_full_scan():
    print("🚀 NASDAQ Master-Quant System Starting...")
    if not TELEGRAM_TOKEN or not CHAT_ID: 
        return print("토큰 설정 확인 필요")
        
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    
    vix, m_perf = get_market_status()
    is_risky = float(vix) > 24.0 or float(m_perf) < -1.5
    risk_mode = "⚠️방어운전" if is_risky else "✅안정적"
    score_min = 45 if risk_mode == "⚠️방어운전" else 30

    # 동적 가중치 배열 (순서: 1.RSI, 2.MACD기울기, 3.거래량, 4.낙폭과대, 5.BB하단, 6.V자반등(데드캣), 7.CMF, 8.ADX)
    if is_risky:
        WEIGHTS = np.array([20, 5, 5, 20, 15, 20, 10, 5])
    else:
        WEIGHTS = np.array([10, 15, 15, 5, 5, 15, 20, 15])

    review_list, super_buys, strong_buys, normal_buys = [], [], [], []
    sector_momentum = {k: 0 for k in SECTORS.keys()}
    results = []

    print("📥 250일치 과거 데이터 일괄 다운로드 중 (백테스트 포함)...")
    bulk_data = yf.download(STOCKS, period="250d", group_by="ticker", progress=False, threads=True)

    for idx, s in enumerate(STOCKS):
        try:
            if s not in bulk_data.columns.levels[0]: continue
            df = bulk_data[s].dropna()
            if len(df) < 100: continue
            
            df = calculate_indicators(df)
            curr_p = float(df['Close'].iloc[-1])
            avg_dollar_vol = df['DollarVolume'].rolling(20).mean().iloc[-1]

            # 유동성 필터 (최근 20일 평균 거래대금 5천만 달러 이상)
            if avg_dollar_vol < 50_000_000: continue
            
            # [전일 RSI 과매도 적중률 복기]
            if len(df) >= 3 and df['RSI'].iloc[-2] < 35:
                hit = float(df['High'].iloc[-1]) >= float(df['Close'].iloc[-2]) * 1.025
                review_list.append(f"{s}:{'🎯' if hit else '⏳'}")

            drop_rate = float((1 - (curr_p / float(df['High'].max()))) * 100)
            
            macd_slope = (df['MACD'].iloc[-1] - df['MACD'].iloc[-5]) / 5
            is_turning = bool(df['MACD'].iloc[-1] > df['Signal'].iloc[-1])
            is_vol = bool(df['Volume'].iloc[-1] > df['Volume'].rolling(5).mean().iloc[-1] * 1.5)
            is_bb_support = bool(curr_p <= df['BB_Low'].iloc[-1] * 1.02)
            
            # [데드캣 vs V자 반등 로직]
            is_deadcat = bool(df['Disparity'].iloc[-3] < 92 and df['ROC3'].iloc[-1] > 2 and df['OBV_Slope'].iloc[-1] < 0)
            is_v_rebound = bool(df['Disparity'].iloc[-3] < 93 and df['ROC3'].iloc[-1] > 4 and df['OBV_Slope'].iloc[-1] > 0)
            
            if is_vol and curr_p > float(df['Close'].iloc[-2]):
                for s_name, stocks in SECTORS.items():
                    if s in stocks: sector_momentum[s_name] += 1

            # 벡터 내적을 통한 베이스 점수 도출
            features = np.array([
                1.0 if df['RSI'].iloc[-1] < 35 else 0.0,
                (min(max(macd_slope, 0) * 10, 1.5) if is_turning else 0.0),
                1.0 if is_vol else 0.0,
                1.0 if drop_rate > 30 else 0.0,
                1.0 if is_bb_support else 0.0,
                1.5 if is_v_rebound else (-1.0 if is_deadcat else 0.0),
                1.0 if df['CMF'].iloc[-1] > 0 else 0.0,
                1.0 if df['ADX'].iloc[-1].item() > 25 else 0.0
            ])
            tech_score = float(np.dot(features, WEIGHTS))

            # 점수가 25점 이상인 유망 종목만 외부 데이터 호출 & 백테스팅 (속도 최적화)
            if tech_score >= 25:
                t_obj = yf.Ticker(s)
                external = get_external_data(s, t_obj, curr_p, df)
                win_rate = run_strategy_backtest(s, df)
            else:
                external = {"sentiment": "➖생략", "earnings": "➖", "upside": "N/A", "upside_tag": "", "score": 0}
                win_rate = 0.0

            # [ATR 기반 포지션 사이징]
            atr = df['ATR'].iloc[-1]
            stop_loss = curr_p - (atr * 1.5)
            risk_per_share = curr_p - stop_loss if (curr_p - stop_loss) > 0 else 1
            max_risk_amount = TOTAL_CAPITAL * RISK_TOLERANCE_PER_TRADE
            
            recommended_shares = int(max_risk_amount / risk_per_share)
            alloc_pct = ((recommended_shares * curr_p) / TOTAL_CAPITAL) * 100

            results.append({
                "symbol": s, "price": curr_p, "rsi": float(df['RSI'].iloc[-1]), 
                "drop": drop_rate, "is_vol": is_vol, "is_bb": is_bb_support,
                "is_deadcat": is_deadcat, "is_v_rebound": is_v_rebound, "cmf": df['CMF'].iloc[-1],
                "external": external, "tech_score": tech_score, "win_rate": win_rate,
                "target_price": curr_p + (atr * 3), "stop_loss": stop_loss,
                "rec_shares": recommended_shares, "alloc_pct": alloc_pct
            })
            time.sleep(0.01)
        except Exception as e: continue

    # ==========================================
    # [6. 결과 집계 및 리포팅]
    # ==========================================
    hot_sectors = [k for k, v in sector_momentum.items() if v >= 2]
    
    for item in results:
        s = item['symbol']
        theme_bonus = 10 if any(s in SECTORS[hs] for hs in hot_sectors) else 0
        
        # 합산 및 메시지 작성
        total_score = item['tech_score'] + item['external']['score'] + theme_bonus
        upside_str = f"{item['external']['upside']}%" if item['external']['upside'] != "N/A" else "N/A"
        
        status_tag = ""
        if item['is_deadcat']: status_tag = "⚠️ [데드캣 경고] "
        elif item['is_v_rebound']: status_tag = "🚀 [V자 반등] "
        elif item['cmf'] > 0.1: status_tag = "🐳 [세력매집] "
        
        msg = (f"{status_tag}🔥 **`{s}`** (총점:{total_score:.1f})\n"
               f"📍 Price: ${item['price']:.2f} (RSI:{item['rsi']:.1f})\n"
               f"🎯 TP: ${item['target_price']:.2f} | 🆙 Upside: {upside_str} {item['external']['upside_tag']}\n"
               f"🛑 손절가: ${item['stop_loss']:.2f} | 🏆 과거 승률: {item['win_rate']:.1f}%\n"
               f"⚖️ 권장 비중: 자산의 {item['alloc_pct']:.1f}% ({item['rec_shares']}주)\n"
               f"📊 뉴스:{item['external']['sentiment']} | 낙폭:{item['drop']:.1f}% | 🏛 실적:{item['external']['earnings']}\n"
               f"🔗 https://tossinvest.com/stocks/{s}")

        if "⚠️" in item['external']['earnings']: continue
        
        if total_score >= 85 and item['is_vol'] and risk_mode == "✅안정적":
            super_buys.append(msg)
        elif total_score >= 65:
            strong_buys.append(msg)
        elif total_score >= score_min:
            normal_buys.append(msg)

    # 텔레그램 메시지 포맷팅
    header = [
        f"🇺🇸 *QUANT PORTFOLIO REPORT*",
        f"📅 {now.strftime('%Y-%m-%d %H:%M')} | {risk_mode}",
        f"📉 VIX: {vix:.2f} | NASDAQ: {m_perf:+.2f}%",
        f"💼 기준 자산: ${TOTAL_CAPITAL:,.0f} (1회 리스크 {RISK_TOLERANCE_PER_TRADE*100}%)",
        f"🚩 Hot Sectors: {', '.join(hot_sectors) if hot_sectors else '없음'}",
        "━━━━━━━━━━━━━━",
        f"📊 **[전일 RSI 과매도 적중률]**\n" + (", ".join(review_list[:8]) if review_list else "데이터 부족"),
        "━━━━━━━━━━━━━━"
    ]
    
    full_text = "\n".join(header + 
                ([f"🚀 **[SUPER BUY]** - 강력 추천\n" + "\n\n".join(super_buys[:3])] if super_buys else []) +
                ([f"\n💎 **[STRONG BUY]** - 매수 유효\n" + "\n\n".join(strong_buys[:5])] if strong_buys else []) +
                ([f"\n🔍 **[NORMAL BUY]** - 관망/소액\n" + "\n\n".join(normal_buys[:8])] if normal_buys else []) +
                ["━━━━━━━━━━━━━━", f"✅ {len(results)}개 종목 분석 완료"])

    print("\n텔레그램 전송 중...")
    for part in [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      json={"chat_id": CHAT_ID, "text": part, "parse_mode": "Markdown", "disable_web_page_preview": True})
    print("완료되었습니다.")

if __name__ == "__main__":
    run_full_scan()






