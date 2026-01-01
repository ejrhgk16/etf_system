import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import math


# ---------------------------------------------------------
# 1. Roofing Filter 함수 정의 (Ehlers Logic)
# ---------------------------------------------------------
def calculate_roofing_filter(series, high_pass_period=48, smooth_period=10):
    """
    Roofing Filter = High Pass Filter + Super Smoother Filter
    이후 이 값에 Stochastic을 적용해야 'Roofing Filtered Stochastic'이 됩니다.
    """
    df_calc = pd.DataFrame({'price': series})

    # 1. High Pass Filter
    # ----------------------------------------
    # alpha1 = (math.cos(math.radians(360 / high_pass_period)) + math.sin(math.radians(360 / high_pass_period)) - 1) / math.cos(math.radians(360 / high_pass_period))
    # 위 식은 근사치 혹은 변형이 많으므로 Ehlers의 표준 파이썬 변환 로직을 사용합니다.

    alpha_arg = 2 * np.pi / high_pass_period if high_pass_period > 0 else 0
    alpha1 = (1 - np.sin(alpha_arg)) / np.cos(alpha_arg)

    df_calc['hp'] = 0.0
    price = df_calc['price'].values
    hp = np.zeros_like(price)

    # HP Filter Loop
    for i in range(2, len(price)):
        hp[i] = (1 - alpha1 / 2) ** 2 * (price[i] - 2 * price[i - 1] + price[i - 2]) + 2 * (1 - alpha1) * hp[i - 1] - (
                    1 - alpha1) ** 2 * hp[i - 2]

    df_calc['hp'] = hp

    # 2. Super Smoother Filter
    # ----------------------------------------
    a1 = np.exp(-1.414 * np.pi / smooth_period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / smooth_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    ss = np.zeros_like(hp)
    for i in range(2, len(hp)):
        ss[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * ss[i - 1] + c3 * ss[i - 2]

    return pd.Series(ss, index=series.index)


# ---------------------------------------------------------
# 2. 지표 계산 함수
# ---------------------------------------------------------
def add_indicators(df, rsi_len=14, stoch_k=14, stoch_d=3, roofing_hp_len=48, roofing_smooth_len=10):
    # 1. 기본 RSI 계산
    rsi_col_name = f'RSI_{rsi_len}'
    df[rsi_col_name] = ta.rsi(df['SPY'], length=rsi_len)

    # -------------------------------------------------------
    # [NEW] 커스텀 RSI 스케일링 (70~100 -> 0~1 / 30~0 -> 0~-1)
    # -------------------------------------------------------
    conditions = [
        df[rsi_col_name] >= 70,  # 과매수 구간
        df[rsi_col_name] <= 30  # 과매도 구간
    ]

    choices = [
        (df[rsi_col_name] - 70) / 30,  # 70->0, 100->1
        (df[rsi_col_name] - 30) / 30  # 30->0, 0->-1
    ]

    # 그 외 구간(30~70)은 0으로 처리
    df['RSI_Custom'] = np.select(conditions, choices, default=0)

    # 2. Roofing Filter 계산
    roofing_series = calculate_roofing_filter(df['SPY'], high_pass_period=roofing_hp_len,
                                              smooth_period=roofing_smooth_len)
    df['Roofing_Val'] = roofing_series

    # 3. Roofing Filtered Stochastic
    lowest_low = roofing_series.rolling(window=stoch_k).min()
    highest_high = roofing_series.rolling(window=stoch_k).max()

    df['Roofing_Stoch_K'] = 100 * ((roofing_series - lowest_low) / (highest_high - lowest_low))
    df['Roofing_Stoch_D'] = df['Roofing_Stoch_K'].rolling(window=stoch_d).mean()

    return df

# ---------------------------------------------------------
# 3. 백테스트 실행 함수
# ---------------------------------------------------------
def run_backtest(df, initial_capital=10000):
    capital = initial_capital
    position = 0  # 0: None, 1: Long, -1: Short
    entry_price = 0

    trades = []
    equity_curve = []

    print("--- Backtest Start ---")

    # 데이터 순회
    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        # 현재 포지션 상태
        current_pos = position

        # -----------------------------------------------------------
        # [[ 사용자 작성 구역: 진입 및 청산 조건 ]]
        # 아래 변수들에 True/False 논리식을 할당하세요.
        # curr['SPY'], curr['RSI_14'], curr['Roofing_Stoch_K'] 등으로 접근 가능
        # -----------------------------------------------------------

        entry_long_cond = curr['RSI_14'] # <--- 여기에 롱 진입 조건 작성 (예: curr['RSI_14'] < 30)
        entry_short_cond = ''  # <--- 여기에 숏 진입 조건 작성

        exit_long_cond = ''# <--- 여기에 롱 청산 조건 작성
        exit_short_cond = '' # <--- 여기에 숏 청산 조건 작성

        # -----------------------------------------------------------
        # 로직 처리 (수정 불필요, 위 조건에 따라 동작)
        # -----------------------------------------------------------

        # 1. 청산 (Exit)
        if current_pos == 1 and exit_long_cond:
            # Long 청산
            pnl = (curr['SPY'] - entry_price) / entry_price * capital
            capital += pnl
            position = 0
            trades.append({'date': curr.name, 'type': 'exit_long', 'price': curr['SPY'], 'capital': capital})

        elif current_pos == -1 and exit_short_cond:
            # Short 청산
            pnl = (entry_price - curr['SPY']) / entry_price * capital
            capital += pnl
            position = 0
            trades.append({'date': curr.name, 'type': 'exit_short', 'price': curr['SPY'], 'capital': capital})

        # 2. 진입 (Entry) - 포지션이 없을 때만
        if position == 0:
            if entry_long_cond:
                position = 1
                entry_price = curr['SPY']
                trades.append({'date': curr.name, 'type': 'entry_long', 'price': curr['SPY']})
            elif entry_short_cond:
                position = -1
                entry_price = curr['SPY']
                trades.append({'date': curr.name, 'type': 'entry_short', 'price': curr['SPY']})

        equity_curve.append(capital)

    print(f"--- Backtest End ---\nFinal Capital: {capital:.2f}")
    return pd.DataFrame(trades)


# ---------------------------------------------------------
# 4. 메인 실행 블록
# ---------------------------------------------------------

# 설정값
ticker_list = ["SPY", "^VIX", "^VIX1D", "^VIX3M", "SVXY", "VIXY", "SVIX"]
period_start = "2020-01-01"
period_end = "2023-12-31"

# 지표 파라미터 (여기서 길이 조절 가능)
RSI_LENGTH = 14
ROOFING_HP_LENGTH = 48  # High Pass Filter 길이
ROOFING_SMOOTH_LENGTH = 10  # Super Smoother 길이
STOCH_K_LENGTH = 14  # Roofing 값에 적용할 Stoch 기간
STOCH_D_LENGTH = 3

print(f"데이터 다운로드 중... ({period_start} ~ {period_end})")
# yfinance 다운로드
raw_data = yf.download(ticker_list, start=period_start, end=period_end, auto_adjust=True, progress=False)

# 종가(Close)만 추출하여 정리
# Multi-index 컬럼일 경우 처리 ('Close' 레벨 제거 혹은 선택)
if isinstance(raw_data.columns, pd.MultiIndex):
    try:
        df = raw_data["Close"].copy()
    except KeyError:
        # yfinance 버전에 따라 컬럼 구조가 다를 수 있음 (Adj Close 등)
        df = raw_data.xs('Close', axis=1, level=0).copy()
else:
    df = raw_data["Close"].copy()

df.dropna(inplace=True)

# 지표 추가 (SPY 기준)
print("지표 계산 중...")
df = add_indicators(df,
                    rsi_len=RSI_LENGTH,
                    stoch_k=STOCH_K_LENGTH,
                    stoch_d=STOCH_D_LENGTH,
                    roofing_hp_len=ROOFING_HP_LENGTH,
                    roofing_smooth_len=ROOFING_SMOOTH_LENGTH)

df.dropna(inplace=True)  # 지표 계산으로 생긴 NaN 제거

# 결과 확인용 출력
print(df[['SPY', f'RSI_{RSI_LENGTH}', 'Roofing_Stoch_K', 'Roofing_Stoch_D']].tail())

# 백테스트 실행
trades_df = run_backtest(df)