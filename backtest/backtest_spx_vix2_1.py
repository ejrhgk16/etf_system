import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 지표 계산 함수
# ---------------------------------------------------------
def add_indicators(df, rsi_len=2):
    # RSI 계산 시 'Close' 가격 사용 명시
    rsi_col_name = ('RSI_', str(rsi_len))
    df[rsi_col_name] = ta.rsi(df[('Close', '^SPX')], length=rsi_len)

    # RSI 과매수/과매도 기준에 따라 RSI_Custom 값 정규화
    conditions = [df[rsi_col_name] >= 90, df[rsi_col_name] <= 30]
    # RSI 30~10이 0~-1이 되도록 수식 수정 (30-10=20)
    choices = [(df[rsi_col_name] - 90) / 10, (df[rsi_col_name] - 30) / 30]
    
    df[('RSI_Custom', '')] = np.select(conditions, choices, default=0)

    spx_high = df.get(('High', '^SPX'))
    spx_low = df.get(('Low', '^SPX'))
    
    # 고가와 저가 데이터가 모두 존재할 때만 지표를 계산합니다.
    if spx_high is not None and spx_low is not None:
        median_price = (spx_high + spx_low) / 2

        # 2. Alligator 지표(jaw, teeth, lips) 계산 및 DataFrame에 추가
        #    일관성을 위해 ('jaw', '^SPX') 형태의 다중 인덱스로 저장합니다.
        df[('jaw', '^SPX')] = median_price.ewm(alpha=1/21, adjust=False).mean().shift(8)
        df[('teeth', '^SPX')] = median_price.ewm(alpha=1/13, adjust=False).mean().shift(5)
        df[('lips', '^SPX')] = median_price.ewm(alpha=1/8, adjust=False).mean().shift(3)
        
    return df

# ---------------------------------------------------------
# 2. 백테스트 실행 함수 (복리 적용)
# ---------------------------------------------------------
def run_backtest(df, initial_capital=10000):
    cash = initial_capital
    shares = 0
    position = None  # None, 'VIXM', 'SVXY'
    equity_curve = [{'date': df.index[0], 'equity': initial_capital}]
    trades_log = []
    invested_value = 0 
    VIXM_pnls = []
    SVXY_pnls = []
    commission_rate = 0.001 # 수수료 0.1%

    print("--- 백테스트 시작 (복리 적용) ---")
    for i in range(1, len(df)):
        prev = df.iloc[i-1] # 신호 계산 및 자산 기준가용 (어제)
        curr = df.iloc[i]   # 거래 실행 및 자산 평가용 (오늘)

        # --- 1. 리밸런싱 기준 자산 계산 (어제 종가 기준 총자산) ---
        prev_close_price = 0
        if position == 'VIXM': prev_close_price = prev.get(('Close', 'VIXM'), 0)
        elif position == 'SVXY': prev_close_price = prev.get(('Close', 'SVXY'), 0)
        rebalance_equity = cash + (shares * prev_close_price)

        # # 'prev'를 사용하여 어제 하루의 데이터로만 비교합니다.
        # is_long = (prev.get(('lips', '^SPX')) > prev.get(('teeth', '^SPX'))) and \
        #         (prev.get(('teeth', '^SPX')) > prev.get(('jaw', '^SPX')))

        # --- 2. 진입/청산 신호 생성 (어제 종가 기준) ---
        isContango = prev.get(('Close', '^VIX'), 0) < prev.get(('Close', '^VIX3M'), 0)
        VIXM_entry_cond = prev.get(('RSI_Custom', ''), 0) > 0 and isContango
        VIXM_exit_cond = prev.get(('RSI_', str(RSI_LENGTH)), 50) <= 50 or not isContango
        #VIXM_exit_cond = prev.get(('RSI_Custom', ''), 0) <= 0 or not isContango
        
        SVXY_entry_cond = prev.get(('RSI_Custom', ''), 0) < 0 and isContango
        SVXY_exit_cond = prev.get(('RSI_', str(RSI_LENGTH)), 50) >= 50 or not isContango
        #SVXY_exit_cond = prev.get(('RSI_Custom', ''), 0) >= 0 or not isContango

        # --- 3. 청산 로직 (오늘 시가 기준) ---
        if (position == 'VIXM' and VIXM_exit_cond) or (position == 'SVXY' and SVXY_exit_cond):
            asset_name = 'VIXM' if position == 'VIXM' else 'SVXY'
            exit_price = curr.get(('Open', asset_name), 0)
            if exit_price > 0 and not np.isnan(exit_price):
                sale_value = shares * exit_price
                commission = sale_value * commission_rate
                returned_value = sale_value - commission
                trade_pnl = returned_value - invested_value
                
                if position == 'VIXM': VIXM_pnls.append(trade_pnl)
                else: SVXY_pnls.append(trade_pnl)
                
                cash += returned_value
                trades_log.append({'date': curr.name, 'ticker': position, 'action': 'exit', 'shares': shares, 'price': exit_price, 'pnl': trade_pnl})
                shares, position, invested_value = 0, None, 0

        # --- 4. 진입 및 추가매수 로직 (오늘 시가 기준) ---
        # 신규 진입
        if position is None:
            if VIXM_entry_cond or SVXY_entry_cond:
                asset_name = 'VIXM' if VIXM_entry_cond else 'SVXY'
                # 복리 적용: rebalance_equity를 기준으로 투자 금액 결정
                target_value = rebalance_equity * abs(prev.get(('RSI_Custom', ''), 0))
                #target_value = initial_capital * abs(prev.get(('RSI_Custom', ''), 0))
                entry_price = curr.get(('Open', asset_name), 0)

                if entry_price > 0 and not np.isnan(entry_price):
                    shares_to_buy = math.floor(target_value / entry_price)
                    if shares_to_buy > 0:
                        buy_cost = shares_to_buy * entry_price
                        commission = buy_cost * commission_rate
                        if cash >= buy_cost + commission:
                            actual_shares = shares_to_buy
                        else :
                            actual_shares = math.floor(cash / (entry_price * (1 + commission_rate)))
                        if actual_shares > 0:
                            buy_cost = actual_shares * entry_price
                            commission = buy_cost * commission_rate
                            
                            cash -= (buy_cost + commission)
                            shares = actual_shares
                            position = asset_name
                            invested_value = buy_cost
                            trades_log.append({
                                'date': curr.name, 
                                'ticker': position, 
                                'action': 'entry_max', # 최대치 매수 표시
                                'shares': shares, 
                                'price': entry_price
                            })
        
        # 추가 매수 (물타기)
        else:
            if (position == 'VIXM' and VIXM_entry_cond) or (position == 'SVXY' and SVXY_entry_cond):
                # 복리 적용: rebalance_equity를 기준으로 투자 금액 결정
                target_value = rebalance_equity * abs(prev.get(('RSI_Custom', ''), 0))
                #target_value = initial_capital * abs(prev.get(('RSI_Custom', ''), 0))
                current_value = shares * prev_close_price
                
                if target_value > current_value: # and (target_value / portfolio_equity) > (current_value / portfolio_equity + 0.1):
                    value_to_add = target_value - current_value
                    add_price = curr.get(('Open', position), 0)
                    if add_price > 0 and not np.isnan(add_price):
                        shares_to_add = math.floor(value_to_add / add_price)
                        if shares_to_add > 0:
                            add_cost = shares_to_add * add_price
                            commission = add_cost * commission_rate
                            if cash >= add_cost + commission:
                                cash -= (add_cost + commission)
                                shares += shares_to_add
                                invested_value += add_cost
                                trades_log.append({'date': curr.name, 'ticker': position, 'action': 'add', 'shares': shares_to_add, 'price': add_price})

        # --- 5. 일일 자산 평가 및 기록 (오늘 종가 기준) ---
        todays_close_price = 0
        if position == 'VIXM': todays_close_price = curr.get(('Close', 'VIXM'), 0)
        elif position == 'SVXY': todays_close_price = curr.get(('Close', 'SVXY'), 0)
        portfolio_equity = cash + (shares * todays_close_price)
        equity_curve.append({'date': df.index[i], 'equity': portfolio_equity})

    print("--- 백테스트 종료 ---")
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    trades_df = pd.DataFrame(trades_log)
    if not equity_df.empty: print(f"최종 포트폴리오 자산: ${equity_df['equity'].iloc[-1]:,.2f}")
    return equity_df, trades_df, VIXM_pnls, SVXY_pnls

# ---------------------------------------------------------
# 3. 성과 분석 및 시각화
# ---------------------------------------------------------
def analyze_performance(equity_df, spx_series):
    if equity_df.empty or len(equity_df) < 2:
        print("\n성과 분석을 위한 거래 데이터가 부족합니다.")
        return
    
    returns = equity_df['equity'].pct_change().fillna(0)
    days = (equity_df.index[-1] - equity_df.index[0]).days
    cagr = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0])**(365.0 / days) - 1 if days > 0 else 0
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = cagr / volatility if volatility != 0 else 0
    rolling_max = equity_df['equity'].cummax()
    daily_drawdown = (equity_df['equity'] / rolling_max) - 1.0
    mdd = daily_drawdown.min()

    print("\n--- 성과 분석 ---")
    print(f"연복리수익률 (CAGR): {cagr:.2%}")
    print(f"변동성 (Volatility): {volatility:.2%}")
    print(f"샤프 지수 (Sharpe Ratio): {sharpe_ratio:.2f}")
    print(f"최대 낙폭 (MDD): {mdd:.2%}")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(equity_df.index, equity_df['equity'], label='전략', color='royalblue', linewidth=2)
    spx_benchmark = spx_series.loc[equity_df.index[0]:equity_df.index[-1]]
    spx_scaled = (spx_benchmark / spx_benchmark.iloc[0]) * equity_df['equity'].iloc[0]
    ax.plot(spx_scaled.index, spx_scaled, label='spx', color='grey', linestyle='--')
    ax.set_title('전략 성과(복리) vs. SPX', fontsize=16)
    ax.set_xlabel('날짜'); ax.set_ylabel('포트폴리오 가치 ($)')
    ax.legend(); ax.grid(True)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${int(y/1000)}k'))
    plt.tight_layout()
    plt.savefig('backtest_performance2_1.png')
    print("\n그래프가 'backtest_performance2.png' 파일로 저장되었습니다.")

# ---------------------------------------------------------
# 4. 거래 유형별 성과 분석 함수
# ---------------------------------------------------------
def analyze_trade_performance(VIXM_pnls, SVXY_pnls):
    print("\n--- 거래 유형별 성과 분석 ---")
    def print_stats(title, pnl_list):
        if not pnl_list: print(f"\n{title}: 거래 없음"); return
        win_rate = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
        print(f"\n{title}")
        print(f"  - 총 수익/손실: ${sum(pnl_list):,.2f}")
        print(f"  - 승률: {win_rate:.2f}% ({sum(1 for p in pnl_list if p > 0)}승 / {len(pnl_list) - sum(1 for p in pnl_list if p > 0)}패)")

    print_stats("VIXM (RSI_Custom > 0) 거래", VIXM_pnls)
    print_stats("SVXY (RSI_Custom < 0) 거래", SVXY_pnls)
    print_stats("전체", VIXM_pnls + SVXY_pnls)

# ---------------------------------------------------------
# 5. 메인 실행 블록
# ---------------------------------------------------------
if __name__ == "__main__":
    ticker_list = ["^SPX", "^VIX",  "^VIX3M", "SVXY", "VIXM"]
    period_start = "2010-03-01"
    period_end = "2025-12-31"
    RSI_LENGTH = 2

    print(f"데이터 다운로드 중... ({period_start} ~ {period_end})")
    df = yf.download(ticker_list, start=period_start, end=period_end, auto_adjust=True, progress=False)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    
    print("지표 계산 중...")
    df = add_indicators(df, rsi_len=RSI_LENGTH)
    df.dropna(inplace=True)

    equity_df, trades_df, VIXM_pnls, SVXY_pnls = run_backtest(df, initial_capital=10000)

    if not trades_df.empty:
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print("\n--- 거래 기록 ---")
        print(trades_df.to_string())
    
    analyze_trade_performance(VIXM_pnls, SVXY_pnls)
    analyze_performance(equity_df, df[('Close', '^SPX')])
