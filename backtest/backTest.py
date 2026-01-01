# import yfinance as yf
# import pandas as pd
# import numpy as np
#
# # ETF 목록 (공격자산 + 수비자산 + 카나리아 자산)
# etfs = ['XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'VNQ', 'XLK', 'XLP', 'XLY', 'XLU', 'SPY', 'IWM', 'VEA', 'VWO', 'TLT', 'IEF', 'DBC', 'TIP']
#
# # 데이터 가져오기
# historical_data = {etf: yf.download(etf, start='2010-01-01', end='2024-01-01')['Adj Close'] for etf in etfs}
#
# # 데이터프레임으로 변환
# df = pd.concat(historical_data.values(), axis=1, keys=historical_data.keys())
#
# # 모멘텀 스코어 계산 함수
# def momentum_score(prices):
#     returns = np.log(prices).diff()
#     return returns.rolling(21).sum() + returns.rolling(63).sum() + returns.rolling(126).sum() + returns.rolling(252).sum()
#
# # 모멘텀 스코어 계산
# momentum_scores = df.apply(momentum_score)
#
# # HAA 전략 함수
# def haa_strategy(data, tip_data, original=False):
#     portfolio = pd.DataFrame(index=data.index, columns=data.columns, data=0.0, dtype=float)
#     for date in portfolio.index:
#         if tip_data.loc[date].item() > 0:  # .item() 추가
#             if original:
#                 assets = ['SPY', 'IWM', 'VEA', 'VWO', 'TLT', 'IEF', 'DBC', 'VNQ']
#             else:
#                 assets = data.columns.tolist()
#             top_assets = data.loc[date, assets].nlargest(4).index
#             portfolio.loc[date, top_assets] = 0.25
#         else:
#             if data.loc[date, 'IEF'].item() > 0:  # .item() 추가
#                 portfolio.loc[date, 'IEF'] = 1.0
#             else:
#                 portfolio.loc[date, 'TLT'] = 1.0
#     return portfolio
#
# # 원래 HAA 전략과 수정된 HAA 전략 실행
# original_portfolio = haa_strategy(momentum_scores, momentum_scores['TIP'], original=True)
# modified_portfolio = haa_strategy(momentum_scores, momentum_scores['TIP'])
#
# # 수익률 계산
# original_returns = (original_portfolio * df.pct_change()).sum(axis=1)
# modified_returns = (modified_portfolio * df.pct_change()).sum(axis=1)
#
# # 누적 수익률 계산
# original_cumulative_returns = (1 + original_returns).cumprod()
# modified_cumulative_returns = (1 + modified_returns).cumprod()
#
# # MDD 계산
# def calculate_mdd(returns):
#     cumulative = (1 + returns).cumprod()
#     running_max = np.maximum.accumulate(cumulative)
#     drawdown = (cumulative - running_max) / running_max
#     return drawdown.min()
#
# original_mdd = calculate_mdd(original_returns)
# modified_mdd = calculate_mdd(modified_returns)
#
# # 평균 수익률 계산
# original_avg_return = original_returns.mean() * 252  # 연율화
# modified_avg_return = modified_returns.mean() * 252  # 연율화
#
# # 결과 출력
# print("Original HAA Strategy:")
# print(f"MDD: {original_mdd:.2%}")
# print(f"Average Annual Return: {original_avg_return:.2%}")
# print(f"Cumulative Return: {original_cumulative_returns.iloc[-1] - 1:.2%}")
#
# print("\nModified HAA Strategy (with Sector ETFs):")
# print(f"MDD: {modified_mdd:.2%}")
# print(f"Average Annual Return: {modified_avg_return:.2%}")
# print(f"Cumulative Return: {modified_cumulative_returns.iloc[-1] - 1:.2%}")
#
# import yfinance as yf
# import pandas as pd
# import numpy as np
#
# # ETF 목록 (공격자산 + 수비자산 + 카나리아 자산)
# etfs = ['XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'VNQ', 'XLK', 'XLP', 'XLY', 'XLU', 'SPY', 'IWM', 'VEA', 'VWO', 'TLT', 'IEF', 'DBC', 'TIP']
#
# # 데이터 가져오기
# historical_data = {etf: yf.download(etf, start='2010-01-01', end='2024-01-01')['Adj Close'] for etf in etfs}
#
# # 데이터프레임으로 변환
# df = pd.concat(historical_data.values(), axis=1, keys=historical_data.keys())
#
# # 모멘텀 스코어 계산 함수
# def momentum_score(prices):
#     returns = np.log(prices).diff()
#     return returns.rolling(21).sum() + returns.rolling(63).sum() + returns.rolling(126).sum() + returns.rolling(252).sum()
#
# # 모멘텀 스코어 계산
# momentum_scores = df.apply(momentum_score)
#
# # HAA 전략 함수
# def haa_strategy(data, tip_data, original=False):
#     portfolio = pd.DataFrame(index=data.index, columns=data.columns, data=0.0, dtype=float)
#     for date in portfolio.index:
#         if tip_data.loc[date].item() > 0:
#             if original:
#                 assets = ['SPY', 'IWM', 'VEA', 'VWO', 'TLT', 'IEF', 'DBC', 'VNQ']
#             else:
#                 assets = data.columns.tolist()
#             top_assets = data.loc[date, assets].nlargest(4).index
#             portfolio.loc[date, top_assets] = 0.25
#         else:
#             if data.loc[date, 'IEF'].item() > 0:
#                 portfolio.loc[date, 'IEF'] = 1.0
#             else:
#                 portfolio.loc[date, 'TLT'] = 1.0
#     return portfolio
#
# # 원래 HAA 전략과 수정된 HAA 전략 실행
# original_portfolio = haa_strategy(momentum_scores, momentum_scores['TIP'], original=True)
# modified_portfolio = haa_strategy(momentum_scores, momentum_scores['TIP'])
#
# # 수익률 계산
# original_returns = (original_portfolio * df.pct_change()).sum(axis=1)
# modified_returns = (modified_portfolio * df.pct_change()).sum(axis=1)
#
# # 누적 수익률 계산
# original_cumulative_returns = (1 + original_returns).cumprod()
# modified_cumulative_returns = (1 + modified_returns).cumprod()
#
# # MDD 계산
# def calculate_mdd(returns):
#     cumulative = (1 + returns).cumprod()
#     running_max = np.maximum.accumulate(cumulative)
#     drawdown = (cumulative - running_max) / running_max
#     return drawdown.min()
#
# original_mdd = calculate_mdd(original_returns)
# modified_mdd = calculate_mdd(modified_returns)
#
# # 평균 수익률 계산
# original_avg_return = original_returns.mean() * 252  # 연율화
# modified_avg_return = modified_returns.mean() * 252  # 연율화
#
# # 결과 출력
# print("Original HAA Strategy:")
# print(f"MDD: {original_mdd:.2%}")
# print(f"Average Annual Return: {original_avg_return:.2%}")
# print(f"Cumulative Return: {original_cumulative_returns.iloc[-1] - 1:.2%}")
#
# print("\nModified HAA Strategy (with Sector ETFs):")
# print(f"MDD: {modified_mdd:.2%}")
# print(f"Average Annual Return: {modified_avg_return:.2%}")
# print(f"Cumulative Return: {modified_cumulative_returns.iloc[-1] - 1:.2%}")

import yfinance as yf
import pandas as pd
import numpy as np

# ETF 목록 (공격자산 + 수비자산 + 카나리아 자산)
etfs = ['XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'VNQ', 'XLK', 'XLP', 'XLY', 'XLU', 'SPY', 'IWM', 'VEA', 'VWO', 'TLT', 'IEF',
        'DBC', 'TIP']

# 데이터 가져오기
historical_data = {etf: yf.download(etf, start='2010-01-01', end='2023-03-01')['Adj Close'] for etf in etfs}

# 데이터프레임으로 변환
df = pd.concat(historical_data.values(), axis=1, keys=historical_data.keys())


# 모멘텀 스코어 계산 함수
def momentum_score(prices):
    returns = np.log(prices).diff()
    return returns.rolling(21).sum() + returns.rolling(63).sum() + returns.rolling(126).sum() + returns.rolling(
        252).sum()


# 모멘텀 스코어 계산
momentum_scores = df.apply(momentum_score)


# 월말 여부를 확인하는 함수
def is_month_end(date):
    return date.month != (date + pd.Timedelta(days=1)).month


# HAA 전략 함수 (TIP 모멘텀 부호 변화 리밸런싱)
def haa_strategy_tip(data, tip_data, original=False):
    portfolio = pd.DataFrame(index=data.index, columns=data.columns, data=0.0, dtype=float)
    last_tip_sign = np.sign(tip_data.iloc[0].item())

    for date in portfolio.index:
        current_tip_sign = np.sign(tip_data.loc[date].item())

        # TIP 모멘텀 부호가 바뀌는 경우 리밸런싱
        if current_tip_sign != last_tip_sign:
            if current_tip_sign > 0:
                if original:
                    assets = ['SPY', 'IWM', 'VEA', 'VWO', 'TLT', 'IEF', 'DBC', 'VNQ']
                else:
                    assets = data.columns.tolist()
                top_assets = data.loc[date, assets].nlargest(4).index
                portfolio.loc[date, top_assets] = 0.25
            else:
                if data.loc[date, 'IEF'].item() > 0:
                    portfolio.loc[date, 'IEF'] = 1.0
                else:
                    portfolio.loc[date, 'TLT'] = 1.0

            last_tip_sign = current_tip_sign

    return portfolio


# HAA 전략 함수 (월말 리밸런싱)
def haa_strategy_month_end(data, tip_data, original=False):
    portfolio = pd.DataFrame(index=data.index, columns=data.columns, data=0.0, dtype=float)

    for date in portfolio.index:
        # 월말인 경우 리밸런싱
        if is_month_end(date):
            if tip_data.loc[date].item() > 0:
                if original:
                    assets = ['SPY', 'IWM', 'VEA', 'VWO', 'TLT', 'IEF', 'DBC', 'VNQ']
                else:
                    assets = data.columns.tolist()
                top_assets = data.loc[date, assets].nlargest(4).index
                portfolio.loc[date, top_assets] = 0.25
            else:
                if data.loc[date, 'IEF'].item() > 0:
                    portfolio.loc[date, 'IEF'] = 1.0
                else:
                    portfolio.loc[date, 'TLT'] = 1.0

    return portfolio


# TIP 모멘텀 부호 변화 리밸런싱 전략 실행
portfolio_tip_original = haa_strategy_tip(momentum_scores, momentum_scores['TIP'], original=True)
portfolio_tip_modified = haa_strategy_tip(momentum_scores, momentum_scores['TIP'])

# 월말 리밸런싱 전략 실행
portfolio_month_end_original = haa_strategy_month_end(momentum_scores, momentum_scores['TIP'], original=True)
portfolio_month_end_modified = haa_strategy_month_end(momentum_scores, momentum_scores['TIP'])

# 수익률 계산
returns_tip_original = (portfolio_tip_original * df.pct_change()).sum(axis=1)
returns_tip_modified = (portfolio_tip_modified * df.pct_change()).sum(axis=1)

returns_month_end_original = (portfolio_month_end_original * df.pct_change()).sum(axis=1)
returns_month_end_modified = (portfolio_month_end_modified * df.pct_change()).sum(axis=1)

# 누적 수익률 계산
cumulative_returns_tip_original = (1 + returns_tip_original).cumprod()
cumulative_returns_tip_modified = (1 + returns_tip_modified).cumprod()

cumulative_returns_month_end_original = (1 + returns_month_end_original).cumprod()
cumulative_returns_month_end_modified = (1 + returns_month_end_modified).cumprod()


# MDD 계산 함수
def calculate_mdd(returns):
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


mdd_tip_original = calculate_mdd(returns_tip_original)
mdd_tip_modified = calculate_mdd(returns_tip_modified)

mdd_month_end_original = calculate_mdd(returns_month_end_original)
mdd_month_end_modified = calculate_mdd(returns_month_end_modified)

# 평균 수익률 계산
avg_return_tip_original = returns_tip_original.mean() * 252  # 연율화
avg_return_tip_modified = returns_tip_modified.mean() * 252  # 연율화

avg_return_month_end_original = returns_month_end_original.mean() * 252  # 연율화
avg_return_month_end_modified = returns_month_end_modified.mean() * 252  # 연율화

# 결과 출력
print("TIP Momentum Sign Change Rebalancing - Original HAA Strategy:")
print(f"MDD: {mdd_tip_original:.2%}")
print(f"Average Annual Return: {avg_return_tip_original:.2%}")
print(f"Cumulative Return: {cumulative_returns_tip_original.iloc[-1] - 1:.2%}")

print("\nTIP Momentum Sign Change Rebalancing - Modified HAA Strategy:")
print(f"MDD: {mdd_tip_modified:.2%}")
print(f"Average Annual Return: {avg_return_tip_modified:.2%}")
print(f"Cumulative Return: {cumulative_returns_tip_modified.iloc[-1] - 1:.2%}")

print("\nMonth-End Rebalancing - Original HAA Strategy:")
print(f"MDD: {mdd_month_end_original:.2%}")
print(f"Average Annual Return: {avg_return_month_end_original:.2%}")
print(f"Cumulative Return: {cumulative_returns_month_end_original.iloc[-1] - 1:.2%}")

print("\nMonth-End Rebalancing - Modified HAA Strategy:")
print(f"MDD: {mdd_month_end_modified:.2%}")
print(f"Average Annual Return: {avg_return_month_end_modified:.2%}")
print(f"Cumulative Return: {cumulative_returns_month_end_modified.iloc[-1] - 1:.2%}")