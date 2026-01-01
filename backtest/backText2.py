import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


class HAA:
    def __init__(self, aggressive_assets, defensive_assets= ['BIL', 'IEF'], aggressive_assets2=[]):

        self.aggressive_assets = aggressive_assets
        self.aggressive_assets2 = aggressive_assets2
        self.defensive_assets = defensive_assets
        self.canary_asset = 'TIP'
        self.top_n = 4  # Top 4 aggressive assets
        self.rebalance_method = 'hybrid'
        self.momentum_periods = [1, 3, 6, 12]  # 13612U 모멘텀 계산 기간
        self.last_rebalance_date = None

    def fetch_data(self, start_date, end_date):
        # group_by='column'과 threads=True를 설정하여 안정성을 높입니다.
        raw_aggressive = yf.download(self.aggressive_assets, start=start_date, end=end_date)

        # 데이터가 비어있는지 확인
        if raw_aggressive.empty:
            print("데이터를 가져오지 못했습니다. 티커명이나 인터넷 연결을 확인하세요.")
            return pd.DataFrame()

        # 최신 yfinance는 Adj Close가 없을 경우 Close를 사용하기도 합니다.
        if 'Adj Close' in raw_aggressive.columns:
            aggressive_data = raw_aggressive['Adj Close']
        else:
            aggressive_data = raw_aggressive['Close']

        # 데이터가 Series인 경우(자산이 1개일 때) DataFrame으로 변환
        if isinstance(aggressive_data, pd.Series):
            aggressive_data = aggressive_data.to_frame()

        # 컬럼명 설정 (멀티인덱스 방지)
        aggressive_data.columns = ['a_' + col for col in aggressive_data.columns]

        # 방어 자산 데이터 가져오기
        raw_defensive = yf.download(self.defensive_assets, start=start_date, end=end_date)
        defensive_data = raw_defensive['Adj Close'] if 'Adj Close' in raw_defensive.columns else raw_defensive['Close']
        if isinstance(defensive_data, pd.Series):
            defensive_data = defensive_data.to_frame()
        defensive_data.columns = ['d_' + col for col in defensive_data.columns]

        # 카나리 자산 데이터 가져오기
        raw_canary = yf.download(self.canary_asset, start=start_date, end=end_date)
        canary_data = raw_canary['Adj Close'] if 'Adj Close' in raw_canary.columns else raw_canary['Close']
        canary_data = pd.DataFrame(canary_data)
        canary_data.columns = [self.canary_asset]

        self.aggressive_assets = list(aggressive_data.columns)
        self.defensive_assets = list(defensive_data.columns)

        # 나머지 결합 로직...
        all_data = pd.concat([aggressive_data, defensive_data, canary_data], axis=1)
        return all_data.dropna()  # 결측치 제거 추가

    def calculate_momentum(self, prices):
        momentum = pd.Series(index=prices.index)
        monthly_prices = prices.resample('ME').last()

        # for i in range(12, len(monthly_prices)):  # 12개월 이후부터 계산 시작
        #     p0 = monthly_prices.iloc[i]  # 현재 가격
        #     returns = []
        #     for period in self.momentum_periods:
        #         past_price = monthly_prices.iloc[i - period]
        #         monthly_return = (p0 / past_price - 1)
        #         returns.append(monthly_return)
        #     momentum.loc[monthly_prices.index[i]] = np.mean(returns)
        #
        # return momentum.reindex(prices.index).ffill().dropna()

        momentum = pd.Series(index=prices.index)

        for i in range(360, len(prices)):  # 360일부터 마지막 날짜까지 계산
            p0 = prices.iloc[i]  # 현재 가격
            returns = []

            # 30일, 90일, 180일, 360일 수익률 계산
            for period in [30, 90, 180, 360]:
                past_price = prices.iloc[i - period]
                daily_return = (p0 / past_price - 1)  # 수익률 계산
                returns.append(daily_return)

            # 평균 모멘텀 계산
            momentum.iloc[i] = np.mean(returns)

        return momentum.dropna()



    def calculate_momentum_w(self, prices):#가중 모멘텀 계산
        momentum = pd.Series(index=prices.index)
        monthly_prices = prices.resample('ME').last()

        # for i in range(12, len(monthly_prices)):  # 12개월 이후부터 계산 시작
        #     p0 = monthly_prices.iloc[i]  # 현재 가격
        #     returns = []
        #     for period in self.momentum_periods:
        #         past_price = monthly_prices.iloc[i - period]
        #         monthly_return = (p0 / past_price - 1)
        #         returns.append(monthly_return)
        #     momentum.loc[monthly_prices.index[i]] = np.mean(returns)
        #
        # return momentum.reindex(prices.index).ffill().dropna()

        momentum = pd.Series(index=prices.index)

        for i in range(360, len(prices)):  # 360일부터 마지막 날짜까지 계산
            p0 = prices.iloc[i]  # 현재 가격
            returns = []
            weighted_returns = 0
            total_weight = 0

            periods = [
                {'p': 30, 'w': 4},
                {'p': 90, 'w': 2},
                {'p': 180, 'w': 1},
            ]

            # 30일, 90일, 180일, 360일 수익률 계산
            for period in periods:
                past_price = prices.iloc[i - period['p']]
                daily_return = (p0 / past_price - 1)  # 수익률 계산
                weighted_returns += daily_return * period['w']
                total_weight += period['w']

            # 가중 평균 모멘텀 계산
            momentum.iloc[i] = weighted_returns / total_weight

        return momentum.dropna()

    def calculate_momentum_136(self, prices):

        momentum = pd.Series(index=prices.index)

        for i in range(180, len(prices)):  # 360일부터 마지막 날짜까지 계산
            p0 = prices.iloc[i]  # 현재 가격
            returns = []

            # 30일, 90일, 180일 수익률 계산
            for period in [30, 90, 180]:
                past_price = prices.iloc[i - period]
                daily_return = (p0 / past_price - 1)  # 수익률 계산
                returns.append(daily_return)

            # 평균 모멘텀 계산
            momentum.iloc[i] = np.mean(returns)

        return momentum.dropna()



    def select_assets(self, momentum):
        canary_momentum = momentum[self.canary_asset]
        print(canary_momentum)


        if canary_momentum <= 0:
            # 카나리 자산 모멘텀이 음수일 때
            # print("momentum", momentum[self.defensive_assets])
            best_defensive = momentum[self.defensive_assets].idxmax()
            return pd.Series({best_defensive: 1.0})
        else: # 카나리 자산 모멘텀이 양수일 때

            if(len(self.aggressive_assets2)>0): #굥격자산도 배분 해보기

                top_aggressive = momentum[self.aggressive_assets].nlargest(2)
                top_aggressive_2 = momentum[self.aggressive_assets2].nlargest(1)
                selected = {}
                cash_asset = momentum[self.defensive_assets].idxmax()  # 수비 자산 중 모멘텀이 가장 높은 자산

                for asset, asset_momentum in top_aggressive.items():
                    if asset_momentum > 0:
                        selected[asset] = 0.25  # 25% 비중
                    else:
                        selected[cash_asset] = selected.get(cash_asset, 0) + 0.25  # 현금(수비 자산) 비중 증가

                for asset, asset_momentum in top_aggressive_2.items():
                    if asset_momentum > 0:
                        selected[asset] = 0.25  # 25% 비중
                    else:
                        selected[cash_asset] = selected.get(cash_asset, 0) + 0.25  # 현금(수비 자산) 비중 증가


                return pd.Series(selected)
            
            else : #기존 전략 공격자산중 모멘텀높은자산

                top_aggressive = momentum[self.aggressive_assets].nlargest(4)
                selected = {}
                cash_asset = momentum[self.defensive_assets].idxmax()  # 수비 자산 중 모멘텀이 가장 높은 자산

                for asset, asset_momentum in top_aggressive.items():
                    if asset_momentum > 0:
                        selected[asset] = 0.25  # 25% 비중
                    else:
                        selected[cash_asset] = selected.get(cash_asset, 0) + 0.25  # 현금(수비 자산) 비중 증가

                return pd.Series(selected)


    def backtest(self, start_date, end_date):
        data = self.fetch_data(start_date, end_date)
        portfolio_value = [1.0]  # 포트폴리오 가치 초기화
        last_weights = None
        last_canary_sign = None
        asset_weights = []
        last_year = None
        all_assets = self.all_assets
        total_portfolio_return=0
        returns=0
        portfolio_return = 0


        for i, date in enumerate(data.index):
            current_year = date.year
            if current_year != last_year:
                print(f"현재 처리 중인 연도: {current_year}")
                last_year = current_year

            canary_momentum = self.calculate_momentum(data[self.canary_asset].loc[:date])

            if not canary_momentum.empty:

                canary_momentum = canary_momentum.iloc[-1]
                current_canary_sign = canary_momentum > 0

                rebalance = (last_canary_sign is None or
                             current_canary_sign != last_canary_sign or
                             i < len(data.index) - 1 and date.month != data.index[i + 1].month)
                
                # rebalance = (i < len(data.index) - 1 and date.month != data.index[i + 1].month)#월말 리밸런싱

                last_canary_sign = current_canary_sign
                # print("canary_momentum",canary_momentum)
                if rebalance or last_weights is None:
                    #
                    # print(date)
                    # print("rebalance !")
                    #
                    # print("==============")
                    momentum = pd.Series({asset: self.calculate_momentum(data[asset].loc[:date]).iloc[-1]
                                          for asset in data.columns})
                    selected = self.select_assets(momentum)
                    weights = selected / selected.sum()

                    all_weights = pd.Series(0.0, index=all_assets)
                    all_weights.update(weights)
                    all_weights = all_weights / all_weights.sum()

                    last_weights = all_weights
                else:
                    all_weights = last_weights

                asset_weights.append(all_weights)

                if i > 0:
                    returns = data.loc[date] / data.loc[data.index[i - 1]] - 1
                    portfolio_return = (last_weights * returns).sum()
                    portfolio_value.append(portfolio_value[-1] * (1 + portfolio_return))
                else:
                    portfolio_value.append(portfolio_value[-1])
            else:
                # 모멘텀 계산 결과가 비어 있을 경우의 처리
                portfolio_value.append(portfolio_value[-1])
                asset_weights.append(pd.Series(0.0, index=all_assets))

            # print(date)
            # print("data.loc[date]  :::" + str(date))
            # print("data.loc[data.index[i - 1]]  :::" + str(data.index[i - 1]))
            # print("last_wights")
            # print(last_weights)
            # print(" ")
            # print("returns")
            # print(returns)
            # print(" ")
            # print("portfolio_return")
            # print(portfolio_return)
            # print("total_portfolio_return ::: " + str(total_portfolio_return))
            # print("====================================")

        # 결과 반환 부분
        valid_index = data.index
        print(len(valid_index))
        print(len(portfolio_value))
        print(len(asset_weights))
        print(data.index.is_unique)

        portfolio_value = portfolio_value[1:] #초기설정값제거
        portfolio_series = pd.Series(portfolio_value, index=valid_index)

        # asset_weights의 길이가 portfolio_value와 다를 수 있으므로 조정
        asset_weights_df = pd.DataFrame(asset_weights, index=valid_index)
        # asset_weights_df = asset_weights_df.reindex(valid_index)

        return portfolio_series, asset_weights_df

    def backtest_w(self, start_date, end_date):
        data = self.fetch_data(start_date, end_date)
        portfolio_value = [1.0]  # 포트폴리오 가치 초기화
        last_weights = None
        last_canary_sign = None
        asset_weights = []
        last_year = None
        all_assets = self.all_assets
        total_portfolio_return = 0
        returns = 0
        portfolio_return = 0

        for i, date in enumerate(data.index):
            current_year = date.year
            if current_year != last_year:
                print(f"현재 처리 중인 연도: {current_year}")
                last_year = current_year

            canary_momentum = self.calculate_momentum(data[self.canary_asset].loc[:date])

            if not canary_momentum.empty:

                canary_momentum = canary_momentum.iloc[-1]
                current_canary_sign = canary_momentum > 0

                rebalance = (last_canary_sign is None or
                             current_canary_sign != last_canary_sign or
                             i < len(data.index) - 1 and date.month != data.index[i + 1].month)

                # rebalance = (i < len(data.index) - 1 and date.month != data.index[i + 1].month)#월말 리밸런싱

                last_canary_sign = current_canary_sign
                # print("canary_momentum",canary_momentum)
                if rebalance or last_weights is None:
                    #
                    # print(date)
                    # print("rebalance !")
                    #
                    # print("==============")
                    momentum = {}
                    for asset in data.columns:
                        if asset in self.canary_asset:
                            # 카나리아 자산에 대해서는 calculate_momentum 사용
                            momentum[asset] = self.calculate_momentum(data[asset].loc[:date]).iloc[-1]
                        else:
                            # 그 외 자산에 대해서는 calculate_momentum_w 사용
                            momentum[asset] = self.calculate_momentum_136(data[asset].loc[:date]).iloc[-1]

                    selected = self.select_assets(pd.Series(momentum))
                    weights = selected / selected.sum()

                    all_weights = pd.Series(0.0, index=all_assets)
                    all_weights.update(weights)
                    all_weights = all_weights / all_weights.sum()

                    last_weights = all_weights
                else:
                    all_weights = last_weights

                asset_weights.append(all_weights)

                if i > 0:
                    returns = data.loc[date] / data.loc[data.index[i - 1]] - 1
                    portfolio_return = (last_weights * returns).sum()
                    portfolio_value.append(portfolio_value[-1] * (1 + portfolio_return))
                else:
                    portfolio_value.append(portfolio_value[-1])
            else:
                # 모멘텀 계산 결과가 비어 있을 경우의 처리
                portfolio_value.append(portfolio_value[-1])
                asset_weights.append(pd.Series(0.0, index=all_assets))

        # 결과 반환 부분
        valid_index = data.index
        print(len(valid_index))
        print(len(portfolio_value))
        print(len(asset_weights))
        print(data.index.is_unique)

        portfolio_value = portfolio_value[1:]  # 초기설정값제거
        portfolio_series = pd.Series(portfolio_value, index=valid_index)

        # asset_weights의 길이가 portfolio_value와 다를 수 있으므로 조정
        asset_weights_df = pd.DataFrame(asset_weights, index=valid_index)
        # asset_weights_df = asset_weights_df.reindex(valid_index)

        return portfolio_series, asset_weights_df

    def backtest_136_all(self, start_date, end_date):
        data = self.fetch_data(start_date, end_date)
        portfolio_value = [1.0]  # 포트폴리오 가치 초기화
        last_weights = None
        last_canary_sign = None
        asset_weights = []
        last_year = None
        all_assets = self.all_assets
        total_portfolio_return = 0
        returns = 0
        portfolio_return = 0

        for i, date in enumerate(data.index):
            current_year = date.year
            if current_year != last_year:
                print(f"현재 처리 중인 연도: {current_year}")
                last_year = current_year

            canary_momentum = self.calculate_momentum_136(data[self.canary_asset].loc[:date])

            if not canary_momentum.empty:

                canary_momentum = canary_momentum.iloc[-1]
                current_canary_sign = canary_momentum > 0

                rebalance = (last_canary_sign is None or
                             current_canary_sign != last_canary_sign or
                             i < len(data.index) - 1 and date.month != data.index[i + 1].month)

                if self.last_rebalance_date is None:
                    self.last_rebalance_date = date


                # # Calculate the difference in days
                # days_difference = (date - self.last_rebalance_date).days
                # #
                # # # Check if 14 days have passed for rebalance
                # # rebalance = days_difference >= 14
                #
                # rebalance = (last_canary_sign is None or
                #              current_canary_sign != last_canary_sign or
                #              i < len(data.index) - 1 and days_difference >= 14)


                # rebalance = (i < len(data.index) - 1 and date.month != data.index[i + 1].month)#월말 리밸런싱

                last_canary_sign = current_canary_sign
                # print("canary_momentum",canary_momentum)
                if rebalance or last_weights is None:

                    self.last_rebalance_date = date  # Update last rebalance date

                    # print(date)
                    # print("rebalance !")
                    #
                    # print("==============")
                    momentum = pd.Series({asset: self.calculate_momentum_136(data[asset].loc[:date]).iloc[-1]
                                          for asset in data.columns})
                    selected = self.select_assets(momentum)
                    weights = selected / selected.sum()

                    all_weights = pd.Series(0.0, index=all_assets)
                    all_weights.update(weights)
                    all_weights = all_weights / all_weights.sum()

                    last_weights = all_weights
                else:
                    all_weights = last_weights

                asset_weights.append(all_weights)

                if i > 0:
                    returns = data.loc[date] / data.loc[data.index[i - 1]] - 1
                    portfolio_return = (last_weights * returns).sum()
                    portfolio_value.append(portfolio_value[-1] * (1 + portfolio_return))
                else:
                    portfolio_value.append(portfolio_value[-1])
            else:
                # 모멘텀 계산 결과가 비어 있을 경우의 처리
                portfolio_value.append(portfolio_value[-1])
                asset_weights.append(pd.Series(0.0, index=all_assets))

            # print(date)
            # print("data.loc[date]  :::" + str(date))
            # print("data.loc[data.index[i - 1]]  :::" + str(data.index[i - 1]))
            # print("last_wights")
            # print(last_weights)
            # print(" ")
            # print("returns")
            # print(returns)
            # print(" ")
            # print("portfolio_return")
            # print(portfolio_return)
            # print("total_portfolio_return ::: " + str(total_portfolio_return))
            # print("====================================")

        # 결과 반환 부분
        valid_index = data.index
        print(len(valid_index))
        print(len(portfolio_value))
        print(len(asset_weights))
        print(data.index.is_unique)

        portfolio_value = portfolio_value[1:]  # 초기설정값제거
        portfolio_series = pd.Series(portfolio_value, index=valid_index)

        # asset_weights의 길이가 portfolio_value와 다를 수 있으므로 조정
        asset_weights_df = pd.DataFrame(asset_weights, index=valid_index)
        # asset_weights_df = asset_weights_df.reindex(valid_index)

        return portfolio_series, asset_weights_df




# 결과 분석 및 출력
def analyze_performance(portfolio_value, strategy_name):
    # 전체 수익률 계산
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100

    # CAGR 계산
    # cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (252 / len(portfolio_value)) - 1

    initial_exclusion_period = 180  # 약 1년치의 거래일 수
    cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[initial_exclusion_period]) ** (
                252 / (len(portfolio_value) - initial_exclusion_period)) - 1

    mdd = (portfolio_value / portfolio_value.cummax() - 1).min() * 100

    # total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
    #
    # # CAGR 계산
    # years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    # cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1
    #
    # # MDD 계산
    # mdd = (portfolio_value / portfolio_value.cummax() - 1).min() * 100


    print(f"{strategy_name}:")
    print(f"Total Return: {total_return:.2f}%")
    print(f"CAGR: {cagr * 100:.2f}%")
    print(f"Max Drawdown: {mdd:.2f}%")
    print()

def calculate_yearly_returns(portfolio_value):
    yearly_returns = {}
    for year in range(portfolio_value.index[0].year, portfolio_value.index[-1].year + 1):
        start_value = portfolio_value[portfolio_value.index.year == year].iloc[0]
        end_value = portfolio_value[portfolio_value.index.year == year].iloc[-1]
        yearly_return = (end_value / start_value - 1) * 100
        yearly_returns[year] = yearly_return
    return yearly_returns

#

# 자산 정의
#수비자산 gld제외
original_assets = ['SPY', 'IWM', 'VWO', 'VNQ', 'PDBC', 'VEA', 'TLT', 'IEF']

new_assets_v2 = ['IWM', 'VWO', 'VNQ', 'PDBC', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'VEA', 'TLT', 'IEF']#s&P 섹터 추가


new_assets_v3 = ['IWM',  'VNQ', 'PDBC', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'TLT', 'IEF'] #s&P 섹터 추가, vea, vwo 제외

new_assets_v4 = ['IWM', 'VNQ', 'PDBC', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'GLD',  'VEA', 'VWO', 'TLT', 'IEF']#s&P 섹터 추가, 금추가

new_assets_v5 = ['IWM', 'VNQ', 'PDBC', 'XLE', 'XLV', 'XLF', 'XLC', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'GLD', 'TLT', 'IEF' ]#s&P 섹터 추가, 금추가 'VEA', 'VWO'제외

new_assets_v6 = ['IWM', 'VNQ', 'PDBC', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'GLD', 'QQQ', 'TLT', 'IEF']#s&P 섹터 추가, 금추가 'VEA', 'VWO'제외, qqq추가

new_assets_v7 = ['IWM', 'VNQ', 'PDBC', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'GLD',  'QQQ', 'VEA', 'VWO', 'TLT', 'IEF']#s&P 섹터 추가, 금추가  qqq추가

new_assets_v8 = ['IWM', 'VNQ', 'PDBC', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'QQQ', 'VEA', 'VWO', 'TLT', 'IEF']#s&P 섹터 추가, qqq추가

new_assets_v9 = ['IWM', 'VNQ', 'PDBC', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'QQQ', 'TLT', 'IEF' ]#s&P 섹터 추가,'VEA', 'VWO'제외 qqq추가



#xlc는 데이터땜에 일단 뺌
new_assets_v10 = ['VNQ', 'XLE', 'XLC', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'GLD', 'TLT', 'IEF' ]#s&P 섹터 추가, 금추가 'VEA', 'VWO'제외, pdbc제외 xlb대체 276000.KS
new_assets_v11 = ['VNQ', 'XLE', 'XLV', 'XLF', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'GLD', 'TLT', 'IEF' ]#s&P 섹터 추가, 금추가 'VEA', 'VWO'제외, pdbc제외 xlb대체 276000.KS
new_assets_v12_1 = ['XLE', 'XLV', 'XLC', 'XLF', 'XLI', 'XLB', 'XLK', 'XLP', 'XLY', 'XLU', 'VNQ']
new_assets_v12_2 = ['GLD', 'TLT', 'IEF', 'LQD', 'HYG']
new_assets_v13_2 = ['GLD', 'TLT', 'IEF', 'LQD', 'HYG', 'SCHD']
new_assets_v13 = ['GLD']

new_assets_v14 = ['QQQ']#s&P 섹터 추가
new_assets_v15 = ['SPY']#s&P 섹터 추가
new_assets_v16 = ['TQQQ']#s&P 섹터 추가

#수비자산 gld추가
new_defensive_assets = ['BIL', 'IEF', '261240.KS']#261240.KS 달러선물

new_defensive_assets2 = ['BIL', 'IEF']#114800.KS 코스피 인버스

# HAA 인스턴스 생성, 수비자산 ['BIL', 'IEF']
haa_original = HAA(original_assets)
# haa_v2 = HAA(new_assets_v2)
# haa_v3 = HAA(new_assets_v3)
# haa_v4 = HAA(new_assets_v4)
haa_v5 = HAA(new_assets_v5)
haa_v6 = HAA(new_assets_v6)
haa_v7 = HAA(new_assets_v7)
# haa_v8 = HAA(new_assets_v8)
haa_v10 = HAA(new_assets_v10)
haa_v11 = HAA(new_assets_v13)
# haa_v12 = HAA(aggressive_assets=new_assets_v12_1, aggressive_assets2=new_assets_v12_2)
haa_v13 = HAA(aggressive_assets=new_assets_v12_1, aggressive_assets2=new_assets_v13_2)

haa_v13_just = HAA(new_assets_v10)
haa_v14 = HAA(new_assets_v14)
haa_v15 = HAA(new_assets_v15)
haa_v16 = HAA(new_assets_v16)


# HAA 인스턴스 생성, 수비자산 ['BIL', 'IEF', 'GLD']

haa_original_2 = HAA(original_assets, new_defensive_assets)
# haa_v2_2 = HAA(new_assets_v2, new_defensive_assets)
# haa_v3_2 = HAA(new_assets_v3, new_defensive_assets)
# haa_v4_2 = HAA(new_assets_v4, new_defensive_assets)
# haa_v5_2 = HAA(new_assets_v5, new_defensive_assets)
# haa_v6_2 = HAA(new_assets_v6, new_defensive_assets)
# haa_v7_2 = HAA(new_assets_v7, new_defensive_assets)
#  haa_v8_2 = HAA(new_assets_v8, new_defensive_assets)

# 백테스트 실행 날짜
start_date = '2015-01-01'
end_date = '2025-12-06'


# portfolio_value_v2, weights_v2 = haa_v2.backtest(start_date, end_date)
# portfolio_value_v3, weights_v3 = haa_v3.backtest(start_date, end_date)
# portfolio_value_v4, weights_v4 = haa_v4.backtest(start_date, end_date)
# portfolio_value_v5, weights_v5 = haa_v5.backtest(start_date, end_date)
# portfolio_value_v6, weights_v6 = haa_v6.backtest(start_date, end_date)
# portfolio_value_v7, weights_v7 = haa_v7.backtest(start_date, end_date)
# portfolio_value_v8, weights_v8 = haa_v8.backtest(start_date, end_date)
# portfolio_value_v12, weights_v12 = haa_v12.backtest(start_date, end_date)
# portfolio_value_v10, weights_v10 = haa_v10.backtest(start_date, end_date)
# portfolio_value_v12, weights_v12 = haa_v10.backtest(start_date, end_date)
# portfolio_value_v13, weights_v13 = haa_v13.backtest_136_all(start_date, end_date)
# portfolio_value_v13_W, weights_v13_W = haa_v13_W.backtest_w(start_date, end_date)



# portfolio_value_v13, weights_v13 = haa_v13.backtest_136_all(start_date, end_date)#전체자산 136모멘텀
# annual_returns = calculate_yearly_returns(portfolio_value_v13)

# portfolio_value_v15, weights_v15 = haa_v15.backtest_136_all(start_date, end_date)#전체자산 136모멘텀
# annual_returns = calculate_yearly_returns(portfolio_value_v15)
# print("v15")
# for year, return_value in annual_returns.items():
#     print(f"{year}: {return_value:.2f}%")
# print("")

portfolio_value_v13, weights_v13 = haa_v13_just.backtest_136_all(start_date, end_date)#전체자산 136모멘텀
annual_returns = calculate_yearly_returns(portfolio_value_v13)
print("v13_just")
for year, return_value in annual_returns.items():
    print(f"{year}: {return_value:.2f}%")
print("")

# portfolio_value_v13_w, weights_v13_w = haa_v13_just.backtest_w(start_date, end_date)#전체자산 136모멘텀
# annual_returns = calculate_yearly_returns(portfolio_value_v13_w)
# print("v13_w")
# for year, return_value in annual_returns.items():
#     print(f"{year}: {return_value:.2f}%")
# print("")
#

# portfolio_value_v13_2, weights_v13_2 = haa_v13.backtest_136_all(start_date, end_date)#전체자산 136모멘텀
# annual_returns = calculate_yearly_returns(portfolio_value_v13)
# print("_v13_2")
# for year, return_value in annual_returns.items():
#     print(f"{year}: {return_value:.2f}%")
# print("")



# portfolio_value_v14, weights_v14 = haa_v14.backtest_136_all(start_date, end_date)#전체자산 136모멘텀
# annual_returns = calculate_yearly_returns(portfolio_value_v14)
# print("v14")
# for year, return_value in annual_returns.items():
#     print(f"{year}: {return_value:.2f}%")
# print("")
#
#
# portfolio_value_v16, weights_v16 = haa_v16.backtest_136_all(start_date, end_date)#전체자산 136모멘텀
# annual_returns = calculate_yearly_returns(portfolio_value_v16)
# print("v16")
# for year, return_value in annual_returns.items():
#     print(f"{year}: {return_value:.2f}%")
# print("")


# portfolio_value_v14, weights_v14 = haa_v14.backtest_w(start_date, end_date)#카나리아 자산은 12312모멘텀으로 그외는 136



# portfolio_value_original_2, weights_original_2 = haa_original_2.backtest(start_date, end_date)
# portfolio_value_v2_2, weights_v2_2 = haa_v2_2.backtest(start_date, end_date)
# portfolio_value_v3_2, weights_v3_2 = haa_v3_2.backtest(start_date, end_date)
# portfolio_value_v4_2, weights_v4_2 = haa_v4_2.backtest(start_date, end_date)
# portfolio_value_v5_2, weights_v5_2 = haa_v5_2.backtest(start_date, end_date)
# portfolio_value_v6_2, weights_v6_2 = haa_v6_2.backtest(start_date, end_date)
# portfolio_value_v7_2, weights_v7_2 = haa_v7_2.backtest(start_date, end_date)
# portfolio_value_v8_2, weights_v8_2 = haa_v8_2.backtest(start_date, end_date)

# annual_returns = calculate_yearly_returns(portfolio_value_original)
#
# for year, return_value in annual_returns.items():
#     print(f"{year}: {return_value:.2f}%")

#누적수익률 그래프

# 백테스트 실행 결과 (포트폴리오 가치)
portfolio_values = {
     # 'Original': portfolio_value_original,
    # 'V2': portfolio_value_v2,
    # 'V3': portfolio_value_v3,
    # 'V4': portfolio_value_v4,
    # 'V10': portfolio_value_v10,
    # 'V6': portfolio_value_v6,
    # 'V7': portfolio_value_v7,
    # 'V8': portfolio_value_v8
    # 'V10': portfolio_value_v10,
    # 'V13_w': portfolio_value_v13_W,
    # 'V11': portfolio_value_v11,
    'V13': portfolio_value_v13,
    # 'V13_w': portfolio_value_v13_w,
    # 'V13_2': portfolio_value_v13_2,
    # 'V14': portfolio_value_v14,
    # 'V15': portfolio_value_v15,
    # 'V16': portfolio_value_v16,
    # 'V14' : portfolio_value_v14
    # 'Original_2': portfolio_value_original_2,
    # 'V2_2': portfolio_value_v2,
    # 'V3_2': portfolio_value_v3_2,
    # 'V4_2': portfolio_value_v4_2,
    # 'V5_2': portfolio_value_v5_2,
    # 'V6_2': portfolio_value_v6_2,
    # 'V7_2': portfolio_value_v7_2,
    # 'V8_2': portfolio_value_v8_2
}

# 누적 수익률 계산
cumulative_returns = {}
for version, values in portfolio_values.items():
    cumulative_returns[version] = (values / values.iloc[0]) - 1  # 기준값을 0으로 설정

#결과분석
for version, values in portfolio_values.items():
    analyze_performance(values, version)

# 그래프 그리기
plt.figure(figsize=(12, 8))
for version, returns in cumulative_returns.items():
    plt.plot(returns.index, returns.values, label=version)

plt.title('Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
print("aaaaaaaa")

pd.set_option('display.max_rows', None)  # 모든 행 출력
pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.width', None)  # 너비 제한 해제
pd.set_option('display.max_colwidth', None)  # 열 너비 제한 해제

while 1==1 :
    plt.pause(0.1)
    print("시작날짜 : ")
    startdate_str = input("")
    print("종료날짜 : ")
    try :
        enddate_str = input("")
        start_date = pd.to_datetime(startdate_str).tz_localize('UTC')
        end_date = pd.to_datetime(enddate_str).tz_localize('UTC')

        filtered_weights = weights_v13[(weights_v13.index >= start_date) & (weights_v13.index <= end_date)]
        # filtered_weights_W = weights_v13_W[(weights_v13_W.index >= start_date) & (weights_v13_W.index <= end_date)]
        print("weights_v13 ==================")
        print(filtered_weights)
        print("======================================")
        # filtered_weights = weights_v13_w[(weights_v13_w.index >= start_date) & (weights_v13_w.index <= end_date)]
        # # filtered_weights_W = weights_v13_W[(weights_v13_W.index >= start_date) & (weights_v13_W.index <= end_date)]
        # print("weights_v13_w ==================")
        # print(filtered_weights)
        # print("======================================")

        # filtered_weights = weights_v13_2[(weights_v13_2.index >= start_date) & (weights_v13_2.index <= end_date)]
        # # filtered_weights_W = weights_v13_W[(weights_v13_W.index >= start_date) & (weights_v13_W.index <= end_date)]
        # print("weights_v13_2 ==================")
        # print(filtered_weights)
        # print("======================================")

        # filtered_weights = weights_v14[(weights_v14.index >= start_date) & (weights_v14.index <= end_date)]
        # # filtered_weights_W = weights_v13_W[(weights_v13_W.index >= start_date) & (weights_v13_W.index <= end_date)]
        # print("weights_v14 ==================")
        # print(filtered_weights)
        # print("======================================")
        #
        # filtered_weights = weights_v15[(weights_v15.index >= start_date) & (weights_v15.index <= end_date)]
        # # filtered_weights_W = weights_v13_W[(weights_v13_W.index >= start_date) & (weights_v13_W.index <= end_date)]
        # print("weights_v15 ==================")
        # print(filtered_weights)
        # print("======================================")
        #
        # filtered_weights = weights_v16[(weights_v16.index >= start_date) & (weights_v16.index <= end_date)]
        # # filtered_weights_W = weights_v13_W[(weights_v13_W.index >= start_date) & (weights_v13_W.index <= end_date)]
        # print("weights_v16 ==================")
        # print(filtered_weights)
        # print("======================================")
        # # print(filtered_weights_W)
    except Exception as e :
        print(e)

