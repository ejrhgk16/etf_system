import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def clean_column(column):
    if column.dtype == 'object' and column.name != 'Date' and column.name != 'Vol.' and column.name != 'Change %':
        return column.str.replace(',', '').astype(float)
    return column
pd.options.mode.chained_assignment = None
#tip 2003년 12월 8일 부터 데이터 존재
# 데이터 기간 설정
start_date = "2015-01-01"
end_date = "2025-01-05"

# 골드 선물 데이터 가져오기
df_1 = pd.read_csv("../history_data/Gold Futures Historical Data (1).csv")
df_2 = pd.read_csv("../history_data/Gold Futures Historical Data (2).csv")
gold_df = pd.concat([df_2, df_1])
gold_df['Date'] = pd.to_datetime(gold_df['Date'], format='%m/%d/%Y')
gold_df.set_index('Date', inplace=True)
gold_df = gold_df.sort_values(by='Date', ascending=True)
gold_df = gold_df.apply(clean_column)
gold_df['MA100'] = gold_df['Price'].rolling(window=100).mean()

gold_data = gold_df[(gold_df.index >= start_date) & (gold_df.index <= end_date)]

print(gold_df.columns)
print(gold_df.index)

# 나스닥 선물 데이터 가져오기
nq_df_1 = pd.read_csv("../history_data/Nasdaq 100 Futures Historical Data.csv")
nq_df_2 = pd.read_csv("../history_data/Nasdaq 100 Futures Historical Data (3).csv")
nasdaq_df = pd.concat([nq_df_2, nq_df_1])
nasdaq_df['Date'] = pd.to_datetime(nasdaq_df['Date'], format='%m/%d/%Y')
nasdaq_df.set_index('Date', inplace=True)
nasdaq_df = nasdaq_df.sort_values(by='Date', ascending=True)
nasdaq_df = nasdaq_df.apply(clean_column)
nasdaq_df['MA100'] = nasdaq_df['Price'].rolling(window=100).mean()

nasdaq_data = nasdaq_df[(nasdaq_df.index >= start_date) & (nasdaq_df.index <= end_date)]


#snp 선물 데이터 가져오기
sp_df_1 = pd.read_csv("../history_data/S&P 500 Futures Historical Data.csv")
sp_df_2 = pd.read_csv("../history_data/S&P 500 Futures Historical Data (1).csv")
sp_df = pd.concat([sp_df_2, sp_df_1])
sp_df['Date'] = pd.to_datetime(sp_df['Date'], format='%m/%d/%Y')
sp_df.set_index('Date', inplace=True)
sp_df = sp_df.sort_values(by='Date', ascending=True)
sp_df = sp_df.apply(clean_column)
sp_df['MA100'] = sp_df['Price'].rolling(window=100).mean()

snp_data = sp_df[(sp_df.index >= start_date) & (sp_df.index <= end_date)]


#tip etf 데이터 가져오기
tip_df_1 = pd.read_csv("../history_data/TIP ETF Stock Price History.csv")
tip_df_2 = pd.read_csv("../history_data/TIP ETF Stock Price History (1).csv")
tip_df = pd.concat([tip_df_2, tip_df_1])
tip_df['Date'] = pd.to_datetime(tip_df['Date'], format='%m/%d/%Y')
tip_df.set_index('Date', inplace=True)
tip_df = tip_df.sort_values(by='Date', ascending=True)
tip_df = tip_df.apply(clean_column)
tip_df['MA100'] = tip_df['Price'].rolling(window=100).mean()

tip_data = tip_df[(tip_df.index >= start_date) & (tip_df.index <= end_date)]
print(tip_data.columns)
print(tip_data.index)


# 1개월, 3개월, 6개월 수익률 계산
tip_data.loc[:,'1M_Return'] = tip_data['Price'].pct_change(periods=21)
tip_data.loc[:,'3M_Return'] = tip_data['Price'].pct_change(periods=63)
tip_data.loc[:,'6M_Return'] = tip_data['Price'].pct_change(periods=126)

# 1개월, 3개월, 6개월 수익률의 평균 계산
tip_data['136momentum'] = (tip_data['1M_Return'] + tip_data['3M_Return'] + tip_data['6M_Return']) / 3


def backtest(info, product_data, tip_data, condition_type):
    # 상수 정의
    TICK_SIZE = info['tick_size']
    TICK_VALUE = info['tick_value']
    INITIAL_MARGIN = info['initial_margin']
    initial_capital = info['initial_capital']

    CONTRACT_SIZE = 10  # 계약 단위 (100 oz)
    COMMISSION_PER_CONTRACT = 2  # 한 계약당 수수료 (달러)

    positions = []
    net_pnls = []  # net_pnl을 저장할 리스트
    current_position = 0


    entry_price = 0
    exit_price = 0

    capital = initial_capital

    print(tip_data)



    for i, date in enumerate(product_data.index):
        if i < 130:  # 첫 번째 인덱스 건너뛰기
            continue

        previous_date = product_data.index[i - 1]
        nearest_date = tip_data.index.asof(previous_date)

        # if pd.isna(tip_data.loc[nearest_date, '136momentum']) or pd.isna(tip_data.loc[nearest_date, 'MA100']) or pd.isna(product_data.loc[previous_date, 'MA100']):
        #     continue

        if condition_type == "tip_100":

            long_condition = tip_data.loc[nearest_date, 'Price'] >= tip_data.loc[nearest_date, 'MA100']
            short_condition = tip_data.loc[nearest_date, 'Price'] < tip_data.loc[nearest_date, 'MA100']

        if condition_type == "tip_136momentum":

            long_condition = 0 >= tip_data.loc[nearest_date, '136momentum']
            short_condition = 0 < tip_data.loc[nearest_date, '136momentum']

        if condition_type == "product_100":

            long_condition = product_data.loc[nearest_date, 'MA100'] >= product_data.loc[nearest_date, 'MA100']
            short_condition = product_data.loc[nearest_date, 'MA100'] < product_data.loc[nearest_date, 'MA100']

            print(long_condition)

        x = abs(product_data.loc[previous_date, 'High'] - product_data.loc[previous_date, 'Low'])*0.5
        # if(x >5):
        #     x = 5

        if current_position == 0:
            if long_condition:

                target_price = product_data.loc[date, 'Open'] + x

                if product_data.loc[date, 'High'] >= target_price:
                    if capital >= INITIAL_MARGIN + COMMISSION_PER_CONTRACT:
                        entry_price = target_price
                        current_position = 3
                        capital -= INITIAL_MARGIN + COMMISSION_PER_CONTRACT
                    else:
                        print(f"Warning: Not enough capital for margin and commission on {date}")
                        return

                if product_data.loc[date, 'Low'] < product_data.loc[date, 'Open'] - x and product_data.loc[date, 'Open'] > product_data.loc[date, 'Price']:   # stop 터진걸로 처리
                    exit_price = product_data.loc[date, 'Open'] - x
                else:
                    exit_price = product_data.loc[date, 'Price']

            elif short_condition:
                target_price = product_data.loc[date, 'Open'] - x

                if product_data.loc[date, 'Low'] <= target_price :
                    if capital >= INITIAL_MARGIN + COMMISSION_PER_CONTRACT:
                        entry_price = target_price
                        current_position = -3
                        capital -= INITIAL_MARGIN + COMMISSION_PER_CONTRACT
                    else:
                        print(f"Warning: Not enough capital for margin and commission on {date}")
                        return

                if product_data.loc[date, 'High'] > product_data.loc[date, 'Open'] + x and product_data.loc[date, 'Open'] < product_data.loc[date, 'Price']:  # stop 터진걸로 처리
                    exit_price = product_data.loc[date, 'Open'] + x
                else:
                    exit_price = product_data.loc[date, 'Price']

        if current_position != 0:
            price_change_in_ticks = (exit_price - entry_price) / TICK_SIZE
            pnl = price_change_in_ticks * TICK_VALUE * current_position
            total_commission = COMMISSION_PER_CONTRACT  # 진입 및 청산 수수료
            net_pnl = pnl - total_commission
            print(current_position, date, net_pnl, sum(net_pnls), x, entry_price, exit_price)

            positions.append(current_position)

            capital += INITIAL_MARGIN + net_pnl
            current_position = 0  # 포지션 청산
        else:
            positions.append(current_position)
            net_pnl = 0  # 포지션이 없는 경우 pnl을 0으로 설정

        net_pnls.append(net_pnl)  # net_pnl 저장


    total_net_pnl = sum(net_pnls)
    total_return = (capital - initial_capital) / initial_capital

    return pd.Series(net_pnls), pd.Series(positions), capital, total_return

    # TICK_SIZE = 0.10  # 최소 가격 변동 단위
    # TICK_VALUE = 1  # 틱당 가치
    # INITIAL_MARGIN = 1265  # 초기 증거금 (달러)
    # initial_capital = 10000  # 초기 자본금 설정
gold_info = {'tick_size' : 0.10, 'tick_value':1, 'initial_margin':1265, 'initial_capital':10000}
nasdaq_info = {'tick_size' : 0.25 , 'tick_value': 0.5, 'initial_margin':2600, 'initial_capital':10000}
snp_info = {'tick_size' : 0.25 , 'tick_value': 1.25, 'initial_margin':1600, 'initial_capital':10000}
# 백테스팅 실행
net_pnls_gold, positions_gold, final_capital_gold, total_return_gold = backtest(gold_info, gold_data, tip_data, 'tip_100')

# 결과 출력
print(f"초기 자본: ${10000:.2f}")
print(f"최종 자본: ${final_capital_gold:.2f}")
print(f"총 수익률: {total_return_gold:.2%}")
print(f"총 거래횟수: {(positions_gold != 0).sum()}")

# 누적 수익률 계산
cumulative_returns_gold = net_pnls_gold.cumsum() / 10000


# 최대 낙폭(MDD) 계산
def calculate_mdd(returns):
    cumulative = 1 + returns.cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


mdd_gold = calculate_mdd(net_pnls_gold / 10000)
print(f"최대 낙폭 (MDD): {mdd_gold:.2%}")

print(len(gold_data.index[130:]))
print(len(cumulative_returns_gold))

# 수익률 그래프 그리기
plt.figure(figsize=(12, 6))

plt.plot(gold_data.index, cumulative_returns_gold * 100, label='Cumulative Returns')
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(True)

# X축 날짜 형식 지정
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))  # 30일 간격으로 표시

# X축 레이블 회전
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()
