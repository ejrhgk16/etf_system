import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 골드 선물과 TIP ETF 데이터 가져오기

# 오일 CL=F, 스닥 NQ=F, 골드 GC=F
gold_futures = yf.Ticker("GC=F")
tip_etf = yf.Ticker("TIP")

# 데이터 기간 설정
start_date = "2023-06-01"
end_date = "2025-01-05"

# 골드 선물 데이터 가져오기
gold_data = gold_futures.history(start=start_date, end=end_date)


# TIP ETF 데이터 가져오기
tip_data = tip_etf.history(start=start_date, end=end_date)

# TIP ETF 100일 이동평균 계산
tip_data['MA100'] = tip_data['Close'].rolling(window=100).mean()

# 일일 수익률 계산
tip_data['Returns'] = tip_data['Close'].pct_change()

# 1개월, 3개월, 6개월 수익률 계산
tip_data['Returns_1M'] = (1 + tip_data['Returns']).rolling(window=21).apply(lambda x: np.prod(x) - 1)
tip_data['Returns_3M'] = (1 + tip_data['Returns']).rolling(window=63).apply(lambda x: np.prod(x) - 1)
tip_data['Returns_6M'] = (1 + tip_data['Returns']).rolling(window=126).apply(lambda x: np.prod(x) - 1)

# 136 모멘텀 (1개월, 3개월, 6개월 수익률의 평균)
tip_data['136_momentum'] = (tip_data['Returns_1M'] + tip_data['Returns_3M'] + tip_data['Returns_6M']) / 3





# 백테스팅 함수
def backtest(gold_data, tip_data):
    positions = []
    returns = []
    current_position = 0
    entry_price = 0

    #tip 조건없이
    positions2 = []
    returns2 = []
    current_position2 = 0
    entry_price2 = 0

    position_data = []

    long_returns = []
    short_returns = []
    long_returns2 = []
    short_returns2 = []


    for i in range(130, len(gold_data)):

        date = gold_data.index[i]

        if current_position == 0:  # 현재 포지션이 없는 경우
            if tip_data['Close'].iloc[i-1] > tip_data['MA100'].iloc[i-1]:
                target_price = gold_data['Open'].iloc[i] + abs(gold_data['Close'].iloc[i-1] - gold_data['Open'].iloc[i-1])*0.5
                if gold_data['High'].iloc[i] >= target_price:  # 당일 고가가 목표가 이상일 때 진입
                    entry_price = target_price
                    current_position = 3

            elif tip_data['Close'].iloc[i-1] < tip_data['MA100'].iloc[i-1]:
                target_price = gold_data['Open'].iloc[i] - abs(gold_data['Close'].iloc[i-1] - gold_data['Open'].iloc[i-1])*0.5
                if gold_data['Low'].iloc[i] <= target_price:  # 당일 저가가 목표가 이하일 때 진입
                    entry_price = target_price
                    current_position = -3

            #tip 조건없이
        if current_position2 == 0:

            target_price2_L = gold_data['Open'].iloc[i] + abs(gold_data['Close'].iloc[i-1] - gold_data['Open'].iloc[i-1])*0.5
            target_price2_S = gold_data['Open'].iloc[i] - abs(gold_data['Close'].iloc[i - 1] - gold_data['Open'].iloc[i - 1]) * 0.5

            if gold_data['High'].iloc[i] >= target_price2_L:  # 당일 고가가 목표가 이상일 때 진입
                entry_price2 = target_price2_L
                current_position2 = 3
            elif gold_data['Low'].iloc[i] <= target_price2_S:  # 당일 저가가 목표가 이하일 때 진입
                entry_price2 = target_price2_S
                current_position2 = -3

        positions.append(current_position)
        positions2.append(current_position2)

        if current_position != 0:  # 포지션이 있는 경우
            exit_price = gold_data['Close'].iloc[i]

            position_data.append({
                'Date': date,
                'Condition': 'With TIP',
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Position': current_position,
                'Returns': (exit_price - entry_price) / entry_price * current_position
            })


            if current_position == 1:  # 롱 포지션
                long_returns.append((exit_price - entry_price) / entry_price)
            elif current_position == -1:  # 숏 포지션
                short_returns.append((entry_price - exit_price) / entry_price)
            returns.append((exit_price - entry_price) / entry_price * current_position)
            current_position = 0  # 포지션 청산

        else:
            returns.append(0)

        # tip 조건없이
        if current_position2 != 0:  # 포지션이 있는 경우
            exit_price2 = gold_data['Close'].iloc[i]

            position_data.append({
                'Date': date,
                'Condition': 'Without TIP',
                'Entry Price': entry_price2,
                'Exit Price': exit_price2,
                'Position': current_position2,
                'Returns': (exit_price2 - entry_price2) / entry_price2 * current_position2
            })


            if current_position2 == 1:  # 롱 포지션 (TIP 조건 없이)
                long_returns2.append((exit_price2 - entry_price2) / entry_price2)
            elif current_position2 == -1:  # 숏 포지션 (TIP 조건 없이)
                short_returns2.append((entry_price2 - exit_price2) / entry_price2)

            returns2.append((exit_price2 - entry_price2) / entry_price2 * current_position2)
            current_position2 = 0  # 포지션 청산


        else:
            returns2.append(0)





    return pd.Series(returns), pd.Series(positions), pd.Series(returns2), pd.Series(positions2), pd.Series(long_returns), pd.Series(short_returns), pd.Series(long_returns2), pd.Series(short_returns2), pd.DataFrame(position_data)




# 백테스팅 실행
returns, positions, returns2, positions2, long_returns, short_returns, long_returns2, short_returns2, position_data = backtest(gold_data, tip_data)

# 누적 수익률 계산 - 복리계산방식
# cumulative_returns = (1 + returns).cumprod() - 1
#단리계산방식

cumulative_returns = returns.cumsum()

cumulative_returns2 = returns2.cumsum()

# 최대 낙폭(MDD) 계산
def calculate_mdd(returns):
    cumulative = 1 + returns.cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


mdd = calculate_mdd(returns)
mdd2 = calculate_mdd(returns2)


# 결과 출력
print("팁 조건 있이 : ")
print(f"총 수익률: {cumulative_returns.iloc[-1]:.2%}")
print(f"최대 낙폭 (MDD): {mdd:.2%}")
print("총 거래횟수 : " + str((positions != 0).sum()))
print(f"롱 포지션 총 수익률: {long_returns.sum():.2%}")
print(f"숏 포지션 총 수익률: {short_returns.sum():.2%}")

print("::::::::::::::::::::::::::::::::::::")

print("팁 조건 없이 : ")
print(f"총 수익률: {cumulative_returns2.iloc[-1]:.2%}")
print(f"최대 낙폭 (MDD): {mdd2:.2%}")
print("총 거래횟수 : " + str((positions2 != 0).sum()))
print(f"롱 포지션 총 수익률: {long_returns2.sum():.2%}")
print(f"숏 포지션 총 수익률: {short_returns2.sum():.2%}")

# 수익률 추이 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(gold_data.index[130:], cumulative_returns * 100, label='With TIP Condition')
plt.plot(gold_data.index[130:], cumulative_returns2 * 100, label='Without TIP Condition')

plt.title('Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(True)

# X축 날짜 형식 지정
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))  # 30일 간격으로 표시

# X축 레이블 회전
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()

pd.set_option('display.max_rows', None)  # 모든 행 출력
pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.width', None)  # 너비 제한 해제
pd.set_option('display.max_colwidth', None)  # 열 너비 제한 해제

gold_data_df = pd.DataFrame(gold_data)
print(gold_data_df)

# 포지션 정보 출력 함수
def get_position_info(start_date, end_date):

    filtered_data_gold = gold_data_df.loc[start_date:end_date]
    if filtered_data_gold.empty:
        print("해당 날짜 범위에 filtered_data_gold 데이터가 없습니다.")
    else:
        print("골드 원 데이터 ::: ")
        print(filtered_data_gold)
        print(":::::::::::::::")

    filtered_data = position_data[(position_data['Date'] >= start_date) & (position_data['Date'] <= end_date)]
    if filtered_data.empty:
        print("해당 날짜 범위에 포지션 데이터가 없습니다.")
    else:
        for _, row in filtered_data.iterrows():
            print(f"Date: {row['Date'].date()}, {row['Condition']} - Entry Price: {row['Entry Price']:.2f}, "
                  f"Exit Price: {row['Exit Price']:.2f}, Position: {row['Position']}, "
                  f"수익률: {row['Returns']:.6f}")



while 1==1 :
    plt.pause(0.1)
    print("시작날짜 (YYYY-MM-DD): ")
    startdate_str = input("")
    print("종료날짜 (YYYY-MM-DD): ")

    try:
        enddate_str = input("")
        start_date = pd.to_datetime(startdate_str).tz_localize('UTC')
        end_date = pd.to_datetime(enddate_str).tz_localize('UTC')
        get_position_info(startdate_str, enddate_str)


    except Exception as e:
        print(f"오류 발생: {e}")
        print("올바른 날짜 형식(YYYY-MM-DD)으로 입력해주세요.")






