import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 설정 및 데이터 로드
# ==========================================
period_start = "2023-04-22"
period_end = "2025-12-28"
capital_per_trade = 10000  # 회당 투입 원금
vix_lower_bound = 10
vix_upper_bound = 15
trading_fee = 0.0015  # 수수료 0.15%

ticker_list = ["SPY", "^VIX", "^VIX1D", "^VIX3M", "SVXY", "VIXY", "SVIX"]

print(f"데이터 다운로드 중...")
raw_data = yf.download(ticker_list, start=period_start, end=period_end, auto_adjust=True, progress=False)
data = raw_data["Close"].dropna()

# ==========================================
# 2. 수익률 및 매매 신호 계산
# ==========================================
for ticker in ticker_list:
    data[f"{ticker}_ret"] = data[ticker].pct_change()
    data[f"{ticker}_prior"] = data[ticker].shift(1)

data = data.dropna()

# 포지션 신호 (1: 보유, 0: 현금)
data["sig_vixy"] = np.where((data["^VIX1D_prior"] <= vix_lower_bound) & (data["^VIX_prior"] < data["^VIX3M_prior"]), 1, 0)
data["sig_svix"] = np.where((data["^VIX1D_prior"] >= vix_upper_bound) & (data["^VIX_prior"] < data["^VIX3M_prior"]), 1, 0)

# 현재 어떤 종목을 보유 중인지 표시 (0: 현금, 1: VIXY, 2: SVIX)
data["current_pos"] = 0
data.loc[data["sig_vixy"] == 1, "current_pos"] = 1
data.loc[data["sig_svix"] == 1, "current_pos"] = 2

# 매매 발생 여부 확인 (어제와 오늘의 포지션이 다르면 매매 발생)
data["trades"] = np.where(data["current_pos"] != data["current_pos"].shift(1), 1, 0)
# 첫 날 신호가 있으면 매매로 간주
data.iloc[0, data.columns.get_loc("trades")] = 1 if data.iloc[0]["current_pos"] != 0 else 0

# ==========================================
# 3. 수수료를 포함한 PnL 계산
# ==========================================

# 1) 순수 매매 손익 (수수료 제외)
data["raw_pnl"] = (data["sig_vixy"] * data["VIXY_ret"] * capital_per_trade) + \
                  (data["sig_svix"] * data["SVIX_ret"] * capital_per_trade)

# 2) 수수료 계산 (매매 발생 시 원금의 0.15% 차감)
data["fee_cost"] = data["trades"] * capital_per_trade * trading_fee

# 3) 최종 일일 손익 (손익 - 수수료)
data["strat_daily_pnl"] = data["raw_pnl"] - data["fee_cost"]

# 4) 누적 자산 (단리)
data["strat_ideal_equity"] = capital_per_trade + data["strat_daily_pnl"].cumsum()

# 비교군 (Buy & Hold는 시작 시 1회 매수 수수료만 적용)
for ticker in ["VIXY", "SVIX", "SPY"]:
    buy_hold_ret = data[f"{ticker}_ret"].cumsum()
    data[f"{ticker}_equity"] = capital_per_trade * (1 + buy_hold_ret) - (capital_per_trade * trading_fee)

# ==========================================
# 4. 결과 저장 및 통계 출력
# ==========================================
plt.figure(figsize=(12, 7), dpi=150)
final_list = ["strat_ideal", "VIXY", "SVIX", "SPY"]

for ticker in final_list:
    plt.plot(data.index, data[f"{ticker}_equity"], label=ticker, linewidth=2 if ticker=="strat_ideal" else 1)

plt.title(f"Strategic PnL with 0.15% Fee (Simple Interest)\nDaily Capital: ${capital_per_trade:,}")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("backtest19.png")
plt.close()

print("\n" + "="*60)
print(f"수수료(0.15%) 반영 백테스트 결과 (단리 모델)")
print(f"매매 시 투입 원금: ${capital_per_trade:,}")
print("="*60)

for ticker in final_list:
    final_equity = data[f"{ticker}_equity"].iloc[-1]
    net_pnl = final_equity - capital_per_trade
    ret_pct = (net_pnl / capital_per_trade) * 100
    
    label = "IDEAL_STRAT" if ticker == "strat_ideal" else ticker
    print(f"[{label:<11}] 최종 자산: ${final_equity:>10,.2f} | 순손익: ${net_pnl:>9,.2f} | 수익률: {ret_pct:>7.2f}%")

total_trades = data["trades"].sum()
print("-" * 60)
print(f"총 매매 횟수: {int(total_trades)}회")
print(f"총 발생 수수료: ${data['fee_cost'].sum():,.2f}")
print("="*60)
print("차트가 'backtest19.png'로 저장되었습니다.")