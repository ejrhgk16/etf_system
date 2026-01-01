import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


class HAA:
    def __init__(self, aggressive_assets, defensive_assets=['BIL', 'IEF'], aggressive_assets2=[]):
        self.aggressive_assets = aggressive_assets
        self.aggressive_assets2 = aggressive_assets2
        self.defensive_assets = defensive_assets
        self.canary_asset = 'TIP'
        self.all_assets = []
        self.aggressive_assets_named = []
        self.defensive_assets_named = []

    def fetch_data(self, start_date, end_date):
        def download_and_clean(tickers, prefix=''):
            raw = yf.download(tickers, start=start_date, end=end_date)
            if raw.empty: return pd.DataFrame()
            data = raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close']
            if isinstance(data, pd.Series):
                ticker_name = tickers if isinstance(tickers, str) else tickers[0]
                data = data.to_frame(name=ticker_name)
            if prefix:
                data.columns = [prefix + col for col in data.columns]
            return data

        a_data = download_and_clean(self.aggressive_assets, 'a_')
        d_data = download_and_clean(self.defensive_assets, 'd_')
        c_data = download_and_clean(self.canary_asset)

        if a_data.empty or d_data.empty or c_data.empty:
            print("데이터를 가져오는 데 실패했습니다.")
            return pd.DataFrame()

        self.aggressive_assets_named = list(a_data.columns)
        self.defensive_assets_named = list(d_data.columns)
        self.all_assets = self.aggressive_assets_named + self.defensive_assets_named + ['CASH']
        return pd.concat([a_data, d_data, c_data], axis=1).ffill().dropna()

    def calculate_momentum_batch(self, data):
        r1 = data.pct_change(21)
        r3 = data.pct_change(63)
        r6 = data.pct_change(126)
        momentum_df = (r1 + r3 + r6) / 3
        return momentum_df

    def select_assets(self, momentum_row):
        canary_mom = momentum_row.get(self.canary_asset, -1)
        cash_asset = momentum_row[self.defensive_assets_named].idxmax()
        if canary_mom <= 0:
            return {cash_asset: 1.0}
        else:
            top_agg = momentum_row[self.aggressive_assets_named].nlargest(4)
            selected = {}
            for asset, mom in top_agg.items():
                if mom > 0:
                    selected[asset] = 0.25
                else:
                    selected[cash_asset] = selected.get(cash_asset, 0) + 0.25
            return selected

    def backtest(self, start_date, end_date, initial_investment=10000):
        data = self.fetch_data(start_date, end_date)
        if data.empty: return None, None
        momentum_all = self.calculate_momentum_batch(data)
        daily_returns = data.pct_change().fillna(0)
        portfolio_values = [initial_investment]
        weights_history = []
        current_weights = pd.Series(0.0, index=self.all_assets)
        last_month = -1

        for i in range(len(data)):
            current_date = data.index[i]
            if current_date.month != last_month:
                if i > 0:
                    mom_row = momentum_all.iloc[i - 1]
                    if not mom_row.isna().any():
                        selected_dict = self.select_assets(mom_row)
                        new_weights = pd.Series(0.0, index=self.all_assets)
                        for asset, w in selected_dict.items():
                            new_weights[asset] = w
                        current_weights = new_weights
                last_month = current_date.month

            if i > 0:
                day_ret = 0.0
                for asset, weight in current_weights.items():
                    if weight <= 0: continue
                    if asset in daily_returns.columns:
                        day_ret += weight * daily_returns.iloc[i][asset]
                portfolio_values.append(portfolio_values[-1] * (1 + day_ret))
            weights_history.append(current_weights.copy())

        return pd.Series(portfolio_values, index=data.index), pd.DataFrame(weights_history, index=data.index)


# --- [수정 포인트] 시각화 및 수익률 표시 강화 ---
def analyze(portfolio_series, initial_investment):
    if portfolio_series is None: return

    final_value = portfolio_series.iloc[-1]
    cum_ret = (final_value / initial_investment - 1) * 100

    # MDD 계산
    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    mdd = drawdown.min() * 100

    plt.figure(figsize=(12, 7))
    plt.plot(portfolio_series, label='HAA Strategy', color='#1f77b4', linewidth=2)

    # 원금 표시선
    plt.axhline(y=initial_investment, color='red', linestyle='--', alpha=0.5)

    # 차트 설정
    plt.title('HAA Strategy Backtest (Portfolio Value)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Value ($)')

    # Y축 천 단위 콤마(,) 표시
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # 결과 요약 박스 추가
    stats_text = (f'Initial: ${initial_investment:,.0f}\n'
                  f'Final: ${final_value:,.0f}\n'
                  f'Return: {cum_ret:.2f}%\n'
                  f'MDD: {mdd:.2f}%')

    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.show()


# 자산 리스트
agg = ['TQQQ','SPY', 'IWM', 'VEA', 'VWO', 'TLT', 'IEF', 'DBC', 'VNQ', 'GLD']
defen = ['BIL', 'IEF', 'GLD']

# 실행
initial_money = 10000  # 초기 자금 설정
strategy = HAA(agg, defen)
p_val, p_weights = strategy.backtest('2015-01-01', '2025-12-30', initial_money)

# 결과 분석 (수정된 함수 호출)
analyze(p_val, initial_money)