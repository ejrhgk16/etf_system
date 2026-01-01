import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import os
from dotenv import load_dotenv

class HAA:
    def __init__(self, canary_asset='TIP'):
        # .env 파일 로드
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path=dotenv_path)

        # 투자금 설정
        try:
            self.capital = int(os.getenv('haa_plus_capital', 10000))
        except (ValueError, TypeError):
            print("Warning: .env의 'haa_plus_capital'이 유효하지 않습니다. 기본값 10000을 사용합니다.")
            self.capital = 10000

        # .env에서 자산 목록 로드
        agg_assets_str = os.getenv('HAA_AGGRESSIVE_ASSETS')
        def_assets_str = os.getenv('HAA_DEFENSIVE_ASSETS')

        if not agg_assets_str or not def_assets_str:
            print("Warning: .env 파일에서 자산 목록을 찾을 수 없어 기본 자산을 사용합니다.")
            self.aggressive_assets = ['TQQQ','SPY', 'IWM', 'VEA', 'VWO', 'TLT', 'IEF', 'PDBC', 'VNQ', 'GLD']
            self.defensive_assets = ['BIL', 'IEF', 'GLD']
        else:
            self.aggressive_assets = [asset.strip() for asset in agg_assets_str.split(',')]
            self.defensive_assets = [asset.strip() for asset in def_assets_str.split(',')]

        # 유효성 검사
        if not self.aggressive_assets or not self.defensive_assets:
            raise ValueError("공격적 자산과 방어적 자산 목록은 비어 있을 수 없습니다.")

        self.canary_asset = canary_asset
        self.aggressive_assets_named = []
        self.defensive_assets_named = []

    def fetch_data(self, start_date, end_date):
        def download_and_clean(tickers, prefix=''):
            raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, back_adjust=True)
            if raw.empty or 'Close' not in raw.columns:
                return pd.DataFrame()
            data = raw['Close']
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
            print("오류: 모멘텀 계산을 위한 과거 데이터를 가져오는 데 실패했습니다.")
            return pd.DataFrame(), pd.Series(dtype=float)

        self.aggressive_assets_named = list(a_data.columns)
        self.defensive_assets_named = list(d_data.columns)
        
        all_data = pd.concat([a_data, d_data, c_data], axis=1).ffill()
        
        all_tickers = self.aggressive_assets + self.defensive_assets + [self.canary_asset]
        latest_prices_data = yf.download(all_tickers, period="2d", auto_adjust=True, back_adjust=True)
        # print(latest_prices_data.head())
        if latest_prices_data.empty or 'Close' not in latest_prices_data.columns:
            print("오류: 수량 계산을 위한 최신 가격 정보를 가져오는 데 실패했습니다.")
            return all_data.dropna(), pd.Series(dtype=float)

        latest_prices = latest_prices_data['Close'].iloc[-1]
        
        return all_data.dropna(), latest_prices

    def calculate_momentum_batch(self, data):
        r1 = data.pct_change(21)
        r3 = data.pct_change(63)
        r6 = data.pct_change(126)
        momentum_df = (r1 + r3 + r6) / 3
        return momentum_df

    def select_assets(self, momentum_row):
        canary_mom = momentum_row.get(self.canary_asset, -np.inf)
        
        defensive_momentums = momentum_row[self.defensive_assets_named]
        best_defensive_asset = defensive_momentums.idxmax() if not defensive_momentums.empty and not defensive_momentums.isna().all() else f"d_{self.defensive_assets[0]}"

        if canary_mom <= 0:
            return {best_defensive_asset: 1.0}
        else:
            positive_agg_mom = momentum_row[self.aggressive_assets_named][momentum_row[self.aggressive_assets_named] > 0]
            
            if positive_agg_mom.empty:
                return {best_defensive_asset: 1.0}

            top_agg = positive_agg_mom.nlargest(4)
            
            selected = {}
            num_selected = len(top_agg)
            if num_selected > 0:
                weight_per_asset = 1.0 / num_selected
                for asset in top_agg.index:
                    selected[asset] = weight_per_asset
            else:
                 return {best_defensive_asset: 1.0}

            return selected

    def get_today_portfolio(self):
        end_date = datetime.date.today()
        start_date = end_date - pd.DateOffset(months=8)

        data, latest_prices = self.fetch_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if data.empty or latest_prices.empty or latest_prices.isna().all():
            print("데이터 로드 실패로 포트폴리오 계산을 중단합니다.")
            return

        momentum_all = self.calculate_momentum_batch(data)
        
        if len(momentum_all) < 2:
            print("Error: 유효한 모멘텀 데이터가 부족하여 포트폴리오를 계산할 수 없습니다.")
            return
            
        latest_momentum = momentum_all.iloc[-2]

        if latest_momentum.isna().any():
            print("Error: 모멘텀 데이터에 NaN 값이 포함되어 포트폴리오를 계산할 수 없습니다.")
            return

        holdings = self.select_assets(latest_momentum)

        total_spent = 0
        
        print(f"--- {end_date} 기준 HAA+ 포트폴리오 (총 투자금: ${self.capital:,.0f}) ---")
        if not holdings:
            print("보유할 자산이 없습니다. 전액 현금 보유.")
        else:
            print("현재 보유해야 할 자산:")
            
            asset_details = []
            for asset, weight in holdings.items():
                clean_asset = asset.replace('a_', '').replace('d_', '')
                asset_price = latest_prices.get(clean_asset)
                
                if asset_price is None or pd.isna(asset_price) or asset_price <= 0:
                    print(f"- {clean_asset}: {weight*100:.2f}% (오류: 가격 정보 없음)")
                    continue

                investment_amount = self.capital * weight
                quantity_int = int(investment_amount / asset_price)
                actual_cost = quantity_int * asset_price
                
                if quantity_int > 0:
                    asset_details.append({
                        'clean_asset': clean_asset,
                        'weight': weight,
                        'asset_price': asset_price,
                        'quantity_int': quantity_int,
                        'actual_cost': actual_cost
                    })
                    total_spent += actual_cost

            if not asset_details:
                 print("자본금 부족 또는 가격 정보 오류로 매수 가능한 자산이 없습니다.")
            else:
                for detail in asset_details:
                    print(f"- {detail['clean_asset']}:")
                    print(f"  - 목표 비중: {detail['weight']*100:.2f}%")
                    print(f"  - 매수 수량: {detail['quantity_int']} 주")
                    print(f"  - 예상 매수 금액: ${detail['actual_cost']:,.2f}")

            remaining_cash = self.capital - total_spent
            print("\n--- 요약 ---")
            print(f"총 사용 금액 (예상): ${total_spent:,.2f}")
            print(f"남는 현금: ${remaining_cash:,.2f}")

if __name__ == "__main__":
    try:
        # HAA 클래스 인스턴스 생성
        strategy = HAA()
        
        # 포트폴리오 계산 및 출력 실행
        strategy.get_today_portfolio()
    except ValueError as e:
        # HAA 클래스 생성 시 발생할 수 있는 오류 처리
        print(f"Error: {e}")
