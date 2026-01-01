import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
from dotenv import load_dotenv
import math
import datetime

class SPX_VIX:
    def __init__(self, rsi_len=2, commission_rate=0.001):
        # .env 파일에서 설정값 로드
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path=dotenv_path)
        try:
            self.capital = int(os.getenv('SPX_VIX_CAPITAL', 10000))
            self.current_svxy_shares = int(os.getenv('CURRENT_SVXY_SHARES', 0))
            self.current_vixm_shares = int(os.getenv('CURRENT_VIXM_SHARES', 0))
        except (ValueError, TypeError) as e:
            print(f"Error: .env 파일의 설정값을 읽는 중 오류 발생 - {e}")
            self.capital, self.current_svxy_shares, self.current_vixm_shares = 10000, 0, 0

        self.rsi_len = rsi_len
        self.commission_rate = commission_rate
        self.all_tickers = ["^SPX", "^VIX", "^VIX3M", "SVXY", "VIXM"]

    def fetch_data(self, start_date, end_date):
        def download_and_clean(tickers):
            raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, back_adjust=True, progress=True)
            if raw.empty: return pd.DataFrame()
            data = raw.get('Close')
            if data is None: return pd.DataFrame()
            if isinstance(data, pd.Series):
                ticker_name = tickers if isinstance(tickers, str) else tickers[0]
                data = data.to_frame(name=ticker_name)
            return data

        historical_data = download_and_clean(self.all_tickers)
        if historical_data.empty:
            raise ConnectionError("지표 계산을 위한 과거 데이터를 가져오는 데 실패했습니다.")

        historical_data.columns = [col.replace('^', '') for col in historical_data.columns]
        all_data = historical_data.ffill()
        
        latest_prices_data = yf.download(self.all_tickers, period="2d", auto_adjust=True, back_adjust=True, progress=True)
        if latest_prices_data.empty or 'Close' not in latest_prices_data.columns:
            raise ConnectionError("수량 계산을 위한 최신 가격 정보를 가져오는 데 실패했습니다.")

        latest_prices = latest_prices_data['Close'].iloc[-1]
        latest_prices.index = [idx.replace('^', '') for idx in latest_prices.index]
        
        return all_data.dropna(), latest_prices

    def add_indicators(self, df):
        if 'SPX' not in df.columns or df['SPX'].isnull().all(): return df
        rsi_col_name = f'RSI_{self.rsi_len}'
        df[rsi_col_name] = ta.rsi(df['SPX'], length=self.rsi_len)
        overbought = (df[rsi_col_name] - 90) / 10
        oversold = (df[rsi_col_name] - 30) / 30
        df['RSI_Custom'] = np.select(
            [df[rsi_col_name] >= 90, df[rsi_col_name] <= 30],
            [overbought, oversold], default=0
        )
        return df

    def get_todays_action(self):
        try:
            end_date = datetime.date.today()
            start_date = end_date - pd.DateOffset(days=90)
            df, latest_prices = self.fetch_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if df.empty or latest_prices.empty: raise ValueError("데이터 다운로드 실패")
            df = self.add_indicators(df).dropna(subset=['RSI_Custom', 'VIX', 'VIX3M'])
        except Exception as e:
            return print(f"FATAL: 데이터 처리 중 오류 발생 - {e}")

        if df.empty: return print("분석을 위한 최종 데이터가 없습니다.")
        
        latest_signal_data = df.iloc[-1]
        latest_svxy_price = latest_prices.get('SVXY', 0); latest_vixm_price = latest_prices.get('VIXM', 0)
        if pd.isna(latest_svxy_price): latest_svxy_price = 0
        if pd.isna(latest_vixm_price): latest_vixm_price = 0

        current_svxy_value = self.current_svxy_shares * latest_svxy_price
        current_vixm_value = self.current_vixm_shares * latest_vixm_price
        cash = self.capital - (current_svxy_value + current_vixm_value)

        # print("--- 현재 포트폴리오 상태 ---")
        print(f"총 자본: ${self.capital:,.2f}")
        print(f"보유 SVXY: {self.current_svxy_shares}주 (현재가: ${latest_svxy_price:,.2f}, 평가액: ${current_svxy_value:,.2f})")
        print(f"보유 VIXM: {self.current_vixm_shares}주 (현재가: ${latest_vixm_price:,.2f}, 평가액: ${current_vixm_value:,.2f})")
        print(f"추정 보유 현금: ${cash:,.2f}")
        print("-" * 25)
        
        # --- 신호 정의 ---
        is_contango = latest_signal_data.get('VIX', 0) < latest_signal_data.get('VIX3M', 0)
        rsi_custom = latest_signal_data.get('RSI_Custom', 0)
        rsi_plain = latest_signal_data.get(f'RSI_{self.rsi_len}', 50)
        current_asset = 'SVXY' if self.current_svxy_shares > 0 else 'VIXM' if self.current_vixm_shares > 0 else None

        print("--- 시장 신호 ---")
        print(f"Contango = {is_contango}, RSI({self.rsi_len}) = {rsi_plain:.2f}, RSI_Custom = {rsi_custom:.4f}")

        # --- 1. 명시적 청산 신호 확인 --- 
        if current_asset:
            exit_reason = ""
            if not is_contango: 
                exit_reason = "Contango 조건 소멸"
            elif current_asset == 'VIXM' and rsi_plain <= 50:
                exit_reason = f"VIXM 청산 신호 (RSI <= 50)"
            elif current_asset == 'SVXY' and rsi_plain >= 50:
                exit_reason = f"SVXY 청산 신호 (RSI >= 50)"
            
            if exit_reason:
                print(f"\n[ACTION] 전량 매도: {current_asset}을(를) 모두 매도하여 현금 보유로 전환하세요.")
                print(f"  - 이유: {exit_reason}")
                current_asset = None 

        # --- 2. 신규 진입 및 수량 조절 --- 
        target_asset = None
        if is_contango:
            if rsi_custom > 0: target_asset = 'VIXM'
            elif rsi_custom < 0: target_asset = 'SVXY'

        if current_asset is None:
            if target_asset:
                target_price = latest_prices.get(target_asset, 0)
                if pd.isna(target_price) or target_price <= 0: return print(f"\n[Error] 신규 매수 불가: {target_asset} 가격 정보 없음")
                target_shares = math.floor((self.capital * abs(rsi_custom)) / target_price)
                cost = target_shares * target_price * (1 + self.commission_rate)

                print(f"\n[ACTION] 신규 매수: {target_asset} {target_shares}주 매수를 고려하세요.")
                print(f"  - 이유: 신규 투자 신호 발생 (RSI_Custom = {rsi_custom:.4f}) ")
                print(f"  - 예상 비용: ${cost:,.2f} (수수료 포함)")
            else:
                print("\n[ACTION] 현금 보유: 유효한 투자 신호가 없습니다.")
        elif current_asset == target_asset:
            target_price = latest_prices.get(target_asset, 0)
            if pd.isna(target_price) or target_price <= 0: return print(f"\n[Error] 수량 조절 불가: {target_asset} 가격 정보 없음")
            
            target_shares = math.floor((self.capital * abs(rsi_custom)) / target_price)
            current_shares = self.current_svxy_shares if target_asset == 'SVXY' else self.current_vixm_shares
            share_diff = target_shares - current_shares

            if share_diff > 0:
                cost = share_diff * target_price * (1 + self.commission_rate)
                print(f"\n[ACTION] 추가 매수: {target_asset} {share_diff}주 추가 매수를 고려하세요.")
                print(f"  - 예상 비용: ${cost:,.2f} (수수료 포함)")
            elif share_diff < 0:
                sale_amount = abs(share_diff) * target_price * (1 - self.commission_rate)
                print(f"\n[ACTION] 부분 매도: {target_asset} {abs(share_diff)}주 매도를 고려하세요.")
                print(f"  - 예상 매도 금액: ${sale_amount:,.2f} (수수료 차감 후)")
            else:
                print("\n[ACTION] 포지션 유지: 현재 보유 수량이 적정합니다.")
        else:
             print(f"\n[ACTION] 자산 교체: {current_asset}을(를) 전량 매도하세요.")

if __name__ == "__main__":
    strategy = SPX_VIX()
    strategy.get_todays_action()
