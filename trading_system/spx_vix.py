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
        # .env 파일 로드
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path=dotenv_path)
        try:
            self.capital = int(os.getenv('SPX_VIX_CAPITAL', 10000))
            self.current_svxy_shares = int(os.getenv('CURRENT_SVXY_SHARES', 0))
            self.current_vixm_shares = int(os.getenv('CURRENT_VIXM_SHARES', 0))
        except (ValueError, TypeError):
            self.capital, self.current_svxy_shares, self.current_vixm_shares = 10000, 0, 0

        self.rsi_len = rsi_len 
        self.commission_rate = commission_rate
        self.all_tickers = ["^SPX", "^VIX9D", "^VIX3M", "SVXY", "VIXM"]

    def fetch_data(self, start_date, end_date):
        def download_and_clean(tickers):
            raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if raw.empty: return pd.DataFrame()
            data = raw.get('Close')
            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
            return data

        historical_data = download_and_clean(self.all_tickers)
        if historical_data.empty:
            raise ConnectionError("과거 데이터를 가져오는 데 실패했습니다.")

        historical_data.columns = [col.replace('^', '') for col in historical_data.columns]
        all_data = historical_data.ffill()

        latest_prices_data = yf.download(self.all_tickers, period="7d", auto_adjust=True, progress=False)
        latest_prices_close = latest_prices_data['Close'].dropna()
        latest_prices = latest_prices_close.iloc[-1]
        latest_prices.index = [idx.replace('^', '') for idx in latest_prices.index]

        return all_data.dropna(), latest_prices

    def add_indicators(self, df):
        rsi_col = f'RSI_{self.rsi_len}'
        df[rsi_col] = ta.rsi(df['SPX'], length=self.rsi_len)

        # 진입 강도 계산 (진입시에만 사용)
        overbought = (df[rsi_col] - 90) / 10
        oversold = (df[rsi_col] - 30) / 30
        df['RSI_Custom'] = np.select(
            [df[rsi_col] >= 90, df[rsi_col] <= 30],
            [overbought, oversold], default=0
        )
        return df

    def get_todays_action(self):
        try:
            end_date = datetime.date.today()
            start_date = end_date - pd.DateOffset(days=90)
            df, latest_prices = self.fetch_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            df = self.add_indicators(df).dropna(subset=[f'RSI_{self.rsi_len}', 'VIX9D', 'VIX3M'])
        except Exception as e:
            return print(f"FATAL: 데이터 처리 오류 - {e}")

        latest_signal_data = df.iloc[-1]
        latest_svxy_price = latest_prices.get('SVXY', 0)
        latest_vixm_price = latest_prices.get('VIXM', 0)

        # 현재 상태 파악
        current_svxy_value = self.current_svxy_shares * latest_svxy_price
        current_vixm_value = self.current_vixm_shares * latest_vixm_price
        current_asset = 'SVXY' if self.current_svxy_shares > 0 else 'VIXM' if self.current_vixm_shares > 0 else None

        # 신호 변수
        is_contango = latest_signal_data['VIX9D'] < latest_signal_data['VIX3M']
        rsi_plain = latest_signal_data[f'RSI_{self.rsi_len}']
        rsi_custom = latest_signal_data['RSI_Custom']

        #print(f"--- 포트폴리오 ($ {self.capital:,.0f}) ---")
        print(f"보유: {current_asset if current_asset else '현금'}")
        print(f"시장: {'Contango' if is_contango else 'Backwardation'}, RSI: {rsi_plain:.2f}")
        print("-" * 30)

        # --- 핵심 로직: 타겟 자산 결정 ---
        target_asset = None
        exit_reason = ""

        if is_contango:
            # 1. 유지 조건 확인 (이미 보유 중인 경우)
            if current_asset == 'SVXY' and rsi_plain < 50:
                target_asset = 'SVXY'
            elif current_asset == 'VIXM' and rsi_plain > 50:
                target_asset = 'VIXM'

            # 2. 신규 진입 조건 확인 (보유 자산이 없거나 청산된 경우)
            if target_asset is None:
                if rsi_custom < 0:  # RSI 30 이하
                    target_asset = 'SVXY'
                elif rsi_custom > 0:  # RSI 90 이상
                    target_asset = 'VIXM'
        else:
            exit_reason = "콘탱고 종료(Backwardation 진입)"

        # --- 액션 출력 ---
        # A. 매도(Exit) 상황
        if current_asset and (target_asset is None):
            if not exit_reason:
                exit_reason = f"RSI 50 도달 (현재 {rsi_plain:.2f})"
            print(f"[ACTION] 전량 매도: {current_asset}을(를) 매도하고 현금화하세요.")
            print(f"이유: {exit_reason}")

        # B. 신규 매수 상황
        elif target_asset and (current_asset is None):
            price = latest_prices.get(target_asset)
            # 신규 진입 시에는 rsi_custom 비중을 사용, 유지 중 rsi_custom이 0이면 최소 비중(예: 0.1) 적용 가능
            weight = abs(rsi_custom) if abs(rsi_custom) > 0 else 0.1
            shares = math.floor((self.capital * weight) / price)
            print(f"[ACTION] 신규 매수: {target_asset} {shares}주")
            print(f"이유: 변동성 신호 발생 (RSI_Custom: {rsi_custom:.4f})")

        # C. 유지 상황
        elif current_asset == target_asset:
            print(f"[ACTION] 포지션 유지: {current_asset}을(를) 계속 보유하세요.")
            print(f"상태: RSI가 50에 도달할 때까지 대기 중 (현재 {rsi_plain:.2f})")

        # D. 자산 교체 (매우 드문 경우)
        elif current_asset and target_asset and current_asset != target_asset:
            print(f"[ACTION] 자산 교체: {current_asset} 전량 매도 후 {target_asset} 신규 매수")

        # E. 관망
        else:
            print("[ACTION] 현금 관망: 진입 신호가 없습니다.")