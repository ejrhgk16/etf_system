from trading_system.haa_plus import HAA
from trading_system.spx_vix import SPX_VIX

if __name__ == "__main__":
    try:
        # HAA+ 전략 실행
        print("--- HAA+ Strategy Execution ---")
        strategy_haa = HAA()
        strategy_haa.get_today_portfolio()
        print("-" * 40)

        # SPX/VIX 전략 실행
        print("\n--- SPX/VIX Strategy Execution ---")
        strategy_vix = SPX_VIX()
        strategy_vix.get_todays_action()
        print("-" * 40)

    except Exception as e:
        # 오류 처리
        print(f"An error occurred: {e}")
