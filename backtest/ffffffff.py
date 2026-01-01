import requests
import tradingeconomics as te
te.login("317c124f8804470:dlh2q1ri4058ulh")
gold_futures = te.fetchMarkets(symbol=['GC1:COM'], initDate='2024-01-01', endDate='2025-01-01')

print(gold_futures)