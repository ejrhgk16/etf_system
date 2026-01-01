import requests
from urllib.request import urlopen
import configparser as parser
import json
import reqLsApi as req
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import math
import time
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
from openpyxl.styles import Font, Alignment
from openpyxl import Workbook
from datetime import date, timedelta

import os
from os import path

if __name__ == '__main__':

    print("경로 ::: " + os.path.dirname(__file__))

    try:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 200)

        # req.setAppKeys()
        # req.getToken()

        # 오늘 날짜 구하기
        today = date.today().strftime("%Y%m%d")


        # 3일 전 날짜 구하기
        three_days_ago = (date.today() - timedelta(days=3)).strftime("%Y%m%d")


        # req.getMasterData()
        # req.getYesterdayData("snp", three_days_ago, today)



    except Exception as e:

        print(f"에러 ::: {e}")
        input("...")