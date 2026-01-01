import requests
import configparser as parser
import json
import os
import time

appKey = ''
appSecret = ''
access_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6ImQ5ZTY0OTIyLTlkMmUtNDRhMC1hMmRiLWFiNDZjNTJkN2U1NyIsIm5iZiI6MTcxNDc1MDk0MCwiZ3JhbnRfdHlwZSI6IkNsaWVudCIsImlzcyI6InVub2d3IiwiZXhwIjoxNzE0ODU5OTk5LCJpYXQiOjE3MTQ3NTA5NDAsImp0aSI6IlBTQ1Jwa2xEekJmcUxKTU9ZdTJYZFYwajBkMVVHSTRaVTN2RSJ9.pl16x80iXrGX37NFkUhJrk3AXKCyprz7SPez7GncV3E3DjH5qzx5Pa-ijcSheEqnEqfoU0mOgrx9NXMvjzukqQ'

actNum = ''
actCode = ''
htsId = ''

URL_BASE = "https://openapi.ls-sec.co.kr:8080"

gold_cd = ''
nasdaq_cd = ''
snp_cd = ''

def setAppKeys() :
    global appKey
    global appSecret

    global actNum
    global actCode
    global htsId

    global gold_cd
    global nasdaq_cd
    global snp_cd

    # 현재 실행 파일의 위치
    base_path = os.path.dirname(__file__)

    # data_folder 안의 data_file.txt 파일 경로
    appKey_file_path = os.path.join(base_path, 'info', 'appKey.ini')

    properties = parser.ConfigParser()
    properties.read(appKey_file_path)

    appKey = properties['ebset']['appkey']
    appSecret = properties['ebset']['appSecret']

    act_file_path = os.path.join(base_path, 'info', 'act.ini')
    properties.read(act_file_path)

    actNum = properties['act']['actNum']
    actCode = properties['act']['actCode']
    htsId = properties['act']['htsId']

    product_file_path = os.path.join(base_path, 'info', 'product.ini')
    properties.read(product_file_path)

    gold_cd = properties['gold']['code']
    nasdaq_cd = properties['nasdaq']['code']
    snp_cd = properties['snp']['code']

def getToken():
    print('getToken ::: ')

    global access_token

    headers = {"content-type":  "application/x-www-form-urlencoded"}
    params = {
            "appkey": appKey,
            "appsecretkey": appSecret,
            "grant_type": "client_credentials",
            "scope": "oob"
              }

    PATH = "oauth2/token"

    URL = f"{URL_BASE}/{PATH}"

    res = requests.post(URL, params=params, headers=headers)
    print(res)

    access_token = res.json()["access_token"]

    print(res.text)
    print(res.json())


def getMasterData():



    headers = {
                "content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {access_token}",
                "tr_cd": "o3121",
                "tr_cont": "Y",
                "tr_cont_key" : "0",
    }

    body = {
        "o3121InBlock": {
            "MktGb" : "F",
            "BscGdsCd" : ""

        }
    }

    PATH = "overseas-futureoption/market-data"

    URL = f"{URL_BASE}/{PATH}"

    res = requests.post(URL, headers=headers, data=json.dumps(body))
    result = res.json()

    for key, value in res.headers.items():
        print(f"{key}: {value}")

    print(result)

    headers2 = {
                "content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {access_token}",
                "tr_cd": "o3121",
                "tr_cont": "Y",
                "tr_cont_key": "1",
    }

    body2 = {
        "o3121InBlock": {
            "MktGb" : "F",
            "BscGdsCd" : ""

        }
    }

    PATH = "overseas-futureoption/market-data"

    URL = f"{URL_BASE}/{PATH}"

    print(headers2)

    time.sleep(1)

    res2 = requests.post(URL, headers=headers2, data=json.dumps(body2))

    print(res2)

    result2 = res2.json()

    for key, value in res2.headers.items():
        print(f"{key}: {value}")

    print(result2)

    return result


def getYesterdayData(pdName, sdate, edate):
    print('getInquireInvestorEbest ::: ' + pdName)

    shcode = ''
    if pdName == "gold" :
        shcode = gold_cd

    if pdName == "snp" :
        shcode = snp_cd

    if pdName == "nasdaq" :
        shcode = nasdaq_cd

    headers = {
                "content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {access_token}",
                "tr_cd": "o3128",
                "tr_cont": "N"
    }

    body = {
        "o3128InBlock": {
            "mktgb" : "F",
            "shcode": shcode,
            "gubun" : "0",
            "qrycnt":  5,
            "sdate": sdate, #20230525
            "edate": edate, #20230609
            "cts_date": "",
        }
    }

    print(body)



    PATH = "overseas-futureoption/market-data"

    URL = f"{URL_BASE}/{PATH}"

    res = requests.post(URL, headers=headers, data=json.dumps(body))
    result = res.json()

    for key, value in res.headers.items():
        print(f"{key}: {value}")

    print(result)

    return result