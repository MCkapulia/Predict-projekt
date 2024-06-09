from datetime import datetime, timedelta
from tinkoff.invest.services import InstrumentsService, MarketDataService
from pandas import DataFrame

from tinkoff.invest import Client, RequestError, CandleInterval, HistoricCandle, InstrumentStatus
from dotenv import load_dotenv
import os

import pandas as pd

load_dotenv()
TOKEN = os.getenv('TOKEN')
TICKER = "ROSN"

def getFigi():
    with Client(TOKEN) as client:
        instruments: InstrumentsService = client.instruments
        market_data: MarketDataService = client.market_data
        r = DataFrame(instruments.shares(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE).instruments,
                      columns=['name', 'figi', 'ticker', 'class_code'])
        r = r[r['ticker'] == TICKER]['figi'].iloc[0]
        return r

def run():
    try:
        with Client(TOKEN) as client:
            r = client.market_data.get_candles(
                figi=getFigi(),
                from_=datetime.now() - timedelta(days=144),
                to=datetime.now(),
                interval=CandleInterval.CANDLE_INTERVAL_DAY
            )

            df = create_df(r.candles)
            df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S.%f%z")
            df['time'] = df['time'].dt.strftime("%d-%m-%Y %H:%M:%S")
            df.to_excel("data.xlsx")


    except RequestError as e:
        print(str(e))


def create_df(candles: [HistoricCandle]):
    df = pd.DataFrame([{
        'time': c.time,
        'volume': c.volume,
        'open': cast_money(c.open),
        'close': cast_money(c.close),
        'high': cast_money(c.high),
        'low': cast_money(c.low),
    } for c in candles])

    return df

def currPrice():
    with Client(TOKEN) as client:
        _list = list()
        _list.append(getFigi())
        response = client.market_data.get_last_prices(figi=_list)
        word = str(response)
        word1 = word.split("units=")
        word2 = word1[1].split(",")
        word3 = word2[1].split(" nano=")
        word4 = word3[1].split(")")
        return float(word2[0] + "." + word4[0])
def cast_money(v):
    return v.units + v.nano / 1e9  # nano - 9 нулей
