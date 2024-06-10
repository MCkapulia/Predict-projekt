from datetime import datetime, timedelta
from tinkoff.invest.services import InstrumentsService, MarketDataService
from pandas import DataFrame
from tinkoff.invest.utils import quotation_to_decimal

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

def quote_to_float(x) -> float:
    return float(quotation_to_decimal(x))

def candles():


    interval = CandleInterval.CANDLE_INTERVAL_DAY

    end = pd.Timestamp.utcnow()
    begin = end - pd.offsets.DateOffset(years=4)

    #print(f'Load history for figi={getFigi()} in the period=({begin}, {end})')
    with Client(TOKEN) as client:
        def generator():
            for candle in client.get_all_candles(figi=getFigi(), from_=begin, to=end, interval=interval):
                yield {
                    'open': quote_to_float(candle.open),
                    'high': quote_to_float(candle.high),
                    'low': quote_to_float(candle.low),
                    'close': quote_to_float(candle.close),
                    'volume': candle.volume,
                    'time': candle.time,
                    'is_complete': candle.is_complete,
                }

        df = pd.DataFrame(generator())
        df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S.%f%z")
        df['time'] = df['time'].dt.strftime("%d-%m-%Y %H:%M:%S")
        df.to_excel("data.xlsx")
    return df

