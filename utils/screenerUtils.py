import datetime as dt
import os
import signal
from time import sleep
from urllib.error import HTTPError, URLError

import pandas as pd
import yfinance as yf
from requests.exceptions import ConnectionError
from urllib3.exceptions import ProtocolError

from utils import addIndicators as add


def load_tickers(filename):
    return [t.replace('\n', '').lower() for t in open(filename)]


def save_local_df(start, end, t, path='../local_dfs'):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

    try:
        df = download(lambda: yf.download(
            str(t), start, end, progress=False))
    except (ValueError, IndexError, KeyError):
        df = None

    if df is None:
        return
    df = df[~df.index.duplicated(keep='first')]

    if len(df) == 0:
        return
    df.to_csv(f'{path}/{t}')


def local_df(start, end, t, path='../local_dfs'):
    df = pd.read_csv(f'{path}/{t}', index_col=0, parse_dates=True)
    start = df.index[0] if start is None else start
    end = df.index[-1] if end is None else end
    df = df.loc[start:end, :]

    try:
        df = addIndicators(df)
        if df is not None and len(df) != 0:
            return t, df
    except ValueError:
        print(f'\nInsuficient data on local machine to make indicators for {t}\n')
        return


def download(f: callable, retries_allowed=20):
    def raise_alarm(*args, **kwargs):
        raise TimeoutError

    retries = 0
    while True:
        if retries == retries_allowed:
            print('Data download failed.')
            return
        try:
            signal.signal(signal.SIGALRM, raise_alarm)
            signal.alarm(30)
            return f()
        except (HTTPError, ProtocolError, ConnectionError, URLError):
            sleep(10)
            print('Retrying data download...')
            retries += 1
        except (ValueError, IndexError, KeyError, TimeoutError):
            return
        finally:
            signal.alarm(0)


def handle_dict(d: dict, k):
    try:
        return d[k]
    except (KeyError, TypeError):
        return None


def addIndicators(df):
    df = add.shiftedSqueeze(df)
    df = add.Bollinger(df, period=20)
    df = add.Keltner(df, emaPeriod=20, atrPeriod=20)
    df = add.RSI(df)
    df = add.ATR(df)

    df = add.shortOverLong(df, shortEmas=[12], longEmas=[26])
    df = add.klinger(df, with_temp=True)
    df = add.MACD(df, withHistogram=True,
                    withTriggers=False, withTrends=True)
    df = add.failureSwings(df)
    df = add.observeTrend(df, 'Adj Close', trendlines=True,
                            colName='Price Uptrend?', lineColName='Sup/Res')
    df = add.EMA(df, period=15)

    df = df[~df.isna().any(1)]
    return df


def make_df(start, end, t, try_local=False, **kwargs):
    download_start = start - dt.timedelta(days=180)

    if try_local:
        try:
            df = local_df(download_start, end, t)
            return df
        except FileNotFoundError:
            try_local = False

    if not try_local:
        try:
            df = download(lambda: yf.download(
                str(t), download_start, end, progress=False, **kwargs))
        except (ValueError, IndexError, KeyError):
            df = None

        if df is None:
            return
        df = df[~df.index.duplicated(keep='first')]

    try:
        df = addIndicators(df)
        df = df.loc[pd.Timestamp(start):pd.Timestamp(end), :]
        if df is not None and len(df) != 0:
            return t, df
    except (ValueError, IndexError, KeyError):
        print(f'\nInsuficient data to make indicators for {t}\n')
        return
