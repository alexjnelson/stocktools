if __name__ == '__main__':
    from os.path import dirname
    from sys import path
    path.append(dirname(dirname(__file__)))

import datetime as dt
from typing import Union

from pandas import DataFrame, Timestamp, concat

from utils.addIndicators import hammer
from utils.screenerUtils import make_df
from backtesting import Backtest


class OversoldHammer(Backtest):
    """
    buy when the price dropped below the longer bollinger band and a hammer is made; 
    sell when the price reaches the top 10% of the area between bollinger bands
    """
    def __init__(self, df):
        self.df = hammer(df, trend_days=1)
        self._reset()

    def _reset(self):
        super()._reset()
        self._stoploss = 0

    def _trigger_long(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        if (c := df.index.get_loc(date)) == 0:
            return
        two_days_ago = df.index[0] if c < 2 else df.index[c - 2]
        yesterday = df.index[c - 1]

        # first check if there is a hammer candlestick, and then check if the security is oversold based on the bol bands and kelt channels
        if df.loc[yesterday, 'Hammer'] and df.loc[date, 'Close'] >= df.loc[yesterday, 'Close']:
            mins = concat([df.loc[two_days_ago:date, 'Bol Lower'], df.loc[two_days_ago:date, 'Kelt Lower']], axis=1).min(axis=1)
            if any(df.loc[two_days_ago:date, 'Close'] <= mins[two_days_ago:date]):
                self._stoploss = df.loc[yesterday, 'Low']
                return df.loc[date, 'Close']

    def _trigger_short(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        trigger_price = (df.loc[date, 'Bol Upper'] - df.loc[date, 'Bol Lower']) * 0.9 + df.loc[date, 'Bol Lower']
        if df.loc[date, 'Close'] <= self._stoploss or df.loc[date, 'High'] >= trigger_price:
            return trigger_price if df.loc[date, 'High'] >= trigger_price else df.loc[date, 'Close']


class OversoldHammerRideMACD(OversoldHammer):
    """
    buy when the price dropped below the longer bollinger band and a hammer is made; 
    sell when the price reaches the top 10% of the area between bollinger bands, but wait for macd death cross
    """
    def _reset(self):
        super()._reset()
        self._triggered = False

    def _trigger_long(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        return True if super()._trigger_long(df, date) else False

    def _trigger_short(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        trigger_price = (df.loc[date, 'Bol Upper'] - df.loc[date, 'Bol Lower']) * 0.9 + df.loc[date, 'Bol Lower']
        self._triggered = self._triggered or df.loc[date, 'High'] >= trigger_price

        if df.loc[date, 'Close'] <= self._stoploss or self._triggered and not df.loc[date, 'MACD Uptrend?']:
            self._triggered = False
            return True
        else:
            return False


class Hold(Backtest):
    def _trigger_long(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        return True
    
    def _trigger_short(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        return date == df.index[-1]

if __name__ == '__main__':
    tests = [OversoldHammer, OversoldHammerRideMACD, Hold]
    start = dt.date(2016, 1, 1)
    end = dt.date.today()

    # df = make_df(start, end, '^gsptse')[1]
    df = make_df(dt.date(2020, 11, 16), end, 'txp.to', interval='15m')[1]

    for t in tests:
        test = t(df)
        print (f'--\n{test.__class__.__name__}')
        print(test.backtest_long_only())
