from functools import partial
from multiprocessing import Pool
from typing import Union
from bisect import insort
import numpy as np

from pandas import DataFrame, Timestamp, concat
from utils.screenerUtils import make_df


class Backtest:
    """
    An abstract class to create backtests on a single security. Initialized with a pandas DataFrame in yfinance format.
    Backtests can be defined by overriding the long and short triggers; built-in test functions can be used
    to test the defined strategy, or a custom test can be created. Calculates descriptive stats about a backtest,
    including returns, batting average, and descriptions of gains/losses. The default testing method can be specified by
    setting the "run" attribute to the desired method. By default, this is 'backtest_long_only.'

    Args:
        df: The pandas dataframe with data about the equity. Uses a Timestamp index, consistent with the yfinance library dataframes.
    """
    def _trigger_long(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        """
        This method should be overridden to identify that a long position should taken on the given date (e.g. purchase an equity)
        Users can analyze the whole dataframe, but must return "True" to take a long position only on the given date.
        Return "False" or "None" to decline to take a long position.
        --
        ADVANCED USAGE:
        By default, returning "True" will trigger a long position at the 'Adj Close' price. The user may enter the position at
        a different price by returning a numerical value from this method. Caution is advised because this module will not verify
        that it was possible to enter at the given price. 

        Args:
            df: The pandas dataframe with data about the equity. Uses a Timestamp index, consistent with the yfinance library dataframes.
            date: The Timestamp index of the date being evaluated in the dataframe.

        Returns:
            Returns a boolean value, where "True" means to take a long position, and "False" means don't take a long position

        Raises:
            TO DO
        """
        raise NotImplementedError('You must override the "_trigger_long" method')

    def _trigger_short(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        """
        This method should be overridden to identify that a short position should taken on the given date (e.g. purchase an equity)
        Users can analyze the whole dataframe, but must return "True" to take a short position only on the given date.
        Return "False" or "None" to decline to take a short position.
        --
        ADVANCED USAGE:
        By default, returning "True" will trigger a short position at the 'Adj Close' price. The user may enter the position at
        a different price by returning a numerical value from this method. Caution is advised because this module will not verify
        that it was possible to enter at the given price. 

        Args:
            df: The pandas dataframe with data about the equity. Uses a Timestamp index, consistent with the yfinance library dataframes.
            date: The Timestamp index of the date being evaluated in the dataframe.

        Returns:
            Returns a boolean value, where "True" means to take a short position, and "False" means don't take a short position

        Raises:
            TO DO
        """
        raise NotImplementedError('You must override the "_trigger_short" method')

    def _reset(self):
        self._total_return = 1.0
        self._buy_price = None
        self._sell_price = None
        self._gains = []
        self._losses = []

    def _buy(self, price):
        try:
            self._buy_price = float(price)
        except (TypeError, ValueError):
            raise ValueError('Buy price must be a number')

    def _sell(self, price):
        try:
            self._sell_price = float(price)
        except (TypeError, ValueError):
            raise ValueError('Sell price must be a number')

    def _calculate_gain(self, reset_prices=True):
        result = (self._sell_price - self._buy_price) / self._buy_price + 1
        self._total_return *= result

        if result > 1:
            self._gains.append(result)
        elif result < 1:
            self._losses.append(result)

        if reset_prices:
            self._buy_price = None
            self._sell_price = None

    def get_stats(self):
        """
        Returns a dict containing summary statistics about gains that were saved to an instance of a backtest.

        Args:
            N/A

        Returns:
            dict : {'totalReturn': float, 'battingAvg': float, 'largestGain': float, 'largestLoss': float, 'avgGain': float, 'avgLoss': float}
        """
        n_gains = len(self._gains)
        n_losses = len(self._losses)
        total_trades = n_gains + n_losses
        cagr = self._total_return ** (1 / ((self.df.index[-1] - self.df.index[0]).days / 365.25))
        return {
            'totalReturn': self._total_return,
            'cagr': cagr,
            'battingAvg': None if total_trades == 0 else n_gains / total_trades,
            'largestGain': 0 if n_gains == 0 else max(self._gains),
            'largestLoss': 0 if n_losses == 0 else min(self._losses),
            'avgGain': 0 if n_gains == 0 else sum(self._gains) / n_gains,
            'avgLoss': 0 if n_losses == 0 else sum(self._losses) / n_losses,
            'n_gains': n_gains,
            'n_losses': n_losses
        }

    def _backtest_long_only(self):
        """
        This method is used to backtest long-only equity trading strategies. A security is purchased on dates when the 
        '_trigger_long' method returns True (or a non-zero numeric value), and subsequently sold when the '_trigger_short' method
        returns True (or a non-zero numeric value). Note that the trigger methods are called every time the current long position
        makes it possible - that is, when holding a long position, only the '_short_trigger' method is called and vice-versa. 

        Args:
            N/A

        Returns:
            A dict containing summary statistics of the backtest, as defined in the 'get_stats' method

        Raises:
            TO DO
        """
        pos = False
        for date in self.df.index:
            if not pos and (long_trigger := self._trigger_long(self.df, date)):
                pos = True
                self._buy(self.df.loc[date, 'Adj Close'] if type(long_trigger) is bool else long_trigger)
            elif pos and (short_trigger := self._trigger_short(self.df, date)):
                pos = False
                self._sell(self.df.loc[date, 'Adj Close'] if type(short_trigger) is bool else short_trigger)
                self._calculate_gain()
        return self.get_stats()

    def run(self):
        """
        This method should be used to reference the default backtest method.

        Args:
            N/A

        Returns:
            A dict containing summary statistics of the backtest as returned by the designated backtest method
        """
        return self._backtest_long_only()

    def __init__(self, df):
        self.df = df
        self._reset()


def run_backtests(start, end, ticker_list, *backtests):
    """
    Runs backtests on multiple tickers over the specified time period. Tickers should be passed as a list and 
    backtest classes should be passed (not instances). Outputs results to a csv file.

    Args:
        start: A starting date for the period to test over
        ticker_list: A list of ticker strings; often these lists can be made with the 'utils.screenerUtils.load_tickers' function
        *backtests: The backtest classes to be 

    Returns:
        None

    Raises:
        ValueError if no backtests are passed
        TypeError if backtests aren't a subclass of Backtest or if tickers aren't strings
    """
    if len(backtests) == 0:
        raise ValueError('You must pass at least one backtest class')
    if not all([issubclass(bt, Backtest) for bt in backtests]):
        raise TypeError('All backtests must inherit the Backtest class')
    if any([type(t) != str for t in ticker_list]):
        raise TypeError('All tickers must be strings')

    pool = Pool(4)
    f = partial(make_df, start, end)

    results = {
        bt.__name__: {
            'avg_return': 0,
            'batting_avg': None,
            'n_gains': 0,
            'n_losses': 0,
            'avg_gain': 0,
            'avg_loss': 0,
            'largest_gain': 1,
            'largest_loss': 1,
            'top_n': [],
            'bot_n': [],
            'top_n_avg_return': 1,
            'bot_n_avg_return': 1
        }
        for bt in backtests
    }
    n_selected = len(ticker_list) // 5 + 1  # used to select the 20th and 80th percentiles of stocks

    try:
        for t, df in pool.imap_unordered(f, ticker_list):
            for bt in backtests:
                bt_res = results[bt.__name__]
                res = bt(df).run()

                bt_res['avg_return'] += res['totalReturn']
    
                bt_res['n_gains'] += res['n_gains']
                bt_res['n_losses'] += res['n_losses']

                bt_res['avg_gain'] += res['avgGain'] * res['n_gains']
                bt_res['avg_loss'] += res['avgLoss'] * res['n_losses']

                bt_res['largest_gain'] = res['totalReturn'] if res['totalReturn'] > bt_res['largest_gain'] else bt_res['largest_gain']
                bt_res['largest_loss'] = res['totalReturn'] if res['totalReturn'] < bt_res['largest_loss'] else bt_res['largest_loss']

                pair = (res['totalReturn'], t)
                if len(bt_res['top_n']) < n_selected:  # both lists will fill simultaneously since they both start empty
                    insort(bt_res['top_n'], pair)
                    insort(bt_res['bot_n'], pair)
                # once they are both minimally fully, a new incoming result can only possibly fall into one list
                elif res['totalReturn'] > min(bt_res['top_n'])[0]:
                    bt_res['top_n'].pop(0)
                    insort(bt_res['top_n'], pair)
                elif res['totalReturn'] < max(bt_res['bot_n'])[0]:
                    bt_res['bot_n'].pop(-1)
                    insort(bt_res['bot_n'], pair)

    finally:
        for bt, bt_res in results.items():
            n_gains = bt_res['n_gains']
            n_losses = bt_res['n_losses']
            total_trades = n_gains + n_losses

            bt_res['avg_return'] /= total_trades if total_trades != 0 else 1
            bt_res['batting_avg'] = n_gains / total_trades if total_trades != 0 else None
            bt_res['avg_gain'] /= n_gains if n_gains != 0 else 1
            bt_res['avg_loss'] /= n_losses if n_losses != 0 else 1
            bt_res['top_n_avg_return'] = sum([ret[0] for ret in bt_res['top_n']]) / n_selected
            bt_res['bot_n_avg_return'] = sum([ret[0] for ret in bt_res['bot_n']]) / n_selected

            top_n = DataFrame([t[1] for t in reversed(bt_res['top_n'])], columns=['top_n'])
            bot_n = DataFrame([t[1] for t in bt_res['bot_n']], columns=['bot_n'])
            out_df = DataFrame(bt_res).drop(columns=['top_n', 'bot_n']).loc[0]
            out_df = concat([out_df, top_n, bot_n]).replace(np.nan, '')
            out_df.to_csv(f'results/{bt}.csv')
