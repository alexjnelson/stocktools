from os.path import dirname
from sys import path

path.append(dirname(dirname(__file__)))

from backtesting.backtest_abstract import Backtest, run_backtests