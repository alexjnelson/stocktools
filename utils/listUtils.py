from functools import partial
from multiprocessing import Pool
from utils.screenerUtils import download, handle_dict, load_tickers, make_df
import yfinance as yf
from utils.addIndicators import RSI
import datetime as dt
from bisect import insort
import re


def makeTickerFile(inf, outf, ext=''):
    """
    Formats a file of tickers for use in this module. Input file must be formatted such that
    the tickers are the first characters on each row. If the tickers require an extension (e.g. TSE stocks
    must end in '.to'), the extension can be passed to this function.


    Args:
        inf: string or path referencing the input file
        outf: string or path referencing the output file
        ext: the extension to be appended to each ticker

    Returns:
        None

    Raises:
        TO DO
    """
    with open(inf) as file:
        tickers = set()
        for row in file:
            tickers.add(re.match(r'[\w\.]+', row).group(0).replace('.', '-'))

    with open(outf, 'w+') as file:
        for t in tickers:
            file.write(f'{t}.{ext}\n')


def removeBadTickers(inf='lists/tsx.txt', outf=None, start=None, end=None):
    """
    Certain tickers may not have enough data for the desired period, so this function can be used
    to filter them out.


    Args:
        inf: string or path referencing the input file
        outf: string or path referencing the output file
        start: the datetime date at the start of the desired period (inclusive)
        end: the datetime date at the end of the desired period (inclusive)

    Returns:
        None

    Raises:
        TO DO
    """
    tickers = load_tickers(inf)
    start = dt.date(2018, 1, 1) if start is None else start
    end = dt.date.today() if end is None else end
    inf_no_ext = inf.split('.')[0]
    outf = f'{inf_no_ext}_cleaned.txt' if outf is None else outf
    updated = []

    p = Pool(4)
    f = partial(make_df, start, end)

    for df in p.imap_unordered(f, tickers):
        if df is not None and len(df[1]) > 0:
            updated.append(df[0])

    with open(outf, 'w+') as file:
        for t in updated:
            file.write(t + '\n')


def filterMarketCap(mincap=1e10, tickerfile=None, savefile=None):
    tickers = load_tickers('lists/tsx_cleaned.txt' if tickerfile is None else tickerfile)
    tickerfile_no_ext = tickerfile.split('.')[0]
    savefile = f'{tickerfile_no_ext}_{mincap}.txt' if savefile is None else savefile

    for i, t in enumerate(tickers):
        print(f'{i} stocks ran through market cap filter', end='\r')
        info = download(lambda: yf.Ticker(t).info)
        mcap = handle_dict(info, 'marketCap')
        if mcap is None or mcap < mincap:
            tickers.remove(t)
    print(f'Finished filtering by market cap. {len(tickers)} stocks remain.')

    with open(savefile, 'w+') as file:
        for t in tickers:
            file.write(t + '\n')


def filterVolatility(n_selected=10, avg_days=5, min_vol=45000, max_rsi=60, min_rsi=40, tickerfile=None, savefile=None):
    tickers = load_tickers('lists/tsx_cleaned.txt' if tickerfile is None else tickerfile)
    savefile = f'{tickerfile}_volatile.txt' if savefile is None else savefile

    start = dt.date.today() - dt.timedelta(days=60)
    end = dt.date.today()

    top_n = []

    for t in tickers:
        try:
            df = download(lambda: yf.download(str(t), start, end, progress=False))
            df = RSI(df).dropna()
        except (ValueError, IndexError, KeyError):
            print(f'Insufficient data for {t}')
            continue
        if df is None or len(df) < avg_days:
            print(f'Insufficient data for {t}')
            continue

        adr = (df['High'] - df['Close']).iloc[-avg_days:].mean() / df.loc[df.index[-1], 'Close']
        rsi = df.loc[df.index[-avg_days]:, 'RSI'].mean()
        volume = df['Volume'].mean()
        pair = (adr, t)

        if volume > min_vol and max_rsi >= rsi >= min_rsi:
            if len(top_n) < n_selected:
                insort(top_n, pair)
            elif adr > min(top_n)[0]:
                top_n.pop(0)
                insort(top_n, pair)

    with open(savefile, 'w+') as file:
        for t in top_n:
            file.write(t[1] + '\n')
