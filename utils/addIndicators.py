import datetime as dt
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None


# returns an exponential moving average
def EMA(df, period=20, observeCol='Adj Close', colName=None):
    if len(df) < period:
        raise ValueError('Not enough data')
    if not (type(df) == pd.DataFrame and type(period) == int):
        raise TypeError
    colName = "EMA_{}".format(period) if colName is None else colName

    df[colName] = df[observeCol].ewm(span=period, adjust=False).mean()
    return df


# returns a simple moving average
def SMA(df, period=20, colName=None):
    if len(df) < period:
        raise ValueError('Not enough data')

    if not (type(df) == pd.DataFrame and type(period) == int):
        raise TypeError
    colName = 'SMA_{}'.format(period) if colName is None else colName

    df[colName] = TP(df)['TP'].rolling(window=period).mean()
    del df['TP']
    return df


# returns a stock's daily typical price
def TP(df, colName='TP'):
    if not type(df) == pd.DataFrame:
        raise TypeError

    df[colName] = (df['High'] + df['Low'] + df['Close']) / 3
    return df


# returns the average true range volatility indicator
def ATR(df, period=14, colName='ATR'):
    if len(df) < period:
        raise ValueError('Not enough data')

    if not (type(df) == pd.DataFrame and type(period) == int):
        raise TypeError

    if colName in df.columns:
        return df

    tr = {'TR': []}
    for n in range(len(df.index)):
        if n == 0:
            continue

        tr['TR'].append(max(df['High'][n] - df['Low'][n], abs(df['High']
                                                              [n] - df['Close'][n - 1]), abs(df['Low'][n] - df['Close'][n - 1])))

    df = df[1:]
    tr = pd.DataFrame(data=tr, index=df.index)
    tr[colName] = tr['TR'].rolling(window=period).mean()
    del tr['TR']
    return pd.concat([df, tr], axis=1)[period:]


# the keltner channels don't line up 100% perfectly with MarketWatch's keltner's of same params
def Keltner(df, emaPeriod=20, atrPeriod=20, atrMult=2, colNameUpper='Kelt Upper', colNameLower='Kelt Lower'):
    if len(df) < max([emaPeriod, atrPeriod]):
        raise ValueError('Not enough data')
    if not (type(df) == pd.DataFrame and type(emaPeriod) == int and type(atrPeriod) == int and type(atrMult) == int):
        raise TypeError

    req = EMA(df, emaPeriod, colName='EMA_{} kelt'.format(emaPeriod))
    req['ATR kelt'] = ATR(df, atrPeriod, colName='ATR kelt')['ATR kelt']
    df[colNameUpper] = req['EMA_{} kelt'.format(
        emaPeriod)] + req['ATR kelt'] * atrMult
    df[colNameLower] = req['EMA_{} kelt'.format(
        emaPeriod)] - req['ATR kelt'] * atrMult
    del req['ATR kelt']
    del df['EMA_{} kelt'.format(emaPeriod)]
    return df


# the bollinger bands don't line up 100% perfectly with MarketWatch's bands of same params
def Bollinger(df, period=20, nDeviations=2, colNameUpper='Bol Upper', colNameLower='Bol Lower'):
    if len(df) < period:
        raise ValueError('Not enough data')

    if not (type(df) == pd.DataFrame and type(period) == int and type(nDeviations) == int):
        raise TypeError

    req = SMA(df, period, colName='SMA_{} bol'.format(period))
    req['STD bol'] = TP(df, colName='TP bol')[
        'TP bol'].rolling(window=period).std()

    df[colNameUpper] = req['SMA_{} bol'.format(
        period)] + nDeviations * req['STD bol']
    df[colNameLower] = req['SMA_{} bol'.format(
        period)] - nDeviations * req['STD bol']
    del req['STD bol']
    del df['SMA_{} bol'.format(period)]
    del df['TP bol']
    return df


# returns whether the stock is currently in a TTM squeeze (bollinger bands are between keltner channels)
def TTMSqueeze(df, emaPeriod=20, atrPeriod=20, atrMult=2, bollingerPeriod=20, nDeviations=2, colName='TTMSqueeze'):
    if len(df) < max([emaPeriod, atrPeriod, bollingerPeriod]):
        raise ValueError('Not enough data')

    if not type(df) == pd.DataFrame or any(type(x) != int for x in [emaPeriod, atrPeriod, atrMult, bollingerPeriod, nDeviations]):
        raise TypeError

    req = Keltner(df, emaPeriod, atrPeriod, atrMult,
                  colNameUpper='Kelt Upper ttm', colNameLower='Kelt Lower ttm')
    req[['Bol Upper ttm', 'Bol Lower ttm']] = Bollinger(df, bollingerPeriod, nDeviations, colNameUpper='Bol Upper ttm', colNameLower='Bol Lower ttm')[
        ['Bol Upper ttm', 'Bol Lower ttm']]

    df[colName] = (req['Bol Lower ttm'] >= req['Kelt Lower ttm']) & (
        req['Bol Upper ttm'] <= req['Kelt Upper ttm'])
    del df['Kelt Upper ttm'], df['Kelt Lower ttm'], df['Bol Upper ttm'], df['Bol Lower ttm']
    return df


# returns the relative strength index (technical indicator)
def RSI(df, period=14, colName='RSI'):
    if len(df) < period:
        raise ValueError('Not enough data')

    if not (type(df) == pd.DataFrame and type(period) == int):
        raise TypeError

    df['Gain'] = -(df['Adj Close'].rolling(window=2).sum() -
                   2 * df['Adj Close'])
    df['AvgGain'] = np.nan
    df['AvgLoss'] = np.nan
    df[colName] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        temp = df.iloc[:period]
        gainFilter = temp['Gain'] > 0
        lossFilter = temp['Gain'] < 0
        df['AvgGain'][period - 1] = temp[gainFilter]['Gain'].sum() / period
        df['AvgLoss'][period - 1] = temp[lossFilter]['Gain'].sum() / period
        df[colName][period - 1] = 100 - 100 / \
            (1 + df['AvgGain'][period - 1] / -df['AvgLoss'][period - 1])

        for n in range(len(df.index)):
            if n < period:
                continue
            df['AvgGain'][n] = (df['AvgGain'][n - 1] * (period - 1) +
                                (df['Gain'][n] if df['Gain'][n] > 0 else 0)) / period
            df['AvgLoss'][n] = (df['AvgLoss'][n - 1] * (period - 1) +
                                (df['Gain'][n] if df['Gain'][n] < 0 else 0)) / period
            df[colName][n] = 100 - 100 / (1 + df['AvgGain'][n] / -df['AvgLoss'][n])

    df = df[period - 1:]
    del df['Gain']
    del df['AvgGain']
    del df['AvgLoss']
    del temp
    return df


# returns the relative strength of a stock, which is how well it's performing relative to 63, 126, 189, and 252 days ago
def IBD_RS(df, colName='RS'):
    if len(df) < 252:
        raise ValueError('Not enough data')
    if type(df) != pd.DataFrame:
        raise TypeError

    df[colName] = np.zeros
    for n, i in enumerate(df.index):
        if n < 252:
            continue
        c = df['Adj Close'][n]
        c63 = df['Adj Close'][n - 63]
        c126 = df['Adj Close'][n - 126]
        c189 = df['Adj Close'][n - 189]
        c252 = df['Adj Close'][n - 252]
        df[colName][i] = 2 * c / c63 + c / c126 + c / c189 + c / c252
    return df


# returns the relative strength index of a stock according to IBD (ranks stocks 1-100 based on the above function's results)
def IBD_RelativeStrength(df, ticker, path, smoothing=True, smoothingFactor=5, colName='IBD RS'):
    if type(df) != pd.DataFrame or any(type(x) != str for x in [ticker, colName, path]):
        raise TypeError

    ticker = ticker.upper()
    df[colName] = np.nan

    for i in df.index:
        prev = df[colName][df.index.get_loc(i) - 1]
        try:
            loaded = pd.read_csv(
                '{}/IBD RS {}.csv'.format(path, dt.date(i.year, i.month, i.day)))
            loaded = loaded.loc[loaded['Ticker'] == ticker, 'RS']
            df[colName][i] = loaded if not smoothing or math.isnan(prev) else int(
                (prev * (smoothingFactor - 1) + loaded) / smoothingFactor)
        except Exception:
            df[colName][i] = prev
    return df


# returns whether the stock is the highest it's been in the specified number of days
def recentHigh(df, days=260, colName=None):
    if len(df) < days:
        raise ValueError('Not enough data')

    colName = 'High_{}'.format(days) if colName is None else colName
    df[colName] = ''
    for i in df.index[1:]:
        index = df.index.get_loc(i)
        df[colName][i] = df['Adj Close'][index] == max(
            df['Adj Close'][0:index]) if index < days else df['Adj Close'][index] == max(df['Adj Close'][index - days + 1:index + 1])
    return df


# returns whether the stock is the lowest it's been in the specified number of days
def recentLow(df, days=260, colName=None):
    if len(df) < days:
        raise ValueError('Not enough data')

    colName = 'Low_{}'.format(days) if colName is None else colName
    df[colName] = ''
    for i in df.index[1:]:
        index = df.index.get_loc(i)
        df[colName][i] = df['Adj Close'][index] == min(
            df['Adj Close'][0:index]) if index < days else df['Adj Close'][index] == min(df['Adj Close'][index - days + 1:index + 1])
    return df


# returns True while the short-emas are above the long-emas
def shortOverLong(df, shortEmas=[8], longEmas=[21], withTriggers=False, colName=None):
    if type(shortEmas) != list:
        shortEmas = [shortEmas]
    if type(longEmas) != list:
        longEmas = [longEmas]

    if len(df) < max(shortEmas + longEmas):
        raise ValueError('Not enough data')

    colName = 'SoL'if colName is None else colName

    shortLabels = []
    longLabels = []
    for p in shortEmas:
        label = 'EMA_{}_SoL'.format(p)
        df = EMA(df, p, colName=label)
        shortLabels.append(label)
    for p in longEmas:
        label = 'EMA_{}_SoL'.format(p)
        df = EMA(df, p, colName=label)
        longLabels.append(label)

    df['longmax'] = df[longLabels].max(axis=1)
    df['shortmin'] = df[shortLabels].min(axis=1)
    df[colName] = df['shortmin'] > df['longmax']

    if withTriggers:
        df = goldenCross(df, shortEmas, longEmas, colName=colName + '_Gold')
        df = deathCross(df, shortEmas, longEmas, colName=colName + '_Death')

    try:
        for label in shortLabels + longLabels:
            del df[label]
        del df['longmax'], df['shortmin']
    except KeyError:
        pass
    return df


# identifies when emas crossover; cross can either be golden (short passes long) or death (long passes short)
def _crossover(df, crossType, shortEmas, longEmas, colName):
    if len(df) < max(shortEmas + longEmas):
        raise ValueError('Not enough data')

    if type(crossType) != str:
        raise ValueError
    elif crossType.lower() == 'golden':
        golden = True
    elif crossType.lower() == 'death':
        golden = False
    else:
        raise ValueError

    df = shortOverLong(df, shortEmas, longEmas, colName='SoL_cr')
    for i in range(1, len(df)):
        date = df.index[i]
        prev = df.index[i - 1]
        df.loc[date, colName] = df.loc[date, 'SoL_cr'] and not df.loc[prev,
                                                                      'SoL_cr'] if golden else not df.loc[date, 'SoL_cr'] and df.loc[prev, 'SoL_cr']
    del df['SoL_cr']
    return df


def goldenCross(df, shortEmas=[12], longEmas=[26], colName=None):
    return _crossover(df, 'Golden', shortEmas, longEmas, 'Golden' if colName is None else colName)


def deathCross(df, shortEmas=[12], longEmas=[26], colName=None):
    return _crossover(df, 'Death', shortEmas, longEmas, 'Death' if colName is None else colName)


# selltrigger based on RSI trends indicating the end of an uptrend. works best if the data given begins well before
# the time period to predict so the downtrend or uptrend can be identified
def failureSwings(df, rsiPeriod=14, initialUptrend=None, colName=None):
    colName = 'FS Uptrend?' if colName is None else colName
    df = RSI(df, rsiPeriod, 'RSI_FS')
    df[colName] = ''

    df = observeTrend(df, observeCol='RSI_FS',
                      initialUptrend=initialUptrend, colName=colName)
    del df['RSI_FS']
    return df


def MACD(df, withHistogram=True, withTriggers=False, withTrends=False, colName=None):
    colName = 'MACD' if colName is None else colName

    df = EMA(df, 12, colName='EMA_12 MACD')
    df = EMA(df, 26, colName='EMA_26 MACD')

    df[colName] = df['EMA_12 MACD'] - df['EMA_26 MACD']
    df[colName + '_Sig'] = df[colName].ewm(span=9, adjust=False).mean()
    df[colName + '_Hist'] = df[colName] - df[colName + '_Sig']
    if withTrends:
        df[colName + ' Uptrend?'] = df[colName + '_Hist'] > 0

    if withTriggers:
        for i in range(1, len(df)):
            if withTriggers:
                date = df.index[i]
                prev = df.index[i - 1]
                df.loc[date, colName + '_Gold'] = df.loc[date, colName +
                                                         '_Hist'] >= 0 and df.loc[prev, colName + '_Hist'] < 0
                df.loc[date, colName + '_Death'] = df.loc[date, colName +
                                                          '_Hist'] <= 0 and df.loc[prev, colName + '_Hist'] > 0

    if not withHistogram:
        del df[colName + '_Hist']
    del df['EMA_12 MACD'], df['EMA_26 MACD']
    return df


def observeTrend(df, observeCol, initialUptrend=None, trendlines=False, colName=None, lineColName=None):
    colName = (observeCol + ' Uptrend?') if colName is None else colName
    lineColName = (
        colName + ' Trendline') if lineColName is None else lineColName

    df[colName] = False
    df = shortOverLong(df, 12, 26, colName='SoL_OBS')
    if trendlines:
        df[lineColName] = np.nan
        reg = LinearRegression()

    # false if not given an initial trend
    trendIdentified = initialUptrend is not None
    # in a downtrend, "peak" really means "trough"
    first_peak = df[observeCol][0]
    second_peak = first_peak
    fail_point = first_peak
    uptrend = initialUptrend if initialUptrend is not None else df['SoL_OBS'][0]
    downtrend = not uptrend
    plotted_peaks = []

    for i in range(1, len(df)):
        obs = df[observeCol][i]

        if uptrend and obs > first_peak or downtrend and obs < first_peak:
            first_peak = obs
            second_peak = obs
            fail_point = obs
        elif uptrend and obs <= fail_point and second_peak < first_peak:
            uptrend = False
            downtrend = True
            trendIdentified = True
            plotted_peaks = []
        elif downtrend and obs >= fail_point and second_peak > first_peak:
            uptrend = True
            downtrend = False
            trendIdentified = True
            plotted_peaks = []
        elif uptrend and obs <= fail_point or downtrend and obs >= fail_point:
            fail_point = obs
        else:
            second_peak = obs
            if trendlines:
                plotted_peaks.append(first_peak)

        if trendIdentified:
            df[colName][i] = bool(uptrend)
            if trendlines and len(plotted_peaks) > 0:
                indices = [i for i in range(len(plotted_peaks))]
                reg.fit(np.array(indices).reshape(-1, 1),
                        np.array(plotted_peaks).reshape(-1, 1))
                df[lineColName][i] = reg.predict(
                    np.array(indices[-1]).reshape(1, -1))[0][0]
            elif trendlines:
                try:
                    df[lineColName][i] = df[lineColName][i - 1]
                except IndexError:
                    pass

    del df['SoL_OBS']
    return df


def volumeForce(df, with_temp=True, colName='VF'):
    if len(df) < 3:
        raise ValueError('Not enough data')
    df[colName] = np.nan
    df['T'] = np.nan
    df['dm'] = np.nan
    df['cm'] = np.nan

    j = df.index[0]
    i = df.index[1]
    h = df.index[2]
    df.loc[i, 'T'] = 1 if (df.loc[i, 'High'] + df.loc[i, 'Low'] + df.loc[i, 'Close']) > (
        df.loc[j, 'High'] + df.loc[j, 'Low'] + df.loc[j, 'Close']) else -1
    df.loc[i, 'dm'] = df.loc[i, 'High'] - df.loc[i, 'Low']
    # in first calculation (at index[2]), uses that day's dm instead of previous day's cm, so set previous day cm to that day's dm here
    df.loc[i, 'cm'] = df.loc[h, 'High'] - df.loc[h, 'Low']
    j = i

    for i in df.index[2:]:
        df.loc[i, 'T'] = 1 if (df.loc[i, 'High'] + df.loc[i, 'Low'] + df.loc[i, 'Close']) > (
            df.loc[j, 'High'] + df.loc[j, 'Low'] + df.loc[j, 'Close']) else -1
        if with_temp:
            df.loc[i, 'dm'] = df.loc[i, 'High'] - df.loc[i, 'Low']
            df.loc[i, 'cm'] = (df.loc[j, 'cm'] + df.loc[i, 'dm']) if df.loc[i,
                                                                            'T'] == df.loc[j, 'T'] else (df.loc[i, 'dm'] + df.loc[j, 'dm'])
            df.loc[i, colName] = df.loc[i, 'Volume'] * \
                ((2 * (df.loc[i, 'dm'] / df.loc[i, 'cm'] - 1))
                 if df.loc[i, 'cm'] != 0 else -2) * df.loc[i, 'T'] * 100
        else:
            df.loc[i, colName] = df.loc[i, 'Volume'] * df.loc[i, 'T'] * 100
        j = i

    del df['T'], df['dm'], df['cm']
    return df


def klinger(df, with_temp=True):
    df = volumeForce(df, with_temp=with_temp, colName='VF_Klinger')
    df = EMA(df, period=34, observeCol='VF_Klinger', colName='klinger34')
    df = EMA(df, period=55, observeCol='VF_Klinger', colName='klinger55')

    df['KO'] = df['klinger34'] - df['klinger55']
    df = EMA(df, period=13, observeCol='KO', colName='KO Sig')
    df['KO Divg'] = df['KO'] - df['KO Sig']

    del df['VF_Klinger'], df['klinger34'], df['klinger55']
    return df


def shiftedSqueeze(df, maxPercent=0.01, emaPeriod=20, atrPeriod=20, atrMult=2, bollingerPeriod=20, nDeviations=2, colName='Squeeze'):
    if len(df) < max([emaPeriod, atrPeriod, bollingerPeriod]):
        raise ValueError('Not enough data')

    if not type(df) == pd.DataFrame or any(type(x) != int for x in [emaPeriod, atrPeriod, atrMult, bollingerPeriod, nDeviations]):
        raise TypeError

    req = Keltner(df, emaPeriod, atrPeriod, atrMult,
                  colNameUpper='Kelt Upper sq', colNameLower='Kelt Lower sq')
    req[['Bol Upper sq', 'Bol Lower sq']] = Bollinger(df, bollingerPeriod, nDeviations, colNameUpper='Bol Upper sq', colNameLower='Bol Lower sq')[
        ['Bol Upper sq', 'Bol Lower sq']]

    # if the bollinger belts are closer together than the keltner channels or within a certain % of the keltner's tightness, identifies squeeze
    df[colName] = ((req['Bol Upper sq'] - req['Bol Lower sq']) - (req['Kelt Upper sq'] -
                                                                  req['Kelt Lower sq'])) / (req['Bol Upper sq'] - req['Bol Lower sq']) <= maxPercent
    del df['Kelt Upper sq'], df['Kelt Lower sq'], df['Bol Upper sq'], df['Bol Lower sq']
    return df


def upwardsChannel(df, maxPercent=0.01, emaPeriod=20, atrPeriod=20, atrMult=2, bollingerPeriod=20, nDeviations=2, colName='Up Channel'):
    if len(df) < max([emaPeriod, atrPeriod, bollingerPeriod]):
        raise ValueError('Not enough data')

    if not type(df) == pd.DataFrame or any(type(x) != int for x in [emaPeriod, atrPeriod, atrMult, bollingerPeriod, nDeviations]):
        raise TypeError

    df = shiftedSqueeze(df, maxPercent=maxPercent, emaPeriod=emaPeriod, atrPeriod=atrPeriod,
                        atrMult=atrMult, bollingerPeriod=bollingerPeriod, nDeviations=nDeviations, colName='sq ch')
    df = EMA(df, emaPeriod, colName='ema ch')

    df[colName] = False
    in_squeeze = False
    for i in df.index:
        # squeeze starts
        if not in_squeeze and df.loc[i, 'sq ch']:
            in_squeeze = True
            start = i
            prev = i
        # while still in squeeze, checks if channel is increasing or decreasing. considers squeeze to be over if it's the last day
        elif in_squeeze and df.loc[i, 'sq ch'] and i != df.index[-1]:
            prev = i
        # when squeeze is over, checks if channel increased more than decreased
        # if in squeeze in the last day of the df, checks if the channel was pointed upwards and if so, also includes the last day in the upwards channel
        elif in_squeeze and df.loc[i, 'sq ch']:
            if df.loc[i, 'ema ch'] > df.loc[start, 'ema ch']:
                df.loc[start:i, colName] = True
            else:
                df.loc[start:prev, colName] = False
        elif in_squeeze:
            if df.loc[prev, 'ema ch'] > df.loc[start, 'ema ch']:
                df.loc[start:prev, colName] = True
            else:
                df.loc[start:prev, colName] = False
            in_squeeze = False

    del df['ema ch'], df['sq ch']
    return df


# makes hammer or hanging man indicators (bullish=True for hammer) to indicate trend reversals. if inverted, max_body_distance is from low instead of high
def hammer(df, trend_days=3, max_real_body_ratio=0.5, max_body_distance=0.2, bullish=True, inverted=False, colName=None):
    if not 0 < max_real_body_ratio <= 0.5 or not 0 < max_body_distance < 1 or trend_days < 0:
        raise ValueError

    colName = colName if colName is not None else 'Hammer' if bullish else 'Hanging'

    df[colName] = False
    for c, i in enumerate(df.index):
        # checks if the last 'trend_days' days of candles are all decreasing; if not, it is not a hammer
        break_flag = True
        if c > trend_days:
            for n in range(trend_days):
                break_flag = True
                if bullish and df.loc[df.index[c - n - 1], 'Close'] >= df.loc[df.index[c - n - 2], 'Close']:
                    break
                if not bullish and df.loc[df.index[c - n - 1], 'Close'] <= df.loc[df.index[c - n - 2], 'Close']:
                    break
                break_flag = False
        if break_flag:
            continue               

        opening = df.loc[i, 'Open']
        close = df.loc[i, 'Close']
        high = df.loc[i, 'High']
        low = df.loc[i, 'Low']
        # checks if the real body size (dif. b/w open and close) is 50% (body ratio) smaller than the high-low range
        # and whether the distance between the high and the top of the real body (bigger of open or close) is less than 
        # 20% (max distance) of the high-low range. if inverted, checks distance between lower of open/close and the shadow low
        df.loc[i, colName] = abs(opening - close) <= ((high - low) * max_real_body_ratio) and ((
            high - max(opening, close)) < ((high - low) * max_body_distance)) if not inverted else ((
            min(opening, close)) - low < ((high - low) * max_body_distance))
    return df
