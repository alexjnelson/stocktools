import yfinance as yf

df = yf.download('aapl')
print(df.loc[df.index[-1], 'Adj Close'])

print(type(1) is bool)