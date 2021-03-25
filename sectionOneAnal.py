import pandas as pd

amazon = pd.read_pickle('./AMZN.pkl')

def dailyReturns():
    dailyReturns = []
    for i in range(len(amazon)):
        openingValue = float(amazon['Open'][i])
        closingValue = float(amazon['Adj Close'][i])
        dailyReturn = closingValue - openingValue
        dailyReturns.append(dailyReturn)

    amazon['Daily Return'] = dailyReturns

dailyReturns()

print(amazon)