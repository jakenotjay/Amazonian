import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from datetime import date as dt
import pylab
import scipy.stats as stats

amazonDaily = pd.read_pickle('./AMZNdaily.pkl')
amazonWeekly = pd.read_pickle('./AMZNweekly.pkl')
amazonMonthly = pd.read_pickle('./AMZNmonthly.pkl')

amazonDaily['Dates'] = pd.to_datetime(amazonDaily['Dates'])
amazonDaily = amazonDaily.set_index('Dates')
amazonWeekly['Dates'] = pd.to_datetime(amazonWeekly['Dates'])
amazonWeekly = amazonWeekly.set_index('Dates')
amazonMonthly['Dates'] = pd.to_datetime(amazonMonthly['Dates'])
amazonMonthly = amazonMonthly.set_index('Dates')

# amazonDaily = amazonDaily[amazonDaily.index.year == 2020]
amazonDaily['Adj Close'] = pd.to_numeric(amazonDaily['Adj Close'])

print(amazonDaily.dtypes)
def dailyReturns():
    dailyReturns = []
    # for day 1 of returns
    dailyReturns.append(0)
    for i in range(1, len(amazonDaily)):
        ydayClosingValue = float(amazonDaily['Adj Close'][i-1])
        tdayClosingValue = float(amazonDaily['Adj Close'][i])
        dailyReturn = ((tdayClosingValue - ydayClosingValue)/ydayClosingValue) * 100
        dailyReturns.append(dailyReturn)

    amazonDaily['Daily Return'] = dailyReturns

    fig = px.histogram(
    data_frame=amazonDaily, 
    x='Daily Return',
    #nbins=50
    )
    fig.show()

def weeklyReturns():
    weeklyReturns = []
    weeklyReturns.append(0)
    for i in range(1, len(amazonWeekly)):
        lastWeekClosingValue = float(amazonWeekly['Adj Close'][i-1])
        thisWeekClosingValue = float(amazonWeekly['Adj Close'][i])
        weeklyReturn = ((thisWeekClosingValue - lastWeekClosingValue)/lastWeekClosingValue) * 100
        weeklyReturns.append(weeklyReturn)

    amazonWeekly['Weekly Return'] = weeklyReturns

    fig = px.histogram(
    data_frame=amazonWeekly, 
    x='Weekly Return',
    #nbins=50
    )
    fig.show()

def monthlyReturns():
    monthlyReturns = []
    monthlyReturns.append(0)
    for i in range(1, len(amazonMonthly)):
        lastMonthClosingValue = float(amazonMonthly['Adj Close'][i-1])
        thisMonthClosingValue = float(amazonMonthly['Adj Close'][i])
        monthlyReturn = ((thisMonthClosingValue - lastMonthClosingValue)/lastMonthClosingValue) * 100
        monthlyReturns.append(monthlyReturn)

    amazonMonthly['Monthly Return'] = monthlyReturns

    fig = px.histogram(
    data_frame=amazonMonthly, 
    x='Monthly Return',
    #nbins=50
    )
    fig.show()

def calcNormalDistribution(returns, nPts):
    mean = np.mean(returns)
    std = np.std(returns)
    maxReturn = np.max(returns)
    minReturn = np.min(returns)

    returnRange = np.linspace(minReturn, maxReturn, num = nPts)
    normDist = np.zeros(len(returnRange))
    for i in range(len(returnRange)):
        normDist[i] = normalDistributionFunction(mean, std, returnRange[i])

    return normDist, returnRange


def normalDistributionFunction(mean, std, returnValue):
    constTerm = 1/(std * np.sqrt(2 * np.pi))
    returnMinusMeanOverStd = (returnValue - mean) / std
    exponentialTerm = np.exp(-(1/2) * (returnMinusMeanOverStd ** 2))
    return constTerm * exponentialTerm

def calcVolatility(returns, deltaT):
    std = np.std(returns)
    volatility = 1/((deltaT)**(1/2)) * std
    print('volatility calculated as', volatility)
    return volatility

def calcDriftFrac(startPrice, endPrice):
    driftFraction = ((endPrice - startPrice)/startPrice)
    print('drift fraction calculated as', driftFraction)
    return driftFraction

def calcDriftPercentage(startPrice, endPrice):
    driftPercentage = calcDriftFrac(startPrice, endPrice) * 100
    print('drift percentage calculated as', driftPercentage)
    return driftPercentage

def plotReturns(df):
    # Adjusted close price every day
    priceFig = go.Figure(data=go.Scatter(x=df.index, y=df['Adj Close'], mode='lines'))
    priceFig.show()

# principal in dollars, rate in fraction (5% = 0.05), time in years
def calcContinuousCompoundInterest(principal, interestRate, time):
    # print('Calculating compound interest')
    finalAmount = principal * np.exp(interestRate * time)
    # print('calculated final amount from savings interest is:', finalAmount)
    return finalAmount

def findTimeDifferenceInYears(startDate, endDate):
    difference = endDate - startDate
    differenceInYears = (difference.days + difference.seconds/86400.0) / 365.2425
    return differenceInYears

def calcSavingsInvestment(principal, startDate, endDate):
    # monthly interest rates as a percentage
    interestRates = pd.read_pickle('./interestRates.pkl')
    interestRates = interestRates[startDate: endDate]
    
    print('interest rates for time period are\n', interestRates['Rates'])

    time = findTimeDifferenceInYears(startDate, endDate)
    print('time difference is', time)
    interestRate = interestRates['Rates'].mean() / 100
    # print('calculated interest rate is', interestRate)

    totalReturn = calcContinuousCompoundInterest(principal, interestRate, time)
    roi = totalReturn - principal
    roiPerc = (roi/ principal) * 100

    return totalReturn, roi, roiPerc

# principal value in dollars, start date and end date strings
# YYYY-MM-DD format
def calcInvestmentStats(principal, startDate, endDate):
    startDate = dt.fromisoformat(startDate)
    endDate = dt.fromisoformat(endDate)

    timePeriodData = amazonDaily[startDate: endDate]
    plotReturns(timePeriodData)

    print('With an initial investment of', principal, 'dollars')
    print('Starting from date', startDate, 'to', endDate)

    firstValue = timePeriodData['Adj Close'][0]
    print('first value', firstValue)
    lastValue = timePeriodData['Adj Close'][-1]
    print('last value', lastValue)

    nStocks = np.floor(principal / firstValue)
    print('Initial purchase of', nStocks, 'stocks at a price of', firstValue, 'dollars')

    finalReturns = lastValue * nStocks
    print('Sold at a price of', lastValue, 'with a return of', finalReturns)

    roi = finalReturns - principal
    roiPerc = ((finalReturns-principal)/principal) * 100

    print('The return on investment is', roi, 'dollars, or a percentage of', roiPerc, '%')

    maxStockPrice = np.max(timePeriodData['Adj Close'])
    minStockPrice = np.min(timePeriodData['Adj Close'])

    maxReturnPrice = maxStockPrice * nStocks
    minReturnPrice = minStockPrice * nStocks

    maxRoi = maxReturnPrice - principal
    maxRoiPerc = ((maxReturnPrice-principal)/principal) * 100

    minRoi = minReturnPrice - principal
    minRoiPerc = ((minReturnPrice-principal)/principal) * 100

    print('In the given time period the max stock price is', maxStockPrice, 'dollars, the min price is', minStockPrice, 'dollars')
    print('In the given time period the maximum return price is', maxReturnPrice, 'dollars, the min return price is', minReturnPrice, 'dollars')
    print('In the given time period the max ROI is', maxRoi, 'dollars, the min ROI is', minRoi, 'dollars')
    print('In the given time period the max ROI is', maxRoiPerc, '%, the min ROI is', minRoiPerc, '%')

    savingsTotals, savingsRoi, savingsRoiPerc = calcSavingsInvestment(principal, startDate, endDate)
    print('In the giving time frame, the total value after the principal investment had been placed in savings accounts would be:')
    print('Savings totals - ', savingsTotals, 'dollars')
    print('ROI for savings would be - ', savingsRoi, 'dollars')
    print('ROI percentage for savings would be - ', savingsRoiPerc, '%')

calcInvestmentStats(1000000, '2019-01-01', '2019-12-31')

# dailyReturns()
# # weeklyReturns()
# # monthlyReturns()
# delta_t = len(amazonDaily['Daily Return'])/252
# daily_volatility = calcVolatility(amazonDaily['Daily Return'], delta_t)
# firstValue = amazonDaily['Adj Close'][0]
# print('first value', firstValue)
# lastValue = amazonDaily['Adj Close'][-1]
# print('last value', lastValue)
# calcDriftFrac(firstValue, lastValue)
# calcDriftPercentage(firstValue, lastValue)

# normalDistribution, returnRange = calcNormalDistribution(amazonDaily['Daily Return'], nPts=1000)

# normDistFig = make_subplots(specs=[[{"secondary_y": True}]])
# normDistFig.add_trace(
#     go.Histogram(x=amazonDaily['Daily Return'], name="Histogram of daily returns"), 
#     secondary_y=False,
#     )
# normDistFig.add_trace(
#     go.Scatter(x=returnRange,y=normalDistribution, mode='lines', name="Normal distribution of daily returns"), 
#     secondary_y=True
#     )

# # Add figure title
# normDistFig.update_layout(
#     title_text="Daily returns histogram and normal distribution of daily returns"
# )

# # Set x-axis title
# normDistFig.update_xaxes(title_text="Daily returns")

# # Set y-axes titles
# normDistFig.update_yaxes(title_text="<b>Count</b> of Daily Return", secondary_y=False)
# normDistFig.update_yaxes(title_text="<b>Probability</b> of Daily Return", secondary_y=True)

# normDistFig.show()

# # QQplot of the daily returns against a theoretical normal distribution
# stats.probplot(amazonDaily['Daily Return'], dist='norm', plot=pylab)
# pylab.show()