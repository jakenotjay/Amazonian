import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from datetime import date as dt
import pylab
import scipy.stats as stats

stockname = 'AMZN'

daily = pd.read_pickle('./'+stockname+'daily.pkl')
weekly = pd.read_pickle('./'+stockname+'weekly.pkl')
monthly = pd.read_pickle('./'+stockname+'monthly.pkl')

daily['Dates'] = pd.to_datetime(daily['Dates'])
daily = daily.set_index('Dates')
weekly['Dates'] = pd.to_datetime(weekly['Dates'])
weekly = weekly.set_index('Dates')
monthly['Dates'] = pd.to_datetime(monthly['Dates'])
monthly = monthly.set_index('Dates')

daily['Adj Close'] = pd.to_numeric(daily['Adj Close'])
weekly['Adj Close'] = pd.to_numeric(weekly['Adj Close'])
monthly['Adj Close'] = pd.to_numeric(monthly['Adj Close'])


def dailyReturns():
    dailyReturns = []
    # for day 1 of returns
    dailyReturns.append(0)
    for i in range(1, len(daily)):
        ydayClosingValue = float(daily['Adj Close'][i-1])
        tdayClosingValue = float(daily['Adj Close'][i])
        dailyReturn = ((tdayClosingValue - ydayClosingValue)/ydayClosingValue) * 100
        dailyReturns.append(dailyReturn)

    daily['Daily Return'] = dailyReturns

    fig = px.histogram(
    data_frame=daily, 
    x='Daily Return',
    #nbins=50
    )
    fig.show()

def weeklyReturns():
    weeklyReturns = []
    weeklyReturns.append(0)
    for i in range(1, len(weekly)):
        lastWeekClosingValue = float(weekly['Adj Close'][i-1])
        thisWeekClosingValue = float(weekly['Adj Close'][i])
        weeklyReturn = ((thisWeekClosingValue - lastWeekClosingValue)/lastWeekClosingValue) * 100
        weeklyReturns.append(weeklyReturn)

    weekly['Weekly Return'] = weeklyReturns

    fig = px.histogram(
    data_frame=weekly, 
    x='Weekly Return',
    #nbins=50
    )
    fig.show()

def monthlyReturns():
    monthlyReturns = []
    monthlyReturns.append(0)
    for i in range(1, len(monthly)):
        lastMonthClosingValue = float(monthly['Adj Close'][i-1])
        thisMonthClosingValue = float(monthly['Adj Close'][i])
        monthlyReturn = ((thisMonthClosingValue - lastMonthClosingValue)/lastMonthClosingValue) * 100
        monthlyReturns.append(monthlyReturn)

    monthly['Monthly Return'] = monthlyReturns

    fig = px.histogram(
    data_frame=monthly, 
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
    
    # print('interest rates for time period are\n', interestRates['Rates'])

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

    timePeriodData = daily[startDate: endDate]
    plotReturns(timePeriodData)

    print('With an initial investment of', principal, 'dollars')
    print('Starting from date', startDate, 'to', endDate)

    firstValue = timePeriodData['Adj Close'][0]/100
    print('first value', firstValue, 'dollars')
    lastValue = timePeriodData['Adj Close'][-1]/100
    print('last value', lastValue, 'dollars')

    nStocks = np.floor(principal / firstValue)
    print('Initial purchase of', nStocks, 'stocks at a price of', firstValue, 'dollars')

    finalReturns = lastValue * nStocks
    print('Sold at a price of', lastValue, 'with a return of', finalReturns)

    roi = finalReturns - principal
    roiPerc = ((finalReturns-principal)/principal) * 100

    print('The return on investment is', roi, 'dollars, or a percentage of', roiPerc, '%')

    maxStockPrice = np.max(timePeriodData['Adj Close'])
    minStockPrice = np.min(timePeriodData['Adj Close'])

    maxReturnPrice = (maxStockPrice/100) * nStocks
    minReturnPrice = (minStockPrice/100) * nStocks

    maxRoi = maxReturnPrice - principal
    maxRoiPerc = ((maxReturnPrice-principal)/principal) * 100

    minRoi = minReturnPrice - principal
    minRoiPerc = ((minReturnPrice-principal)/principal) * 100

    print('In the given time period the max stock price is', maxStockPrice, 'cents, the min price is', minStockPrice, 'cents')
    print('In the given time period the maximum return price is', maxReturnPrice, 'dollars, the min return price is', minReturnPrice, 'dollars')
    print('In the given time period the max ROI is', maxRoi, 'dollars, the min ROI is', minRoi, 'dollars')
    print('In the given time period the max ROI is', maxRoiPerc, '%, the min ROI is', minRoiPerc, '%')

    savingsTotals, savingsRoi, savingsRoiPerc = calcSavingsInvestment(principal, startDate, endDate)
    print('In the giving time frame, the total value after the principal investment had been placed in savings accounts would be:')
    print('Savings totals - ', savingsTotals, 'dollars')
    print('ROI for savings would be - ', savingsRoi, 'dollars')
    print('ROI percentage for savings would be - ', savingsRoiPerc, '%')

calcInvestmentStats(1000000, '2019-01-01', '2019-12-31')

dailyReturns()
# # weeklyReturns()
# # monthlyReturns()
# delta_t = len(daily['Daily Return'])/252
# daily_volatility = calcVolatility(daily['Daily Return'], delta_t)
firstValue = daily['Adj Close'][0]
print('first value', firstValue)
lastValue = daily['Adj Close'][-1]
print('last value', lastValue)
calcDriftFrac(firstValue, lastValue)
calcDriftPercentage(firstValue, lastValue)

normalDistribution, returnRange = calcNormalDistribution(daily['Daily Return'], nPts=1000)

normDistFig = make_subplots(specs=[[{"secondary_y": True}]])
normDistFig.add_trace(
    go.Histogram(x=daily['Daily Return'], name="Histogram of daily returns"), 
    secondary_y=False,
    )
normDistFig.add_trace(
    go.Scatter(x=returnRange,y=normalDistribution, mode='lines', name="Normal distribution of daily returns"), 
    secondary_y=True
    )

# Add figure title
normDistFig.update_layout(
    title_text="Daily returns histogram and normal distribution of daily returns"
)

# Set x-axis title
normDistFig.update_xaxes(title_text="Daily returns")

# Set y-axes titles
normDistFig.update_yaxes(title_text="<b>Count</b> of Daily Return", secondary_y=False)
normDistFig.update_yaxes(title_text="<b>Probability</b> of Daily Return", secondary_y=True)

normDistFig.show()

# QQplot of the daily returns against a theoretical normal distribution
stats.probplot(daily['Daily Return'], dist='norm', plot=pylab)
pylab.show()