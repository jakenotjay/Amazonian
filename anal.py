import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from datetime import date as dt
import pylab
import scipy.stats as stats
from functools import reduce
import matplotlib.pyplot as plt

stockname = 'AAPL'

daily = pd.read_pickle('./'+stockname+'daily.pkl')
# weekly = pd.read_pickle('./'+stockname+'weekly.pkl')
# monthly = pd.read_pickle('./'+stockname+'monthly.pkl')

# daily['Dates'] = pd.to_datetime(daily['Dates'])
# daily = daily.set_index('Dates')
# weekly['Dates'] = pd.to_datetime(weekly['Dates'])
# weekly = weekly.set_index('Dates')
# monthly['Dates'] = pd.to_datetime(monthly['Dates'])
# monthly = monthly.set_index('Dates')

daily['Adj Close'] = pd.to_numeric(daily['Adj Close'])
# weekly['Adj Close'] = pd.to_numeric(weekly['Adj Close'])
# monthly['Adj Close'] = pd.to_numeric(monthly['Adj Close'])


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

def logDailyReturns():
    logDailyReturns = []
    logDailyReturns.append(0)
    for i in range(1, len(daily)):
        ydayClosingValue = float(daily['Adj Close'][i-1])
        tdayClosingValue = float(daily['Adj Close'][i])
        logDailyReturn = np.log(tdayClosingValue/ydayClosingValue)
        logDailyReturns.append(logDailyReturn)
    
    daily['Log Daily Return'] = logDailyReturns

def weeklyReturns():
    weeklyReturns = []
    weeklyReturns.append(0)
    for i in range(1, len(weekly)):
        lastWeekClosingValue = float(weekly['Adj Close'][i-1])
        thisWeekClosingValue = float(weekly['Adj Close'][i])
        weeklyReturn = ((thisWeekClosingValue - lastWeekClosingValue)/lastWeekClosingValue) * 100
        weeklyReturns.append(weeklyReturn)

    weekly['Weekly Return'] = weeklyReturns

def monthlyReturns():
    monthlyReturns = []
    monthlyReturns.append(0)
    for i in range(1, len(monthly)):
        lastMonthClosingValue = float(monthly['Adj Close'][i-1])
        thisMonthClosingValue = float(monthly['Adj Close'][i])
        monthlyReturn = ((thisMonthClosingValue - lastMonthClosingValue)/lastMonthClosingValue) * 100
        monthlyReturns.append(monthlyReturn)

    monthly['Monthly Return'] = monthlyReturns

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
    print('std calculated as', std)
    print('delta t for volatility', deltaT)
    volatility = (1/np.sqrt(deltaT)) * std
    print('volatility calculated as', volatility)
    return volatility

# incorrect code
def calcDriftFrac(startPrice, endPrice):
    driftFraction = ((endPrice - startPrice)/startPrice)
    print('drift fraction calculated as', driftFraction)
    return driftFraction

# incorrect code
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
    timePeriodData = daily[startDate: endDate]

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

def normaliseData(returns):
    years = np.unique(returns.index.year)
    # print('years',years)
    # print('dtype', years.dtype)
    nYears = len(years)
    normalisedData = []
    for i in range(nYears):
        yearlyAdjClose = returns[returns.index.year == years[i]]['Adj Close']
        maxValue = np.max(yearlyAdjClose)
        normalisedYearly = yearlyAdjClose / maxValue
        for normValue in normalisedYearly.values:
            normalisedData.append(normValue)
    
    returns['Normalised Adj Close'] = normalisedData
    # print('normalised adj close', returns['Normalised Adj Close'])
    # print('adj close', returns['Adj Close'])

    # normalisedPlot = go.Figure()
    # normalisedPlot.add_trace(go.Scatter(x=returns.index[returns.index.year == 2015], y=returns[returns.index.year == 2015]['Normalised Adj Close'], mode='lines'))
    # normalisedPlot.add_trace(go.Scatter(x=returns.index[returns.index.year == 2016], y=returns[returns.index.year == 2016]['Normalised Adj Close'], mode='lines'))
    # normalisedPlot.add_trace(go.Scatter(x=returns.index[returns.index.year == 2017], y=returns[returns.index.year == 2017]['Normalised Adj Close'], mode='lines'))
    # normalisedPlot.add_trace(go.Scatter(x=returns.index[returns.index.year == 2018], y=returns[returns.index.year == 2018]['Normalised Adj Close'], mode='lines'))
    # normalisedPlot.add_trace(go.Scatter(x=returns.index[returns.index.year == 2019], y=returns[returns.index.year == 2019]['Normalised Adj Close'], mode='lines'))
    # normalisedPlot.show()
    
    dates = []
    avgAdjClose = []

    returns['MM-DD'] = returns.index.strftime('%m-%d')
    commonDates = reduce(np.intersect1d, (returns[returns.index.year == 2015]['MM-DD'].values, returns[returns.index.year == 2016]['MM-DD'].values, returns[returns.index.year == 2017]['MM-DD'].values, returns[returns.index.year == 2018]['MM-DD'].values, returns[returns.index.year == 2019]['MM-DD'].values))
    print(commonDates)
    
def produceNormDistFig(returns, returnRange, normalDistribution):
    normDistFig = make_subplots(specs=[[{"secondary_y": True}]])
    normDistFig.add_trace(
        go.Histogram(x=returns, name="Histogram of daily returns"), 
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
    normDistFig.update_xaxes(title_text="Daily returns (%)")

    # Set y-axes titles
    normDistFig.update_yaxes(title_text="<b>Count</b> of Daily Return", secondary_y=False)
    normDistFig.update_yaxes(title_text="<b>Probability</b> of Daily Return", secondary_y=True)

    normDistFig.show()

def produceQQplot(dailyReturns):
    # QQplot of the daily returns against a theoretical normal distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    stats.probplot(dailyReturns, dist='norm', plot=ax)
    ax.set_title("QQplot of the Daily Returns against a theoretical normal distribution")
    plt.show()

def calcAnnualDrift(dailyClose):
    nDays = len(dailyClose)
    totalDailyDrift = 0
    for i in range(1, nDays):
        dailyReturn = dailyClose[i] - dailyClose[i-1]
        dailyDrift = dailyReturn / dailyClose[i-1]
        totalDailyDrift += dailyDrift

    annualDrift = (totalDailyDrift / (nDays - 1)) * 252
    return annualDrift

# predicts future prices using normal distribution - returns as a fraction, start date, end date, deltaT - time to predict, confidence interval
def predictFuturePrice(returns, startDate, endDate, deltaT, confidenceInterval = 2):
    dailyClose = daily[startDate:endDate]['Adj Close']
    firstDayPrice = dailyClose[0]
    lastDayPrice = dailyClose[-1]

    drift = calcAnnualDrift(dailyClose.values)

    deltaS = drift * lastDayPrice * deltaT

    volatility = calcVolatility(returns, deltaT)

    uncertainty = volatility * np.sqrt(deltaT) * lastDayPrice * confidenceInterval

    print('For annual drift of ', drift, 'and volatility', volatility)
    print('At a starting price of', lastDayPrice)
    print('The change in stock price is ', deltaS, '+-', uncertainty, 'within', confidenceInterval, 'confidence intervals')
    print('The predicted stock price is', lastDayPrice + deltaS, '+-', uncertainty, 'within', confidenceInterval, 'confidence intervals')
    print('This gives a range of', (lastDayPrice + deltaS) - uncertainty, 'to', (lastDayPrice + deltaS) + uncertainty)
    return deltaS, uncertainty

def calcLogVolatility(logReturns, samplingRate):
    S = np.std(logReturns)
    volatility = S/np.sqrt(samplingRate)
    print('calculated volatility', volatility, 'over', samplingRate, 'samplingRate')
    return volatility

def calcLogDrift(logReturns, volatility, samplingRate):
    meanLogReturns = np.mean(logReturns)
    drift = (meanLogReturns/samplingRate) + ((volatility ** 2 )/ 2)
    print('calculated drift', drift, 'over', samplingRate, 'samplingRate')
    return drift

def predictFuturePriceAsLog(logReturns, startDate, endDate, deltaT, samplingRate, confidenceInterval = 2):
    lastDayPrice = daily[startDate: endDate]['Adj Close'][-1]
    volatility = calcLogVolatility(logReturns.values, samplingRate)
    drift = calcLogDrift(logReturns.values, volatility, samplingRate)
    
    lnS = (drift - ((volatility ** 2) / 2)) * deltaT + np.log(lastDayPrice)
    lnUncertainty = volatility * confidenceInterval * np.sqrt(deltaT)
    
    lnLowerValue = lnS - lnUncertainty
    lnUpperValue = lnS + lnUncertainty
    print('starting at ', lastDayPrice)
    print('the predicted range is', np.exp(lnLowerValue), 'to', np.exp(lnUpperValue))

def calcLogNormalDistribution(logReturns, nPts):
    mean = np.mean(logReturns)
    std = np.std(logReturns)
    maxLogReturn = np.max(logReturns)
    minLogReturn = np.min(logReturns)
    
    logReturnRange = np.linspace(minLogReturn, maxLogReturn, num=nPts)
    logNormDist = np.zeros(len(logReturnRange))
    for i in range(len(logReturnRange)):
        logNormDist[i] = logNormalDistributionFunction(mean, std, logReturnRange[i])
        
    return logNormDist, logReturnRange
    
def logNormalDistributionFunction(mean, std, logReturn):
    constTerm = 1/(np.sqrt(2 * np.pi) * std)
    returnMinusMean = (logReturn - mean) ** 2 
    negativeOneOverTwoSSquared = - 1 / (2 * (std ** 2))
    return constTerm * np.exp(negativeOneOverTwoSSquared * returnMinusMean)

def produceLogNormDistFig(logReturns, logReturnRange, logNormDist):
    normDistFig = make_subplots(specs=[[{"secondary_y": True}]])
    normDistFig.add_trace(
        go.Histogram(x=logReturns, name="Histogram of daily log returns"), 
        secondary_y=False,
        )
    normDistFig.add_trace(
        go.Scatter(x=logReturnRange,y=logNormDist, mode='lines', name="Normal distribution of daily log returns"), 
        secondary_y=True
        )

    # Add figure title
    normDistFig.update_layout(
        title_text="Daily logReturns histogram and normal distribution of daily log returns"
    )

    # Set x-axis title
    normDistFig.update_xaxes(title_text="Daily log returns")

    # Set y-axes titles
    normDistFig.update_yaxes(title_text="<b>Count</b> of log return", secondary_y=False)
    normDistFig.update_yaxes(title_text="<b>Probability</b> of log return", secondary_y=True)

    normDistFig.show()

def generatePartOneStats(startDate, endDate):
    print('Generating part one stats between', startDate,'and', endDate)
    dailyReturns()
    # weeklyReturns()
    # monthlyReturns()

    # find returns in given time period
    returns = daily[startDate:endDate]['Daily Return']
    returnsFrac = returns / 100

    # stock price in this period is
    plotReturns(daily[startDate: endDate])

    # generate normal distribution for time period and histogram
    normDist, returnRange = calcNormalDistribution(returns, nPts=1000)
    produceNormDistFig(returns, returnRange, normDist)

    # produce QQ plot
    produceQQplot(returns)

    # time in years to predict to    
    predictionTime = 0.25
    deltaS, uncertainty = predictFuturePrice(returnsFrac, startDate, endDate, predictionTime, confidenceInterval = 2)

    # invest principal of 1 mil over same time period
    calcInvestmentStats(1000000, startDate, endDate)

def generatePartTwoStats(startDate, endDate):
    ## Part Two Stats
    # generate log normal distribution stuff
    # obtain drift
    # obtain volatility
    logDailyReturns()
    logReturns = daily[startDate:endDate]['Log Daily Return']
    
    fig = px.histogram(daily[startDate:endDate], x="Log Daily Return")
    fig.show()
    
    samplingRate = 1/252
    
    volatility = calcLogVolatility(logReturns.values, samplingRate)
    drift = calcLogDrift(logReturns.values, volatility, samplingRate)
    print('calculated log drift', drift)
    print('calculated log volatility', volatility)

    # S, uncertainty = predictFuturePriceAsLog(logReturns, startDate, endDate, 0.25, 2)
    predictionTime = 0.25
    predictFuturePriceAsLog(logReturns, startDate, endDate, predictionTime, samplingRate, 2)
    
    logNormDist, logReturnRange = calcLogNormalDistribution(logReturns.values, 1000)
    produceLogNormDistFig(logReturns.values, logReturnRange, logNormDist)

    # using volatility either from part 1 or 2 create a 2-step binomial tree model:
    # create a fair price of a 2 month european long call option
    # create a fair price of a 2 month european long put option

startDate = dt.fromisoformat('2005-01-01')
endDate = dt.fromisoformat('2019-12-31')
generatePartOneStats(startDate, endDate)
generatePartTwoStats(startDate, endDate)

# startDate = dt.fromisoformat('2019-01-01')
# endDate = dt.fromisoformat('2020-12-31')

# returnsData = daily[startDate: endDate]
# trendsData = pd.read_pickle('./AmazonTrendsWorldwide.pkl')
# trendsData['Trends'] = pd.to_numeric(trendsData['Trends'])
# trendsData = trendsData[startDate: endDate]
# facemaskTrendsData = pd.read_pickle('./AmazonTrendsMaskWorldwide.pkl')
# facemaskTrendsData['Trends'] = pd.to_numeric(facemaskTrendsData['Trends'])
# facemaskTrendsData = facemaskTrendsData[startDate: endDate]

# returnsTrendsFig = make_subplots(specs=[[{"secondary_y": True}]])
# returnsTrendsFig.add_trace(
#     go.Scatter(x=returnsData.index, y=returnsData['Adj Close'], mode='lines', name='AMZN Price (USD)'),
#     secondary_y=False
# )
# returnsTrendsFig.add_trace(
#     go.Scatter(x=trendsData.index, y=trendsData['Trends'], mode='lines', name='Amazon Google Trends'),
#     secondary_y=True
# )
# returnsTrendsFig.add_trace(
#     go.Scatter(x=facemaskTrendsData.index, y=facemaskTrendsData['Trends'], mode='lines', name='Amazon facemasks Google Trends'),
#     secondary_y=True
# )
# returnsTrendsFig.show()
