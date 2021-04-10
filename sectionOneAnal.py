import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

amazonDaily = pd.read_pickle('./AMZNdaily.pkl')
amazonWeekly = pd.read_pickle('./AMZNweekly.pkl')
amazonMonthly = pd.read_pickle('./AMZNmonthly.pkl')

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

def calcProbabilityDistribution(returns, nPts):
    maxReturn = np.max(returns)+1
    minReturn = np.min(returns)
    nReturns = len(returns)
    
    returnRange = np.linspace(minReturn, maxReturn, num = nPts+1)
    binCounts = np.zeros(nPts)

    for i in range(nPts):
        binMin = returnRange[i]
        binMax = returnRange[i+1]
        print('calculating bin count of bin', i+1)
        for j in range(nReturns):
            # update with np.any
            if(returns[j] >= binMin and returns[j] < binMax):
                binCounts[i] += 1

    print('sum of bin counts, should equal the same as nReturns', np.sum(binCounts), 'with nReturn being', nReturns)
    binProbabilities = binCounts / nReturns
    print('sum of bin counts, ensuring it equals one', np.sum(binProbabilities))

    return binProbabilities, returnRange

def normalDistributionFunction(mean, std, returnValue):
    constTerm = 1/(std * np.sqrt(2 * np.pi))
    returnMinusMeanOverStd = (returnValue - mean) / std
    exponentialTerm = np.exp(-(1/2) * (returnMinusMeanOverStd ** 2))
    return constTerm * exponentialTerm

def calcVolatility(returns, deltaT):
    std = np.std(returns)
    volatility = 1/((deltaT)**(1/2)) * std
    return volatility

dailyReturns()
# weeklyReturns()
# monthlyReturns()
delta_t = len(amazonDaily['Daily Return'])/252
daily_volatility = calcVolatility(amazonDaily['Daily Return'], delta_t)

normalDistribution, returnRange = calcNormalDistribution(amazonDaily['Daily Return'], nPts=1000)
binDistribution, returnRange = calcProbabilityDistribution(amazonDaily['Daily Return'], nPts=1000)

normDistFig = make_subplots(specs=[[{"secondary_y": True}]])
normDistFig.add_trace(
    go.Histogram(x=amazonDaily['Daily Return'], name="Histogram of daily returns", nbinsx=1000), 
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

qqPlotFig = go.Figure(
    data=go.Scatter(x=normalDistribution, y=binDistribution), 
    layout_yaxis_range=[0, 0.15], 
    layout_xaxis_range=[0, 0.15]
    )
qqPlotFig.show()