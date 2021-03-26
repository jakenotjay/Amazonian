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
        dailyReturn = tdayClosingValue - ydayClosingValue
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
        weeklyReturn = thisWeekClosingValue - lastWeekClosingValue
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
        monthlyReturn = thisMonthClosingValue - lastMonthClosingValue
        monthlyReturns.append(monthlyReturn)

    amazonMonthly['Monthly Return'] = monthlyReturns

    fig = px.histogram(
    data_frame=amazonMonthly, 
    x='Monthly Return',
    #nbins=50
    )
    fig.show()

def calcNormalDistribution(returns):
    mean = np.mean(returns)
    std = np.std(returns)
    maxReturn = np.max(returns)
    minReturn = np.min(returns)
    nPts = 1000    

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

dailyReturns()
weeklyReturns()
monthlyReturns()

normalDistribution, returnRange = calcNormalDistribution(amazonDaily['Daily Return'])

normDistFig = make_subplots(specs=[[{"secondary_y": True}]])
normDistFig.add_trace(go.Histogram(x=amazonDaily['Daily Return']), secondary_y=False)
normDistFig.add_trace(go.Scatter(x=returnRange,y=normalDistribution, mode='lines'), secondary_y=True)

# Add figure title
normDistFig.update_layout(
    title_text="Double Y Axis Example"
)

# Set x-axis title
normDistFig.update_xaxes(title_text="xaxis title")

# Set y-axes titles
normDistFig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
normDistFig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

normDistFig.show()