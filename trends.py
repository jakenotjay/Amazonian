import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def createDF(filename, columnName):
    df = pd.read_csv(filename)
    df['Week'] = pd.to_datetime(df['Week'])
    df = df.set_index('Week')
    df.columns = [columnName]
    return df

faceShield = createDF('./data/trends/faceShield.csv', 'Face Shield')
handSanitiser = createDF('./data/trends/handSanitiser.csv', 'Hand Sanitiser')
faceMasks = createDF('./data/trends/faceMasks.csv', 'Face Masks')
antiBac = createDF('./data/trends/antiBac.csv', 'Anti Bacterial')

healthDF = [faceShield, handSanitiser, faceMasks, antiBac]
healthDF = pd.concat(healthDF, axis=1)
healthDF['Mean'] = healthDF.mean(axis=1)
print(healthDF)

deskChair = createDF('./data/trends/deskChair.csv', 'Desk Chair')
desk = createDF('./data/trends/desk.csv', 'Desk')
laptop = createDF('./data/trends/laptop.csv', 'Laptop')
webcam = createDF('./data/trends/webcam.csv', 'Webcam')

officeDF = [deskChair, desk, laptop, webcam]
officeDF = pd.concat(officeDF, axis=1)
officeDF['Mean'] = officeDF.mean(axis=1)
print(officeDF)

yogaMat = createDF('./data/trends/yogaMat.csv', 'Yoga Mat')
skippingRope = createDF('./data/trends/skippingRope.csv', 'Skipping Rope')
kettlebell = createDF('./data/trends/kettlebell.csv', 'Kettlebell')
dumbbells = createDF('./data/trends/dumbbells.csv', 'Dumbbells')

fitnessDF = [yogaMat, skippingRope, kettlebell, dumbbells]
fitnessDF = pd.concat(fitnessDF, axis=1)
fitnessDF['Mean'] = fitnessDF.mean(axis=1)
print(fitnessDF)

amazon = createDF('./data/trends/amazon.csv', 'Amazon')
print(amazon)

combinedDF = pd.DataFrame()
combinedDF['Weeks'] = amazon.index
combinedDF['Health'] = healthDF['Mean'].values
combinedDF['Office'] = officeDF['Mean'].values
combinedDF['Fitness'] = fitnessDF['Mean'].values 
combinedDF['Amazon'] = amazon['Amazon'].values

fig = go.Figure()
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Health'], mode="lines", name="Health Products", line_shape='spline'))
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Fitness'], mode="lines", name="Fitness Products", line_shape='spline', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Office'], mode="lines", name="Office Products", line_shape='spline', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Amazon'], mode="lines", name="Amazon", line_shape='spline', line=dict(dash='dashdot')))

# Update title and height
fig.show()

fig = make_subplots(rows = 2, cols = 2, subplot_titles=("Covid-19 Product Search Trends", "Fitness Product Search Trends", "Office Product Search Trends", "Amazon Search Trends"))
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Health'], mode="lines", name="Health Products", line_shape='spline'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Fitness'], mode="lines", name="Fitness Products", line_shape='spline'),
              row=1, col=2)
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Office'], mode="lines", name="Office Products", line_shape='spline'),
              row=2, col=1)
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Amazon'], mode="lines", name="Amazon", line_shape='spline'),
              row=2, col=2)

# Update xaxis properties
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Relative Trend", row=1, col=1)
fig.update_yaxes(title_text="Relative Trend", row=1, col=2)
fig.update_yaxes(title_text="Relative Trend", row=2, col=1)
fig.update_yaxes(title_text="Relative Trend", row=2, col=2)

# Update title and height
fig.update_layout(showlegend=False)
fig.show()

fig = make_subplots(rows = 4, cols = 1, subplot_titles=("Covid-19 Product Search Trends", "Fitness Product Search Trends", "Office Product Search Trends", "Amazon Search Trends"))
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Health'], mode="lines", name="Health Products", line_shape='spline'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Fitness'], mode="lines", name="Fitness Products", line_shape='spline'),
              row=2, col=1)
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Office'], mode="lines", name="Office Products", line_shape='spline'),
              row=3, col=1)
fig.add_trace(go.Scatter(x=combinedDF['Weeks'], y=combinedDF['Amazon'], mode="lines", name="Amazon", line_shape='spline'),
              row=4, col=1)

# Update xaxis properties
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_xaxes(title_text="Date", row=4, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Relative Trend", range=[0, 100], row=1, col=1)
fig.update_yaxes(title_text="Relative Trend", range=[0, 100], row=2, col=1)
fig.update_yaxes(title_text="Relative Trend", range=[0, 100], row=3, col=1)
fig.update_yaxes(title_text="Relative Trend", range=[0, 100], row=4, col=1)

# Update title and height
fig.update_layout(showlegend=False)
fig.show()